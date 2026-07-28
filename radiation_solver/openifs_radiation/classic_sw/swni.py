"""Port of swni.F90 — shortwave solver for spectral bands 4-6 (near-IR).

Fortran reference:
  openifs-48r1/ifs-source/arpifs/phys_radi/swni.F90

swni is called by sw.F90 for JNU=4..6 (NSW=6). It differs fundamentally from
sw1s: instead of accumulating gas amounts through SWTT1+SWUVO3, it uses an
inverse-Laplace grey-gas absorption approach via PAKI (grey absorption
coefficients from swu), with a JABS=1,2 double loop that re-runs the cloud
coupling (swde) for each absorber.

Algorithm (NSW=6 production path, ECMWF NOVLP=1):
  1. Rayleigh + swclr (clear adding) + swr (cloud adding).
  2. JABS=1,2 loop: re-couple cloud with gas absorption (PAKI), recompute
     ZREFZ/ZTR via swde, build ZRJ/ZRK pseudo-fluxes for JN=1..4.
  3. Invert grey pseudo-fluxes: ZRJ(JAJ)-ZRJ(JAJ+1) clamped, then inverse
     Laplace via PAKI -> effective absorber amounts ZW2, SWTT1 -> ZRL.
  4. Cloudy fluxes: PFDOWN = ZRJ(:,1)*ZRL(:,1)*ZRL(:,3) + ZRJ(:,2)*ZRL(:,2)*ZRL(:,4).
  5. Clear-sky fluxes via SWTT1 (6 absorbers) + H2O continuum (RSWCE/RSWCP=0
     for NSW=6, so ZR4=1).
  6. Final fluxes: apply O3 + H2O continuum (SWTT IABSL=3) to combine cloudy
     and clear: PFDOWN = ((1-PCLEAR)*ZR1*ZR4*PFDOWN + PCLEAR*ZFD)*RSUN.

Bottom-up internal convention; PREFZ index 1 = surface.

This version has been rewritten to be autograd-compatible: no in-place index
assignments or in-place += on tensors that participate in the computation
graph. All per-level fills are collected in Python lists and assembled with
torch.stack; all accumulators use functional ``x = x + delta`` updates.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path

from .swclr import swclr as _swclr
from .swr import swr as _swr
from .swtt import swtt1 as _swtt1, swtt as _swtt
from .swde import swde as _swde

_HERE = Path(__file__).resolve().parent
_TABLE_DIR = _HERE.parent / "rrtm_lw" / "tables"
_tables: dict = {}
_dev_cache: dict = {}


def _get(key, dev):
    ck = (key, str(dev))
    t = _dev_cache.get(ck)
    if t is None:
        t = _tables[key].to(dev)
        _dev_cache[ck] = t
    return t
_REPLOG = 1.0e-12
_REPSC = 1.0e-12
_REPSCQ = 1.0e-12


def _load():
    if not _tables:
        _tables["rswce"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_rswce.npy"))).to(torch.float64)
        _tables["rswcp"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_rswcp.npy"))).to(torch.float64)


def swni(knu: int, paer_td: torch.Tensor, paki: torch.Tensor,
         palbp: torch.Tensor, pcg: torch.Tensor, pcld_bu: torch.Tensor,
         pclear: torch.Tensor, pdsig_td: torch.Tensor, pomega: torch.Tensor,
         poz_td: torch.Tensor, prmu: torch.Tensor, psec: torch.Tensor,
         ptau: torch.Tensor, pud: torch.Tensor, pwv_td: torch.Tensor,
         pqs_td: torch.Tensor, rray: torch.Tensor, rsun: torch.Tensor) -> dict:
    """SWNI solver for one near-IR band — port of swni.F90 (NSW=6).

    Args mirror sw1s plus paki (nlon,2,6), pwv_td/pqs_td (nlon,klev top-down).
    pud is per-layer top-down (index 0=TOA boundary). pcld_bu is the effective
    cloud fraction from swu (caller flips to bottom-up for swr).

    Returns dict: pfdown, pfup, pcdown, pcup (nlon,klev+1); psudu2 (nlon,).
    """
    _load()
    nlon = psec.shape[0]
    klev = pud.shape[2] - 1
    nlev1 = klev + 1
    dt = psec.dtype
    dev = psec.device
    ib = knu - 1
    rsun_k = rsun[ib]
    rswce_k = _get("rswce", dev)[ib]
    rswcp_k = _get("rswcp", dev)[ib]

    # Rayleigh ZRAYL (swni uses ZRMUM1 = 1-PRMU form, sw1s uses PRMU form).
    r = rray[ib]
    zrmum1 = 1.0 - prmu
    zrayl = r[5] + torch.zeros_like(zrmum1)
    for k in range(4, -1, -1):
        zrayl = r[k] + zrmum1 * zrayl
    zrayl = zrayl.clamp(min=0.0)

    # ---- 1. swclr + swr ----
    clr = _swclr(knu, 1, paer_td, palbp, pdsig_td, zrayl, psec, rray, prmu)
    pcld_for_swr = pcld_bu.flip(dims=[1])
    cld = _swr(knu, palbp, pcg, pcld_for_swr, pomega, psec, ptau,
               clr["pcgaz"], clr["ppizaz"], clr["ptauz"])
    zrj0 = clr["prj"]; zrk0 = clr["prk"]; zrmu0 = clr["prmu0"]; ztrclr = clr["ptrclr"]
    # swr outputs: pray1/pray2/prefz/ptra1/ptra2 (we re-derive ZRJ/ZRK below).
    zcgaz = clr["pcgaz"]; zpzaz = clr["ppizaz"]; ztauz = clr["ptauz"]
    zray1 = cld["pray1"]; zray2 = cld["pray2"]; zrefz = cld["prefz"]
    ztra1 = cld["ptra1"]; ztra2 = cld["ptra2"]; ztrcld = cld["ptrcld"]
    zrmue = cld["prmue"]

    # palbp for this band
    palbp_k = palbp[:, ib]
    palbd_k = palbp[:, ib]

    # ---- 2. JABS=1,2 loop: re-couple cloud with gas absorption ----
    # autograd-compatible (no JIT).
    pcg_b = pcg[:, ib, :]
    pomega_b = pomega[:, ib, :]
    ptau_b = ptau[:, ib, :]
    # zrj / zrk are (nlon, 6, nlev1). Columns 0,1 come from zrj0/zrk0;
    # columns 2..5 (jn=3..6) are filled by the JABS loop below. We collect
    # per-(jn, level) results in lists and stack at the end.
    # zrefz_local is (nlon, 2, nlev1): level 0 = palbd_k, then recursive
    # per JABS. ztr_local is (nlon, 2, klev): filled per level.
    # Build zrefz_local / ztr_local as lists over the level axis too.
    # --- zrefz_local[jref] level 0 ---
    zrefz0_0 = palbd_k   # jref=1 (idx 0) level 0
    zrefz1_0 = palbd_k   # jref=2 (idx 1) level 0
    # Per-level lists (length nlev1): index = jk_f-1
    zrefz0_lev = [None] * nlev1   # jref=1
    zrefz1_lev = [None] * nlev1   # jref=2
    zrefz0_lev[0] = zrefz0_0
    zrefz1_lev[0] = zrefz1_0
    ztr0_lev = [None] * klev      # jref=1, index = jkm1-1
    ztr1_lev = [None] * klev      # jref=2, index = jkm1-1
    jn = 2
    # zrj/zrk columns 2..5 (jn=3..6): we need per-(jn, level) tensors.
    # Each jn is written only inside the jref loop. Store as dict keyed by
    # jn -> list over level (length nlev1).
    zrj_cols = {0: list(zrj0[:, 0, :].unbind(dim=1)),   # copy of col 0
                1: list(zrj0[:, 1, :].unbind(dim=1))}   # copy of col 1
    zrk_cols = {0: list(zrk0[:, 0, :].unbind(dim=1)),
                1: list(zrk0[:, 1, :].unbind(dim=1))}
    # NOTE: unbind gives per-level tensors of shape (nlon,); we rebuild
    # columns via torch.stack at the very end. Columns 0,1 above are the
    # clear-sky pseudo-fluxes.
    for jabs in range(1, 3):
        for jk_f in range(2, klev + 2):
            jkm1 = jk_f - 1; ikl = klev + 1 - jkm1; pud_idx = ikl
            pwv_l = pwv_td[:, ikl - 1].clamp(min=_REPSCQ)
            pqs_l = pqs_td[:, ikl - 1]; cld_l = pcld_for_swr[:, jkm1 - 1]
            pud_jabs = pud[:, jabs - 1, pud_idx]
            if jabs == 1:
                zbb = pud_jabs * pqs_l / pwv_l
                zcneb = torch.clamp(cld_l, min=_REPSC, max=1.0 - _REPSC)
                zaa = torch.clamp((pud_jabs - zcneb * zbb) / (1.0 - zcneb), min=_REPSCQ)
            else:
                zaa = pud_jabs; zbb = zaa
            om_c = pomega_b[:, jkm1 - 1]; tau_c = ptau_b[:, jkm1 - 1]; zw = om_c
            zto1 = tau_c / zw + ztauz[:, jkm1 - 1] / zpzaz[:, jkm1 - 1] + zbb * paki[:, jabs - 1, ib]
            zr21 = tau_c + ztauz[:, jkm1 - 1]; zr22 = tau_c / zr21
            zgg = zr22 * pcg_b[:, jkm1 - 1] + (1.0 - zr22) * zcgaz[:, jkm1 - 1]; zw = zr21 / zto1
            # Read previous-level reflectivities (functional: from the lists).
            zrefz0_prev = zrefz0_lev[jkm1 - 1]   # jref=1 (idx 0) at level jkm1
            zrefz1_prev = zrefz1_lev[jkm1 - 1]   # jref=2 (idx 1) at level jkm1
            zre1, ztr1, zre2, ztr2 = _swde(zgg, zrefz0_prev, zrmue[:, jk_f - 1], zto1, zw, novlp=1)
            paki_k = paki[:, jabs - 1, ib]
            zchks = torch.clamp(paki_k * zaa * 1.66, max=200.0)
            zchkg = torch.clamp(paki_k * zaa / zrmue[:, jk_f - 1].clamp(min=1e-10), max=200.0)
            zs = torch.exp(-zchks); zg = torch.exp(-zchkg); zsg = zs * zg
            zrr = 1.0 / (1.0 - zray2[:, jkm1 - 1] * zrefz0_prev).clamp(min=1e-300)
            zrefz1_lev[jk_f - 1] = (1.0 - cld_l) * (zray1[:, jkm1 - 1] + zrefz1_prev * ztra1[:, jkm1 - 1] * ztra2[:, jkm1 - 1]) * zsg + cld_l * zre1
            ztr1_lev[jkm1 - 1] = cld_l * ztr1 + ztra1[:, jkm1 - 1] * zg * (1.0 - cld_l)
            zrefz0_lev[jk_f - 1] = (1.0 - cld_l) * (zray1[:, jkm1 - 1] + zrefz0_prev * ztra1[:, jkm1 - 1] * ztra2[:, jkm1 - 1] * zrr) * zsg + cld_l * zre2
            ztr0_lev[jkm1 - 1] = cld_l * ztr2 + ztra1[:, jkm1 - 1] * zrr * zg * (1.0 - cld_l)
        for jref in range(1, 3):
            jn = jn + 1
            # ZRJ(:,jn,klev)=1.0 ; ZRK(:,jn,klev)=zrefz_local(:,jref,klev)
            # Bottom (level index klev) seed, then recurse upward (jk=1..klev).
            if jref == 1:
                zrefz_top = zrefz0_lev[klev]
                ztr_jref = ztr0_lev   # jref=1 -> idx 0
            else:
                zrefz_top = zrefz1_lev[klev]
                ztr_jref = ztr1_lev   # jref=2 -> idx 1
            zrj_jn = [None] * nlev1
            zrk_jn = [None] * nlev1
            # Fortran: ZRJ(:,jn,KLEV)=1.0 ; at jk=1 -> jkl=KLEV -> jkl-1=KLEV-1
            # so the stored level index in the array is jkl-1.
            # Initialise running value at jklp1-1 = klev (the seed).
            zrj_run = torch.ones(nlon, dtype=dt, device=dev)
            zrk_run = zrefz_top
            zrj_jn[klev] = zrj_run
            zrk_jn[klev] = zrk_run
            for jk in range(1, klev + 1):
                jkl = klev + 1 - jk; jklp1 = jkl + 1
                # zre11 uses ZRJ(:,jn,jklp1-1) = previous running value.
                ztr_term = ztr_jref[jkl - 1]
                zre11 = zrj_run * ztr_term
                # zrefz at level jkl-1 for this jref:
                if jref == 1:
                    zrefz_here = zrefz0_lev[jkl - 1]
                else:
                    zrefz_here = zrefz1_lev[jkl - 1]
                zrj_run = zre11
                zrk_run = zre11 * zrefz_here
                zrj_jn[jkl - 1] = zrj_run
                zrk_jn[jkl - 1] = zrk_run
            zrj_cols[jn - 1] = zrj_jn
            zrk_cols[jn - 1] = zrk_jn
    # Assemble zrj/zrk (nlon,6,nlev1) from per-column level lists.
    zrj_col_tensors = []
    zrk_col_tensors = []
    for c in range(6):
        zrj_col_tensors.append(torch.stack(zrj_cols[c], dim=1))   # (nlon,nlev1)
        zrk_col_tensors.append(torch.stack(zrk_cols[c], dim=1))
    zrj = torch.stack(zrj_col_tensors, dim=1)   # (nlon,6,nlev1)
    zrk = torch.stack(zrk_col_tensors, dim=1)

    # ---- 3. Invert grey pseudo-fluxes (swni §4) ----
    # §4.1: ZRJ(JAJ) -= ZRJ(JAJ+1), clamp REPLOG, for JAJ=1,3,5 (odd).
    # Build new zrj/zrk via functional column replacement (no in-place writes).
    zrj_cols_list = [zrj[:, k, :] for k in range(6)]
    zrk_cols_list = [zrk[:, k, :] for k in range(6)]
    for jaj in [1, 3, 5]:
        jajp = jaj + 1
        zrj_cols_list[jaj - 1] = (zrj_cols_list[jaj - 1] - zrj_cols_list[jajp - 1]).clamp(min=_REPLOG)
        zrk_cols_list[jaj - 1] = (zrk_cols_list[jaj - 1] - zrk_cols_list[jajp - 1]).clamp(min=_REPLOG)
    # §4.1 cont: clamp even JAJ=2,4,6.
    for jaj in [2, 4, 6]:
        zrj_cols_list[jaj - 1] = zrj_cols_list[jaj - 1].clamp(min=_REPLOG)
        zrk_cols_list[jaj - 1] = zrk_cols_list[jaj - 1].clamp(min=_REPLOG)
    zrj = torch.stack(zrj_cols_list, dim=1)
    zrk = torch.stack(zrk_cols_list, dim=1)

    # §4.2: inverse Laplace via PAKI -> ZW2, SWTT1 -> ZRL.
    # VECTORISED over all levels: compute ZW2 (nlon,2,nlev1) for each (jaj)
    # then call swtt1 once for all levels. This replaces the per-level Python
    # loop (552 iterations × swtt1 dispatch).
    zrr2 = 1.0 / paki[:, :, ib]                       # (nlon,2)
    pfdown = torch.zeros(nlon, nlev1, dtype=dt, device=dev)
    pfup = torch.zeros(nlon, nlev1, dtype=dt, device=dev)
    # zrl/zruef are (nlon, 8, nlev1); build column lists and stack at the end.
    zrl_cols = [None] * 8
    zruef_cols = [None] * 8
    # For each jaj (1,2): compute ZW2 for both jn_loc (1,2) across all levels.
    # jn_loc=1: ratio = zrj[:,0,:]/zrj[:,jn2j-1,:]; jn_loc=2: zrj[:,1,:]/zrj[:,jn2j-1,:]
    # ZW2[:,0,:] = log(ratio_j) * zrr2[:,jaj-1]; ZW2[:,1,:] = log(ratio_k)*zrr2
    for jaj in range(1, 3):
        jn2j_1 = 1 + 2 * jaj                    # jn_loc=1 -> jn2j
        jn2j_2 = 2 + 2 * jaj                    # jn_loc=2 -> jn2j
        zrrj_1 = zrj[:, 0, :] / zrj[:, jn2j_1 - 1, :]    # (nlon, nlev1)
        zrrk_1 = zrk[:, 0, :] / zrk[:, jn2j_1 - 1, :]
        zrrj_2 = zrj[:, 1, :] / zrj[:, jn2j_2 - 1, :]
        zrrk_2 = zrk[:, 1, :] / zrk[:, jn2j_2 - 1, :]
        zw2_1a = torch.log(zrrj_1.clamp(min=_REPLOG)) * zrr2[:, jaj - 1].unsqueeze(-1)
        zw2_1b = torch.log(zrrk_1.clamp(min=_REPLOG)) * zrr2[:, jaj - 1].unsqueeze(-1)
        zw2_2a = torch.log(zrrj_2.clamp(min=_REPLOG)) * zrr2[:, jaj - 1].unsqueeze(-1)
        zw2_2b = torch.log(zrrk_2.clamp(min=_REPLOG)) * zrr2[:, jaj - 1].unsqueeze(-1)
        # swtt1 with kkind=[jaj,jaj], pu=(nlon, 2). Stack to (nlon*nlev1, 2),
        # call once, unstack. Actually call per-level-group: reshape to
        # (nlon*nlev1, 2) by stacking the two absorber columns.
        iind2 = [jaj, jaj]
        # jn_loc=1: jkki = (jaj-1)*2 + 0
        jkki_1 = (jaj - 1) * 2 + 0
        jkki_2 = (jaj - 1) * 2 + 1
        # Build pu for jn_loc=1: (nlon, 2) per level, but we want all levels.
        # Reshape zw2_1a (nlon,nlev1) -> (nlon*nlev1,), pair with zw2_1b.
        pu1 = torch.stack([zw2_1a.reshape(-1), zw2_1b.reshape(-1)], dim=1)  # (nlon*nlev1, 2)
        zr2_1 = _swtt1(knu, 2, iind2, pu1).reshape(nlon, nlev1, 2)          # (nlon,nlev1,2)
        pu2 = torch.stack([zw2_2a.reshape(-1), zw2_2b.reshape(-1)], dim=1)
        zr2_2 = _swtt1(knu, 2, iind2, pu2).reshape(nlon, nlev1, 2)
        # Store: zrl[:,jkki,:] = zr2[:,:,0] (downward), zrl[:,jkki+4,:] = zr2[:,:,1] (upward)
        zrl_cols[jkki_1] = zr2_1[:, :, 0]
        zrl_cols[jkki_1 + 4] = zr2_1[:, :, 1]
        zrl_cols[jkki_2] = zr2_2[:, :, 0]
        zrl_cols[jkki_2 + 4] = zr2_2[:, :, 1]
        zruef_cols[jkki_1] = zw2_1a
        zruef_cols[jkki_1 + 4] = zw2_1b
        zruef_cols[jkki_2] = zw2_2a
        zruef_cols[jkki_2 + 4] = zw2_2b
    zrl = torch.stack(zrl_cols, dim=1)       # (nlon,8,nlev1)
    zruef = torch.stack(zruef_cols, dim=1)   # (nlon,8,nlev1)  [kept for parity with original]
    # §4.3 cloudy fluxes — vectorised over all levels.
    pfdown = zrj[:, 0, :] * zrl[:, 0, :] * zrl[:, 2, :] + \
             zrj[:, 1, :] * zrl[:, 1, :] * zrl[:, 3, :]
    pfup = zrk[:, 0, :] * zrl[:, 4, :] * zrl[:, 6, :] + \
           zrk[:, 1, :] * zrl[:, 5, :] * zrl[:, 7, :]

    # ---- 5+6. Clear-sky fluxes + O3/H2O continuum (autograd-compatible) ----
    # Pure-Python path (rewritten to be autograd-safe) which uses the
    # already-vectorised _swtt1.
    jaj = 2; iind3 = [1, 2, 3, 1, 2, 3]
    # zw3/zw4/zw5 are cumulative column accumulators over the down sweep;
    # use functional ``x = x + delta`` updates (no in-place +=).
    zw3 = torch.zeros(nlon, 6, dtype=dt, device=dev)
    zw4 = torch.zeros(nlon, 2, dtype=dt, device=dev)
    zw5 = torch.zeros(nlon, 2, dtype=dt, device=dev)
    zr4 = torch.ones(nlon, 2, dtype=dt, device=dev)
    # zfd (nlon,nlev1): collect per-level results in a list, stack at end.
    zfd_list = [None] * nlev1
    zfd_list[klev] = zrj0[:, jaj - 1, klev]
    zr3 = torch.zeros(nlon, 6, dtype=dt, device=dev)
    for jk in range(1, klev + 1):
        ikl = klev + 1 - jk; pud_idx = klev + 1 - ikl; poz_idx = klev - ikl
        zrr0 = 1.0 / zrmu0[:, ikl - 1].clamp(min=1e-10); zrre = 1.0 / zrmue[:, ikl - 1].clamp(min=1e-10)
        # Functional += accumulation (autograd-safe): each column is either
        # carried forward unchanged or rebuilt as old + delta. The per-column
        # stack exactly mirrors the original six ``zw3[:,k] += ...`` lines.
        zw3 = torch.stack([
            zw3[:, 0] + pud[:, 0, pud_idx] * zrr0,
            zw3[:, 1] + pud[:, 1, pud_idx] * zrr0,
            zw3[:, 2] + poz_td[:, poz_idx] * zrr0,
            zw3[:, 3] + pud[:, 0, pud_idx] * zrre,
            zw3[:, 4] + pud[:, 1, pud_idx] * zrre,
            zw3[:, 5] + poz_td[:, poz_idx] * zrre,
        ], dim=1)
        zw4 = torch.stack([
            zw4[:, 0] + pud[:, 3, pud_idx] * zrr0,
            zw4[:, 1] + pud[:, 3, pud_idx] * zrre,
        ], dim=1)
        zw5 = torch.stack([
            zw5[:, 0] + pud[:, 4, pud_idx] * zrr0,
            zw5[:, 1] + pud[:, 4, pud_idx] * zrre,
        ], dim=1)
        zr3 = _swtt1(knu, 6, iind3, zw3)
        zr4_0 = torch.exp(-rswce_k * zw4[:, 0] - rswcp_k * zw5[:, 0])
        zr4_1 = torch.exp(-rswce_k * zw4[:, 1] - rswcp_k * zw5[:, 1])
        zr4 = torch.stack([zr4_0, zr4_1], dim=1)
        zfd_list[ikl - 1] = zr3[:, 0] * zr3[:, 1] * zr3[:, 2] * zr4_0 * zrj0[:, jaj - 1, ikl - 1]
    zfd = torch.stack(zfd_list, dim=1)
    psudu2 = ((1.0 - pclear) * (zr3[:, 3] * zr3[:, 4] * zr3[:, 5] * zr4[:, 1] * ztrcld)
              + pclear * (zr3[:, 0] * zr3[:, 1] * zr3[:, 2] * zr4[:, 0] * ztrclr)) * rsun_k
    # zfu (nlon,nlev1): list-based assembly. Level 0 = zfd[:,0]*palbp_k.
    zfu_list = [None] * nlev1
    zfu_list[0] = zfd[:, 0] * palbp_k
    iind3u = [1, 2, 3]
    for jk in range(2, klev + 2):
        ikm1 = jk - 1; pud_idx = klev + 1 - ikm1; poz_idx = klev - ikm1
        # Original up-sweep only does += into columns 0,1,2 of zw3 and
        # column 0 of zw4/zw5; columns 3,4,5 (zw3) and 1 (zw4,zw5) are left
        # unchanged. Carry them forward verbatim for bit-exact parity.
        zw3 = torch.stack([
            zw3[:, 0] + pud[:, 0, pud_idx] * 1.66,
            zw3[:, 1] + pud[:, 1, pud_idx] * 1.66,
            zw3[:, 2] + poz_td[:, poz_idx] * 1.66,
            zw3[:, 3],
            zw3[:, 4],
            zw3[:, 5],
        ], dim=1)
        zw4 = torch.stack([
            zw4[:, 0] + pud[:, 3, pud_idx] * 1.66,
            zw4[:, 1],
        ], dim=1)
        zw5 = torch.stack([
            zw5[:, 0] + pud[:, 4, pud_idx] * 1.66,
            zw5[:, 1],
        ], dim=1)
        zr3u = _swtt1(knu, 3, iind3u, zw3[:, :3])
        zr4u = torch.exp(-rswce_k * zw4[:, 0] - rswcp_k * zw5[:, 0])
        zfu_list[jk - 1] = zr3u[:, 0] * zr3u[:, 1] * zr3u[:, 2] * zr4u * zrk0[:, jaj - 1, jk - 1]
    zfu = torch.stack(zfu_list, dim=1)
    iabsl = 3; zw1 = torch.zeros(nlon, dtype=dt, device=dev)
    zw4f = torch.zeros(nlon, dtype=dt, device=dev); zw5f = torch.zeros(nlon, dtype=dt, device=dev)
    zr1 = torch.zeros(nlon, dtype=dt, device=dev)
    # --- Autograd-compatible flux assembly (no in-place index assign) ---
    # pfdown sweep: bottom (klev) -> top (0), recursive adding
    _pfdown_list = [None] * nlev1
    _pcdown_list = [None] * nlev1
    _pfdown_list[klev] = ((1.0 - pclear) * pfdown[:, klev] + pclear * zfd[:, klev]) * rsun_k
    _pcdown_list[klev] = zfd[:, klev] * rsun_k
    for jk in range(1, klev + 1):
        ikl = klev + 1 - jk; pud_idx = klev + 1 - ikl; poz_idx = klev - ikl
        zrr = 1.0 / zrmue[:, ikl - 1].clamp(min=1e-10)
        zw1 = zw1 + poz_td[:, poz_idx] * zrr; zw4f = zw4f + pud[:, 3, pud_idx] * zrr; zw5f = zw5f + pud[:, 4, pud_idx] * zrr
        zr4f = torch.exp(-rswce_k * zw4f - rswcp_k * zw5f); zr1 = _swtt(knu, iabsl, zw1)
        _pfdown_list[ikl - 1] = ((1.0 - pclear) * zr1 * zr4f * _pfdown_list[ikl] + pclear * zfd[:, ikl - 1]) * rsun_k
        _pcdown_list[ikl - 1] = zfd[:, ikl - 1] * rsun_k
    pfdown = torch.stack(_pfdown_list, dim=1)
    pcdown = torch.stack(_pcdown_list, dim=1)
    # pfup sweep: top (0) -> bottom (klev), recursive adding
    _pfup_list = [None] * nlev1
    _pcup_list = [None] * nlev1
    _pfup_list[0] = ((1.0 - pclear) * zr1 * zr4f * pfup[:, 0] + pclear * zfu[:, 0]) * rsun_k
    _pcup_list[0] = zfu[:, 0] * rsun_k
    for jk in range(2, klev + 2):
        ikm1 = jk - 1; poz_idx = klev - ikm1; pud_idx = klev + 1 - ikm1
        zw1 = zw1 + poz_td[:, poz_idx] * 1.66; zw4f = zw4f + pud[:, 3, pud_idx] * 1.66; zw5f = zw5f + pud[:, 4, pud_idx] * 1.66
        zr4f = torch.exp(-rswce_k * zw4f - rswcp_k * zw5f); zr1 = _swtt(knu, iabsl, zw1)
        _pfup_list[jk - 1] = ((1.0 - pclear) * zr1 * zr4f * _pfup_list[jk - 2] + pclear * zfu[:, jk - 1]) * rsun_k
        _pcup_list[jk - 1] = zfu[:, jk - 1] * rsun_k
    pfup = torch.stack(_pfup_list, dim=1)
    pcup = torch.stack(_pcup_list, dim=1)

    return dict(pfdown=pfdown, pfup=pfup, pcdown=pcdown, pcup=pcup, psudu2=psudu2)


__all__ = ["swni"]
