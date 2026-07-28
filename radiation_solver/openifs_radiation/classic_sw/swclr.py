"""Port of swclr.F90 — clear-sky reflectivity/transmissivity adding.

Fortran reference:
  openifs-48r1/ifs-source/arpifs/phys_radi/swclr.F90

Computes the clear-sky column reflectivity/transmissivity adding matrix used
by sw1s (UV/visible, bands 1-3) and swni (near-IR, bands 4-6). The algorithm
has three stages:

  1. Aerosol + Rayleigh optical properties per layer (delta-Eddington transform).
  2. Maximum-random overlap clear-sky fraction ZC0I (NOVLP=1 production path).
  3. Delta-Eddington layer R/T + adding matrix PRJ/PRK.

Convention note: this port uses Fortran's BOTTOM-UP convention internally
(JK=1 = surface) to match swclr.F90 line-for-line, which is the easiest way to
guarantee bit-exactness. Callers pass top-down torch tensors; ``swclr``
flips them on entry and flips outputs back.

Production config only: NOVLP=1 (maximum-random), ECMWF aerosol branch
(NOVLP < 5), no dust (LDDUST=.FALSE.), NSW=6.

Autograd note: this version avoids every in-place index assignment
(``arr[..., idx] = expr``), cumulative ``+=`` and ``.fill_``/``.zero_`` on
gradient-carrying tensors, so the whole routine is differentiable. Loops that
previously filled a pre-allocated tensor instead collect per-level results
into Python lists and ``torch.stack`` them at the end; carried state
(``zclear``/``zscat``/``prefz_prev``/``prj_prev``) is rebound with functional
``x = x + delta``-style expressions. The arithmetic is unchanged, so outputs
are bit-exact with the in-place version.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path

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

# Production constants (surdi.F90 / suecrad.F90).
_REPCLC = 1.0e-12      # cloud-cover security (surdi.F90:87)
_REPSCT = 1.0e-12      # SW optical-thickness security (suecrad.F90:959)


def _load():
    if not _tables:
        _tables["rtaua"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_rtaua.npy"))).to(torch.float64)
        _tables["rpiza"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_rpiza.npy"))).to(torch.float64)
        _tables["rcga"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_rcga.npy"))).to(torch.float64)


def _prayl(rray_knu: torch.Tensor, prmu: torch.Tensor) -> torch.Tensor:
    """Rayleigh scattering coefficient (sw1s/swni line: ZRAYL = RRAY(KNU,1)+...).

    RRAY(KNU,1..6) is a 5th-order polynomial in PRMU (cos sza). The leading
    coefficient indexing matches sw1s.F90 (RRAY(KNU,1) + PRMU*(RRAY(KNU,2)+...)).
    """
    # Horner: r = c0 + prmu*(c1 + prmu*(c2 + prmu*(c3 + prmu*(c4 + prmu*c5))))
    r = rray_knu[5] + torch.zeros_like(prmu)
    for k in range(4, -1, -1):
        r = rray_knu[k] + prmu * r
    return r


def swclr(knu: int, kaer: int, paer_td: torch.Tensor, palbp: torch.Tensor,
          pdsig_td: torch.Tensor, prayl: torch.Tensor, psec: torch.Tensor,
          rray: torch.Tensor, prmu: torch.Tensor) -> dict:
    """Clear-sky reflectivity/transmissivity adding — port of swclr.F90.

    Args (top-down torch, JK=0=TOA):
        knu:    spectral interval (1..6).
        kaer:   aerosol mode (1 = Tegen aerosol optical props active).
        paer_td:(nlon, 6, klev) Tegen aerosol optical thickness, top-down.
        palbp:  (nlon, nsw) surface SW albedo (full array; swclr uses knu col).
        pdsig_td:(nlon, klev) sigma-layer thickness, top-down.
        prayl:  (nlon,) Rayleigh coefficient (caller computes from RRAY+prmu).
        psec:   (nlon,) secant = 1/mu0.
        rray:   (6, 6) Rayleigh polynomial coefficients table.
        prmu:   (nlon,) cos(sza) (= PRMU passed through from swu).

    Returns (top-down):
        dict with pcgaz, ppizaz, ptauz (nlon,klev); pray1, pray2, ptra1, ptra2,
        prmu0 (nlon,klev+1); prefz (nlon,2,klev+1); prj, prk (nlon,6,klev+1);
        ptrclr (nlon,).
    """
    _load()
    nlon, six, klev = paer_td.shape
    dt = paer_td.dtype
    dev = paer_td.device
    assert six == 6

    # ---- flip top-down inputs to bottom-up (JK=1=surface) ----
    paer_bu = paer_td.flip(dims=[2])           # (nlon,6,klev) bottom-up
    pdsig_bu = pdsig_td.flip(dims=[1])         # (nlon,klev) bottom-up

    rtaua = _get("rtaua", dev)           # (6,6) [knu, jaer]
    rpiza = _get("rpiza", dev)
    rcga  = _get("rcga", dev)
    ib = knu - 1
    ta_k = rtaua[ib]                            # (6,) for this band
    pi_k = rpiza[ib]
    cg_k = rcga[ib]

    palbp_k = palbp[:, ib]                      # (nlon,) this band's albedo

    # ================================================================
    # 1. Aerosol optical params + Rayleigh, delta-Eddington transform
    #    swclr.F90 lines 138-247 (NOVLP<5 ECMWF branch, KAER/=0 path).
    #    JK=1..KLEV bottom-up; IKL=KLEV+1-JK indexes top-down PAER.
    # ================================================================
    # Per-layer accumulators (bottom-up). Initialise to 0.
    ptauz = torch.zeros(nlon, klev, dtype=dt, device=dev)   # PTAUAZ
    ppizaz = torch.zeros(nlon, klev, dtype=dt, device=dev)  # PPIZAZ
    pcgaz = torch.zeros(nlon, klev, dtype=dt, device=dev)   # PCGAZ

    # Aerosol weighted sums — VECTORISED over all layers (replaces the per-layer
    # Python loop). Fortran: PTAUAZ(:,:,JK) = Σ_aer PAER(:,:,IKL)*RTAUA where
    # IKL=KLEV+1-JK (bottom-up output JK reads top-down input IKL). We compute
    # in top-down order then the stage-2/3 loops index consistently.
    # paer_td is (nlon, 6, klev) top-down. ta_k/pi_k/cg_k are (6,).
    # PTAUAZ(nlon,klev) = einsum('nsk,s->nk', paer_td, ta_k) — but outputs are
    # indexed bottom-up (jk_bu=0=ground). paer_td[:,:,klev-1-jk_bu] is the
    # ground layer for jk_bu=0, so flip paer on the level axis then einsum.
    paer_bu = paer_td.flip(dims=[2])             # (nlon,6,klev) bottom-up
    ptauz = (paer_bu * ta_k.view(1, 6, 1)).sum(dim=1)              # (nlon,klev)
    ppizaz = (paer_bu * (ta_k * pi_k).view(1, 6, 1)).sum(dim=1)
    pcgaz = (paer_bu * (ta_k * pi_k * cg_k).view(1, 6, 1)).sum(dim=1)

    if kaer != 0:
        # Normalise: PCGAZ/=PPIZAZ, PPIZAZ/=PTAUAZ (swclr.F90 lines 207-214).
        zi_piza = 1.0 / ppizaz.clamp(min=1e-300)
        pcgaz = pcgaz * zi_piza
        zi_tauz = 1.0 / ptauz.clamp(min=1e-300)
        ppizaz_n = ppizaz * zi_tauz
        # ZTRAY = PRAYL * PDSIG(JL,JK)  (bottom-up PDSIG)
        ztray = prayl.unsqueeze(1) * pdsig_bu
        zff = pcgaz * pcgaz
        # delta-Eddington transform (swclr.F90 lines 215-224).
        zdenb = ztray + ptauz * (1.0 - ppizaz_n * zff)
        zr = 1.0 / zdenb
        zratio = ztray * zr
        ptauz = ztray + ptauz * (1.0 - ppizaz_n * zff)
        zdiv1 = 1.0 / (1.0 + pcgaz)
        pcgaz = pcgaz * (1.0 - zratio) * zdiv1
        zdiv2 = 1.0 / (1.0 - ppizaz_n * zff)
        ppizaz = zratio + (1.0 - zratio) * ppizaz_n * (1.0 - zff) * zdiv2
    else:
        # KAER=0: Rayleigh only (swclr.F90 lines 227-231).
        ztray = prayl.unsqueeze(1) * pdsig_bu
        ptauz = ztray
        pcgaz = torch.zeros_like(ptauz)
        ppizaz = torch.full_like(ptauz, 1.0 - _REPSCT)

    # ================================================================
    # 2. Maximum-random overlap clear-sky fraction ZC0I (NOVLP=1).
    #    swclr.F90 lines 248-340. ZC0I(:,:,JKL), JKL=KLEV+1-JK (top-down).
    #    Fortran JK=1..KLEV reads PTAUAZ/PPIZAZ/PCGAZ at JKL (top-down index),
    #    so the sweep proceeds TOA -> surface.
    #    Our ptauz/ppizaz/pcgaz are bottom-up (index 0=ground); the layer at
    #    Fortran JKL maps to bottom-up index klev-JKL.
    # ================================================================
    # ZCLEAR/ZSCAT are carried across iterations (sequential overlap sweep);
    # rebinding these Python vars is autograd-safe. Only the per-level ZC0I
    # value is collected into a list (avoids in-place index writes on a tensor
    # that carries gradient history through zss_all -> zcorae_all).
    zclear = torch.ones(nlon, dtype=dt, device=dev)
    zscat = torch.zeros(nlon, dtype=dt, device=dev)
    # ── Stage 2 precompute (vectorised over all levels) ──
    zcorae_all = (1.0 - ppizaz * pcgaz * pcgaz) * ptauz * psec.unsqueeze(-1)
    zr21_all = torch.exp(-zcorae_all)               # (nlon, klev) bottom-up
    zss_all = 1.0 - zr21_all
    # ── Stage 2 sequential accumulation (thin loop) ──
    # zc0i_list collects in Fortran JK order: JK=1 -> JKL=KLEV -> zc0i index
    # klev-1; ...; JK=KLEV -> JKL=1 -> zc0i index 0. So list[k] holds the value
    # for bottom-up index (klev-1-k); we reverse on assembly.
    zc0i_list: list = []
    for jk in range(1, klev + 1):                 # Fortran JK=1..KLEV
        jkl = klev + 1 - jk                        # top-down level 1..KLEV
        zss = zss_all[:, jkl - 1]                  # precomputed
        ziclear = 1.0 / (1.0 - torch.clamp(zscat, max=1.0 - _REPCLC))
        zclear = zclear * (1.0 - torch.maximum(zss, zscat)) * ziclear
        zc0i_list.append(1.0 - zclear)             # Fortran ZC0I(:,:,JKL)
        zscat = zss
    # Reverse so assembled position i == bottom-up index i; pad TOA boundary
    # ZC0I(:,:,KLEV+1)=0 (set at init in Fortran) with a zero column.
    zc0i_core = torch.stack(list(reversed(zc0i_list)), dim=1)   # (nlon, klev)
    zc0i = torch.cat(
        [zc0i_core, torch.zeros(nlon, 1, dtype=dt, device=dev)],
        dim=1)                                     # (nlon, klev+1)

    # ================================================================
    # ================================================================
    # 3. Delta-Eddington layer R/T + adding — autograd-compatible sweep.
    # ================================================================
    # PREFZ has a sequential dependency: PREFZ(:,:,JK) depends on
    # PREFZ(:,:,JK-1). We carry ``prefz_prev`` ([nlon,2]) across iterations
    # (rebinding, not in-place) and collect per-level R/T/PREFZ/ZTR/PRMU0
    # into Python lists, then torch.stack at the end. This preserves the
    # exact arithmetic of the Fortran adding equations while keeping every
    # intermediate a fresh tensor in the autograd graph.
    prefz_list: list = []                     # per-level [nlon,2], idx 0..klev
    pray1_list: list = []                     # per-level [nlon], idx 0..klev-1
    pray2_list: list = []
    ptra1_list: list = []
    ptra2_list: list = []
    ztr_list: list = []                       # per-level [nlon,2], idx 0..klev-1
    prmu0_tail: list = []                     # per-level [nlon], idx 1..klev

    # Level-0 init (Fortran PREFZ(JL,1,1)=PALBP, PREFZ(JL,2,1)=PALBP).
    prefz_prev = torch.stack([palbp_k, palbp_k], dim=1)   # [nlon,2]
    prefz_list.append(prefz_prev)

    for jk_f in range(2, klev + 2):
        jkm1 = jk_f - 1
        zc0i_jk = zc0i[:, jk_f - 1]
        jk_bu = jkm1 - 1
        cg = pcgaz[:, jk_bu]; pz = ppizaz[:, jk_bu]; tz = ptauz[:, jk_bu]
        zmue = (1.0 - zc0i_jk) * psec + zc0i_jk * 1.66
        prmu0_jk = 1.0 / zmue
        zbmu0 = 0.5 - 0.75 * cg * prmu0_jk
        zden = 1.0 + (1.0 - pz + zbmu0 * pz) * tz * zmue + \
               (1.0 - pz) * (1.0 - pz + 2.0 * zbmu0 * pz) * tz * tz * zmue * zmue
        ptra1_jk = 1.0 / zden
        pray1_jk = zbmu0 * pz * tz * zmue * ptra1_jk
        zbmu1 = 0.5 - 0.75 * cg * 0.5
        zden1 = 1.0 + (1.0 - pz + zbmu1 * pz) * tz * 2.0 + \
                (1.0 - pz) * (1.0 - pz + 2.0 * zbmu1 * pz) * tz * tz * 4.0
        ptra2_jk = 1.0 / zden1
        pray2_jk = zbmu1 * pz * tz * 2.0 * ptra2_jk
        prefz_prev0 = prefz_prev[:, 0]
        zrr = 1.0 / (1.0 - pray2_jk * prefz_prev0).clamp(min=1e-300)
        prefz0_new = pray1_jk + \
            prefz_prev0 * ptra1_jk * ptra2_jk * zrr
        ztr0_jk = ptra1_jk * zrr
        prefz_prev1 = prefz_prev[:, 1]
        prefz1_new = pray1_jk + \
            prefz_prev1 * ptra1_jk * ptra2_jk
        ztr1_jk = ptra1_jk
        # collect this level's results.
        pray1_list.append(pray1_jk)
        pray2_list.append(pray2_jk)
        ptra1_list.append(ptra1_jk)
        ptra2_list.append(ptra2_jk)
        ztr_list.append(torch.stack([ztr0_jk, ztr1_jk], dim=1))
        prmu0_tail.append(prmu0_jk)
        # carry updated PREFZ to next level (new tensor, not in-place).
        prefz_prev = torch.stack([prefz0_new, prefz1_new], dim=1)
        prefz_list.append(prefz_prev)
    # After-loop: PRMU0(:,1) (Fortran JKL=1, our index 0) and PTRCLR.
    zmue1 = (1.0 - zc0i[:, 0]) * psec + zc0i[:, 0] * 1.66
    prmu0_head = 1.0 / zmue1
    ptrclr = 1.0 - zc0i[:, 0]

    # Assemble level-indexed tensors from the collected lists.
    # PREFZ: stack [nlon,2] per level on a new level axis -> permute to
    # (nlon,2,klev+1). Every level 0..klev was set explicitly above.
    _prefz = torch.stack(prefz_list, dim=1)              # (nlon, klev+1, 2)
    prefz_f = _prefz.permute(0, 2, 1).contiguous()       # (nlon, 2, klev+1)
    # PRAY1/2: indices 0..klev-1 from loop, index klev stays 0 (Fortran init).
    _zero_col = torch.zeros(nlon, 1, dtype=dt, device=dev)
    pray1_f = torch.cat([torch.stack(pray1_list, dim=1), _zero_col], dim=1)
    pray2_f = torch.cat([torch.stack(pray2_list, dim=1), _zero_col], dim=1)
    # PTRA1/2: indices 0..klev-1 from loop, index klev stays 1 (Fortran init).
    _one_col = torch.ones(nlon, 1, dtype=dt, device=dev)
    ptra1_f = torch.cat([torch.stack(ptra1_list, dim=1), _one_col], dim=1)
    ptra2_f = torch.cat([torch.stack(ptra2_list, dim=1), _one_col], dim=1)
    # ZTR: (nlon,2,klev), all levels 0..klev-1 set in loop.
    _ztr = torch.stack(ztr_list, dim=1)                  # (nlon, klev, 2)
    ztr_f = _ztr.permute(0, 2, 1).contiguous()           # (nlon, 2, klev)
    # PRMU0: index 0 from after-loop head, indices 1..klev from loop tail.
    prmu0_f = torch.stack([prmu0_head] + prmu0_tail, dim=1)   # (nlon, klev+1)

    # ================================================================
    # 4. PRJ/PRK adding matrix (swclr.F90 lines 422-471).
    #    INU1=3 for NSW=6. KNU<=INU1 (bands 1-3): JAJ=2 only.
    #    KNU>INU1 (bands 4-6): JAJ=1,2.
    #
    #    PRJ has a sequential dependency: PRJ(:,JAJ,jkl-1) depends on
    #    PRJ(:,JAJ,jklp1-1) (the level above). We carry ``prj_prev`` across
    #    iterations and build each used JAJ row as a [nlon,klev+1] tensor,
    #    then assemble the full (nlon,6,klev+1) matrix with torch.stack over
    #    rows. Unused rows are constant zeros (no in-place writes anywhere).
    # ================================================================
    inu1 = 3                                      # NSW==6
    ones_n = torch.ones(nlon, dtype=dt, device=dev)
    prj_rows = [torch.zeros(nlon, klev + 1, dtype=dt, device=dev) for _ in range(6)]
    prk_rows = [torch.zeros(nlon, klev + 1, dtype=dt, device=dev) for _ in range(6)]
    if knu <= inu1:
        # KNU<=INU1 (bands 1-3): JAJ=2 only, and uses ZTR/PREFZ column 1
        # (swclr.F90 lines 437-448: ZTR(JL,1,JKL), PREFZ(JL,1,...)) — i.e. the
        # direct-beam column regardless of JAJ.
        jaj = 2
        col = 0                                   # Fortran column 1 -> 0-based 0
        ztr_col = ztr_f[:, col, :]                # (nlon, klev) — slicing read
        prefz_col = prefz_f[:, col, :]            # (nlon, klev+1) — slicing read
        # PRJ(:,JAJ,KLEV)=1, downward scan PRJ(jkl-1)=PRJ_prev*ZTR(jkl-1).
        prj_scan: list = [None] * (klev + 1)      # index = jkl-1 (0..klev)
        prk_scan: list = [None] * (klev + 1)
        prj_scan[klev] = ones_n
        prk_scan[klev] = prefz_col[:, klev]
        prj_prev = ones_n
        for jk in range(1, klev + 1):             # Fortran JK=1..KLEV
            jkl = klev + 1 - jk                   # top-down level
            jklp1 = jkl + 1
            zre11 = prj_prev * ztr_col[:, jkl - 1]
            prj_scan[jkl - 1] = zre11
            prk_scan[jkl - 1] = zre11 * prefz_col[:, jkl - 1]
            prj_prev = zre11
        prj_rows[jaj - 1] = torch.stack(prj_scan, dim=1)
        prk_rows[jaj - 1] = torch.stack(prk_scan, dim=1)
    else:
        # KNU>INU1 (bands 4-6): JAJ=1,2, each using its own ZTR/PREFZ column.
        for jaj in (1, 2):
            col = jaj - 1                         # Fortran column JAJ -> 0-based JAJ-1
            ztr_col = ztr_f[:, col, :]            # (nlon, klev) — slicing read
            prefz_col = prefz_f[:, col, :]        # (nlon, klev+1) — slicing read
            prj_scan: list = [None] * (klev + 1)
            prk_scan: list = [None] * (klev + 1)
            prj_scan[klev] = ones_n
            prk_scan[klev] = prefz_col[:, klev]
            prj_prev = ones_n
            for jk in range(1, klev + 1):         # Fortran JK=1..KLEV
                jkl = klev + 1 - jk
                jklp1 = jkl + 1
                zre11 = prj_prev * ztr_col[:, jkl - 1]
                prj_scan[jkl - 1] = zre11
                prk_scan[jkl - 1] = zre11 * prefz_col[:, jkl - 1]
                prj_prev = zre11
            prj_rows[jaj - 1] = torch.stack(prj_scan, dim=1)
            prk_rows[jaj - 1] = torch.stack(prk_scan, dim=1)
    prj_f = torch.stack(prj_rows, dim=1)          # (nlon, 6, klev+1)
    prk_f = torch.stack(prk_rows, dim=1)          # (nlon, 6, klev+1)

    return {
        "pcgaz":  pcgaz,                            # bottom-up (nlon,klev)
        "ppizaz": ppizaz,
        "ptauz":  ptauz,
        "pray1":  pray1_f, "pray2": pray2_f,        # Fortran-indexed (nlon,klev+1)
        "prefz":  prefz_f,                          # (nlon,2,klev+1)
        "ptra1":  ptra1_f, "ptra2": ptra2_f,
        "prj":    prj_f, "prk": prk_f,              # (nlon,6,klev+1)
        "prmu0":  prmu0_f,                          # (nlon,klev+1)
        "ptrclr": ptrclr,                           # (nlon,)
    }


__all__ = ["swclr"]
