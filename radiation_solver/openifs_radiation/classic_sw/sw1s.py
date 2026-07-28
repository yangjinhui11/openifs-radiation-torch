"""Port of sw1s.F90 — shortwave solver for spectral bands 1-3 (UV + visible).

Fortran reference:
  openifs-48r1/ifs-source/arpifs/phys_radi/sw1s.F90

sw1s is called by sw.F90 for JNU=1..3 (NSW=6). It:
  1. Calls swclr (clear-sky adding) + swr (cloudy adding) to get the adding
     matrices ZRJ0/ZRK0/ZRMU0 (clear) and ZRJ/ZRK/ZRMUE (cloudy).
  2. Accumulates gas column amounts ZW = ZW + PUD*secant (two secants: the
     cloudy equivalent angle ZRMUE and the clear ZRMU0), calls SWTT1 for the
     H2O+CO2 Pade transmission and SWUVO3 for the O3 transmission.
  3. Combines cloudy + clear fluxes weighted by PCLEAR:
       PFD = ((1-PCLEAR)*ZRTMP + PCLEAR*ZCLRTMP) * RSUN
  4. Upward pass adds the diffuse 1.66 path.

Production config: NSW=6 path (bands 1-3 call SWUVO3), NOVLP=1, no dust.

Bottom-up internal convention (matches Fortran); PREFZ index 1 = surface.
"""
from __future__ import annotations
import torch

from .swclr import swclr as _swclr
from .swr import swr as _swr
from .swtt import swtt1 as _swtt1, swuvo3 as _swuvo3


def sw1s(knu: int, paer_td: torch.Tensor, palbp: torch.Tensor,
         pdsig_td: torch.Tensor, psec: torch.Tensor, prmu: torch.Tensor,
         pud: torch.Tensor, poz_td: torch.Tensor, pcg: torch.Tensor,
         pomega: torch.Tensor, ptau: torch.Tensor, pcld_bu: torch.Tensor,
         pclear: torch.Tensor, prayl: torch.Tensor, rray: torch.Tensor,
         rsun: torch.Tensor) -> dict:
    """SW1S solver for one spectral band — port of sw1s.F90 (NSW=6 path).

    Args:
        knu:    spectral interval (1..3).
        paer_td:(nlon, 6, klev) Tegen aerosol, top-down.
        palbp:  (nlon, nsw) surface albedo.
        pdsig_td:(nlon, klev) sigma thickness, top-down.
        psec:   (nlon,) secant = 1/mu0.
        prmu:   (nlon,) cos(sza).
        pud:    (nlon, 5, klev+1) per-layer column amounts from swu (top-down,
                index 0 = TOA boundary = 0).
        poz_td: (nlon, klev) O3 column amount (cm-atm), top-down.
        pcg/pomega/ptau: (nlon, nsw, klev) cloud optical props.
        pcld_bu:(nlon, klev) effective cloud fraction (bottom-up, from swu).
        pclear: (nlon,) clear-sky column fraction (from swu).
        prayl:  (nlon,) Rayleigh coefficient (precomputed by caller).
        rray:   (6, 6) Rayleigh polynomial table.
        rsun:   (6,) solar weighting per band.

    Returns dict: pfd, pfu (nlon, klev+1) total fluxes; pcd, pcu clear-sky
    fluxes; psudu1, pdiffs, pdirfs (diagnostics).
    """
    nlon = psec.shape[0]
    nlev1 = pud.shape[2]
    klev = nlev1 - 1
    dt = psec.dtype
    dev = psec.device
    ib = knu - 1
    rsun_k = rsun[ib]

    # Rayleigh coefficient ZRAYL (sw1s line ~170): polynomial in PRMU.
    # sw1s: ZRAYL = RRAY(KNU,1) + PRMU*(RRAY(KNU,2)+PRMU*(...))
    r = rray[ib]
    zrayl = r[5] + torch.zeros_like(prmu)
    for k in range(4, -1, -1):
        zrayl = r[k] + prmu * zrayl

    # ---- 1. swclr (clear-sky adding) + swr (cloudy adding) ----
    # swclr takes top-down inputs (pdsig_td, paer_td) and flips internally.
    # swr takes bottom-up pcld (it does not flip); flip our top-down pcld.
    clr = _swclr(knu, 1, paer_td, palbp, pdsig_td, zrayl, psec, rray, prmu)
    pcld_for_swr = pcld_bu.flip(dims=[1]) if pcld_bu is not None else None
    cld = _swr(knu, palbp, pcg, pcld_for_swr, pomega, psec, ptau,
               clr["pcgaz"], clr["ppizaz"], clr["ptauz"])

    # ZRJ0/ZRK0/ZRMU0 (clear), ZRJ/ZRK/ZRMUE (cloudy) from the adding matrices.
    zrj0 = clr["prj"]    # (nlon,6,klev+1)
    zrk0 = clr["prk"]
    zrmu0 = clr["prmu0"]  # (nlon,klev+1)
    zrj = cld["prj"]
    zrk = cld["prk"]
    zrmue = cld["prmue"]

    # ---- 2. NSW==6 downward/upward flux assembly — JIT flux sweep ----
    from .swtt import _load as _swtt_load
    _swtt_load()
    from .swtt import _tables as _swtt_tables
    apad = _swtt_tables["apad"].to(dev)
    bpad = _swtt_tables["bpad"].to(dev)
    d_tab = _swtt_tables["d"].to(dev)
    rxpo3 = _swtt_tables["rxpo3"].to(dev)
    nexpo3_val = int(_swtt_tables["nexpo3"][knu - 1].item())
    palbp_k = palbp[:, ib]
    # Flux assembly (autograd-compatible: no in-place ops on tensors that
    # participate in the graph; results are collected in Python lists and
    # stacked at the end).
    jaj = 2
    # Running column-amount accumulators (re-created each step so the graph
    # is a pure functional chain — never mutated in place).
    zw = torch.zeros(nlon, 4, dtype=dt, device=dev)
    zo = torch.zeros(nlon, 2, dtype=dt, device=dev)
    kkind = [1, 2, 1, 2]

    # ---- Downward pass: collect pfd/pcd per level into lists ----
    # TOA level (index klev) is set from the adding matrices directly.
    pfd_klev = ((1.0 - pclear) * zrj[:, jaj - 1, klev]
                + pclear * zrj0[:, jaj - 1, klev]) * rsun_k
    pcd_klev = zrj0[:, jaj - 1, klev] * rsun_k

    # pfd_down[i] / pcd_down[i] will hold level (klev-1-i), i.e. ikl-1 with
    # ikl descending from klev to 1, so level index descends klev-1..0.
    pfd_down = [None] * klev
    pcd_down = [None] * klev

    zrtmp = torch.zeros(nlon, dtype=dt, device=dev)
    zclrtmp = torch.zeros(nlon, dtype=dt, device=dev)
    zr = None
    zt = None
    for jk in range(1, klev + 1):
        ikl = klev + 1 - jk
        zre = 1.0 / zrmue[:, ikl - 1].clamp(min=1e-10)
        zr0 = 1.0 / zrmu0[:, ikl - 1].clamp(min=1e-10)
        pud_idx = klev + 1 - ikl; poz_idx = klev - ikl
        pud_h2o = pud[:, 0, pud_idx]; pud_co2 = pud[:, 1, pud_idx]; poz = poz_td[:, poz_idx]
        zw0 = zw[:, 0] + pud_h2o * zre
        zw1 = zw[:, 1] + pud_co2 * zre
        zw2 = zw[:, 2] + pud_h2o * zr0
        zw3 = zw[:, 3] + pud_co2 * zr0
        zo0 = zo[:, 0] + poz * zre
        zo1 = zo[:, 1] + poz * zr0
        zw = torch.stack([zw0, zw1, zw2, zw3], dim=1)
        zo = torch.stack([zo0, zo1], dim=1)
        zr = _swtt1(knu, 4, kkind, zw); zt = _swuvo3(knu, zo)
        zrtmp = zr[:, 0] * zr[:, 1] * zt[:, 0] * zrj[:, jaj - 1, ikl - 1]
        zclrtmp = zr[:, 2] * zr[:, 3] * zt[:, 1] * zrj0[:, jaj - 1, ikl - 1]
        pfd_down[jk - 1] = ((1.0 - pclear) * zrtmp + pclear * zclrtmp) * rsun_k
        pcd_down[jk - 1] = zclrtmp * rsun_k

    psudu1 = ((1.0 - pclear) * (zr[:, 0] * zr[:, 1] * zt[:, 0] * cld["ptrcld"])
              + pclear * (zr[:, 2] * zr[:, 3] * zt[:, 1] * clr["ptrclr"])) * rsun_k

    # Surface upward flux (level 0) from the last downward-pass values.
    pfu_0 = ((1.0 - pclear) * zrtmp * palbp_k
             + pclear * zclrtmp * palbp_k) * rsun_k
    pcu_0 = zclrtmp * palbp_k * rsun_k

    # ---- Upward pass: collect pfu/pcu per level into lists ----
    # pfu_up[i] / pcu_up[i] will hold level (jk-1) with jk ascending 2..klev+1,
    # i.e. level index ascending 1..klev.
    pfu_up = [None] * klev
    pcu_up = [None] * klev
    for jk in range(2, klev + 2):
        ikm1 = jk - 1; pud_idx = klev + 1 - ikm1; poz_idx = klev - ikm1
        pud_h2o = pud[:, 0, pud_idx]; pud_co2 = pud[:, 1, pud_idx]
        poz = poz_td[:, poz_idx] if poz_idx >= 0 else torch.zeros(nlon, dtype=dt, device=dev)
        zw0 = zw[:, 0] + pud_h2o * 1.66
        zw1 = zw[:, 1] + pud_co2 * 1.66
        zw2 = zw[:, 2] + pud_h2o * 1.66
        zw3 = zw[:, 3] + pud_co2 * 1.66
        zo0 = zo[:, 0] + poz * 1.66
        zo1 = zo[:, 1] + poz * 1.66
        zw = torch.stack([zw0, zw1, zw2, zw3], dim=1)
        zo = torch.stack([zo0, zo1], dim=1)
        zr = _swtt1(knu, 4, kkind, zw); zt = _swuvo3(knu, zo)
        zrtmp = zr[:, 0] * zr[:, 1] * zt[:, 0] * zrk[:, jaj - 1, jk - 1]
        zclrtmp = zr[:, 2] * zr[:, 3] * zt[:, 1] * zrk0[:, jaj - 1, jk - 1]
        pfu_up[jk - 2] = ((1.0 - pclear) * zrtmp + pclear * zclrtmp) * rsun_k
        pcu_up[jk - 2] = zclrtmp * rsun_k

    # ---- Assemble flux profiles by stacking at the correct level order ----
    # Downward: pfd_down holds levels [klev-1, klev-2, ..., 1, 0] (descending),
    # i.e. pfd_down[0] is level klev-1 and pfd_down[klev-1] is level 0.
    # Stacking gives columns [klev-1..0]; flip(1) -> [0..klev-1], then append
    # the TOA level klev at the end.
    pfd = torch.cat([torch.stack(pfd_down, dim=1).flip(dims=[1]),
                     pfd_klev.unsqueeze(1)], dim=1)
    pcd = torch.cat([torch.stack(pcd_down, dim=1).flip(dims=[1]),
                     pcd_klev.unsqueeze(1)], dim=1)
    # Upward: pfu_up holds levels [1, 2, ..., klev] (ascending), i.e. pfu_up[0]
    # is level 1 and pfu_up[klev-1] is level klev. Prepend the surface level 0.
    pfu = torch.cat([pfu_0.unsqueeze(1),
                     torch.stack(pfu_up, dim=1)], dim=1)
    pcu = torch.cat([pcu_0.unsqueeze(1),
                     torch.stack(pcu_up, dim=1)], dim=1)

    return dict(pfd=pfd, pfu=pfu, pcd=pcd, pcu=pcu, psudu1=psudu1)


__all__ = ["sw1s"]
