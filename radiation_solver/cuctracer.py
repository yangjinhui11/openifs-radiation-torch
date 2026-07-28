"""Port of CUCTRACER — convective transport of chemical tracers.

Fortran reference: openifs-48r1/ifs-source/arpifs/phys_ec/cuctracer.F90

CUCTRACER transports positive-definite chemical tracers (aerosols, ozone,
chemical species) through the convective updraft/downdraft mass-flux profile.
It is called from CUMASTRN after the mass-flux closure has produced PMFU/PMFD
and the detrainment rates. The solver supports explicit (RMFSOLCT=0) and
semi-/fully-implicit (RMFSOLCT>0) vertical advection via a bidiagonal solve
(CUBIDIAG).

Production defaults (sucumf.F90):
  RMFSOLCT = 1.0  (fully implicit)
  RMFCMIN  = 1e-8 (minimum mass-flux safety)
  RMFADVW  = 0.0  (no subsidence handed to dynamics)
  RMFADVWDD= 0.0

IMPORTANT: this routine is for positive-definite quantities. Wet scavenging
is applied if PSCAV > 0.

Convention: all level-indexed arrays are top-down (index 0 = TOA), matching
the rest of the torch convection package. The Fortran loops JK=2..KLEV are
mapped directly; JK=1 (TOA half-level) is unused.
"""
from __future__ import annotations
import torch

# Physical constant.
_RG = 9.80665


def _cubidiag(kctop: torch.Tensor, ldcumask: torch.Tensor,
              pa: torch.Tensor, pb: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
    """Solve the bidiagonal system M·U = R (forward substitution only).

    Port of cubidiag.F90. C=0 (upper off-diagonal), so only forward
    substitution is needed. Operates per-column with a cumulative solve from
    cloud-top downward.

    Args:
        kctop:    (nlon,) cloud-top level (1-based, top-down).
        ldcumask: (nlon, klev) boolean mask of active convective levels.
        pa:       (nlon, klev) lower off-diagonal A(k).
        pb:       (nlon, klev) diagonal B(k).
        pr:       (nlon, klev) RHS.

    Returns:
        pu: (nlon, klev) solution.
    """
    nlon, klev = pr.shape
    dt = pr.dtype
    dev = pr.device
    pu = torch.zeros(nlon, klev, dtype=dt, device=dev)
    # Forward substitution: JK = 2..KLEV (0-based 1..klev-1).
    for jk in range(1, klev):
        mask = ldcumask[:, jk]
        if not mask.any():
            continue
        # JK == KCTOP-1: first active level above cloud top.
        is_top = mask & (jk == kctop - 1)
        is_below = mask & (jk > kctop - 1)
        zbet_top = 1.0 / (pb[:, jk] + 1e-35)
        pu[:, jk] = torch.where(
            is_top,
            pr[:, jk] * zbet_top,
            torch.where(
                is_below,
                (pr[:, jk] - pa[:, jk] * pu[:, jk - 1]) * zbet_top,
                pu[:, jk],
            ),
        )
    return pu


def cuctracer(
    ptsphy: float,
    paph: torch.Tensor,       # (nlon, klev+1) half-level pressure [Pa]
    pap: torch.Tensor,        # (nlon, klev) full-level pressure [Pa]
    pmfu: torch.Tensor,       # (nlon, klev) updraft mass flux
    pmfd: torch.Tensor,       # (nlon, klev) downdraft mass flux (negative)
    pmfuo: torch.Tensor,      # (nlon, klev) original updraft mass flux
    pmfdo: torch.Tensor,      # (nlon, klev) original downdraft mass flux
    pudrate: torch.Tensor,    # (nlon, klev) updraft detrainment
    pddrate: torch.Tensor,    # (nlon, klev) downdraft detrainment
    pdmfup: torch.Tensor,     # (nlon, klev) updraft precip production
    pdmfdp: torch.Tensor,     # (nlon, klev) downdraft precip production
    pcen: torch.Tensor,       # (nlon, klev, ktrac) environment tracer conc.
    pscav: torch.Tensor,      # (ktrac,) scavenging coefficients
    kctop: torch.Tensor,      # (nlon,) cloud-top level (1-based top-down)
    kdtop: torch.Tensor,      # (nlon,) downdraft-top level (1-based)
    ktype: torch.Tensor,      # (nlon,) convection type (1=deep,2=shallow,3=mid)
    ldcum: torch.Tensor,      # (nlon,) convective flag
    lddraf: torch.Tensor,     # (nlon,) downdraft flag
    ptenc: torch.Tensor,      # (nlon, klev, ktrac) IN/OUT tendency (1/s)
    rmfsolct: float = 1.0,
    rmfcmin: float = 1.0e-8,
    rmfadvw: float = 0.0,
    rmfadvwdd: float = 0.0,
) -> torch.Tensor:
    """Convective tracer transport — port of cuctracer.F90.

    Updates and returns ``ptenc`` (the tracer tendency, 1/s) in-place semantics
    (returns the updated tensor). All inputs are top-down (index 0 = TOA).

    Args match the Fortran argument list. ``rmfsolct`` selects the solver:
    0 = explicit, >0 = implicit (bidiagonal).

    Returns:
        ptenc: (nlon, klev, ktrac) updated tendency.
    """
    nlon, klev, ktrac = pcen.shape
    dt = pcen.dtype
    dev = pcen.device
    zrmfsolct = float(rmfsolct)
    zrmfcmin = float(rmfcmin)
    zimp = 1.0 - zrmfsolct
    ztsphy = 1.0 / float(ptsphy)

    # RMFADVW: applied only for deep convection (ktype==1).
    zadvw = torch.where(ktype == 1, torch.full_like(ktype, float(rmfadvw), dtype=dt),
                        torch.zeros(nlon, dtype=dt, device=dev))

    # Cumulus mask + dp setup (JK=2..KLEV → 0-based 1..klev-1).
    llcumask = torch.zeros(nlon, klev, dtype=torch.bool, device=dev)
    zdp = torch.zeros(nlon, klev, dtype=dt, device=dev)
    for jk in range(1, klev):
        active = ldcum
        llcumask[:, jk] = active & (jk >= kctop - 1)
        zdp[:, jk] = torch.where(active, _RG / (paph[:, jk + 1] - paph[:, jk]).clamp(min=1e-10),
                                 zdp[:, jk])

    # Working arrays.
    zcen = pcen.clone()                          # (nlon, klev, ktrac) env at full levels
    zcu = torch.zeros_like(pcen)                 # updraft values
    zcd = torch.zeros_like(pcen)                 # downdraft values
    ztenc = torch.zeros_like(pcen)               # local tendency
    zmfc = torch.zeros_like(pcen)                # fluxes

    for jn in range(ktrac):
        # ---- 1.0 Tracers at half-levels ----
        # Fortran: ZCEN(JL,JK)=PCEN(JL,JK); ZCD(JL,JK)=PCEN(JL,IK=JK-1);
        #          ZCU(JL,JK)=PCEN(JL,JK-1).
        for jk in range(1, klev):
            ik = jk - 1
            zcen[:, jk, jn] = pcen[:, jk, jn]
            zcd[:, jk, jn] = pcen[:, ik, jn]
            zcu[:, jk, jn] = pcen[:, ik, jn]
        zcu[:, klev - 1, jn] = pcen[:, klev - 1, jn]

        # ---- 2.0 Updraft values (JK=KLEV-1 down to 3) ----
        for jk in range(klev - 2, 2, -1):       # 0-based: klev-3 down to 2
            ik = jk + 1
            mask = llcumask[:, jk]
            if not mask.any():
                continue
            zerate = pmfu[:, jk] - pmfu[:, ik] + pudrate[:, jk]
            zmfa = 1.0 / torch.clamp(pmfu[:, jk], min=zrmfcmin)
            # JK >= KCTOP
            top_mask = mask & (jk >= kctop)
            numerator = (pmfu[:, ik] * zcu[:, ik, jn] + zerate * pcen[:, jk, jn]
                         - (pudrate[:, jk] + pdmfup[:, jk] * float(pscav[jn])) * zcu[:, ik, jn])
            zcu[:, jk, jn] = torch.where(top_mask, numerator * zmfa, zcu[:, jk, jn])

        # ---- 3.0 Downdraft values (JK=3..KLEV) ----
        for jk in range(2, klev):               # 0-based: 2..klev-1
            ik = jk - 1
            # JK == KDTOP
            dt_mask = lddraf & (jk == kdtop)
            zcd[:, jk, jn] = torch.where(
                dt_mask,
                0.1 * zcu[:, jk, jn] + 0.9 * pcen[:, ik, jn],
                zcd[:, jk, jn])
            # JK > KDTOP
            dd_mask = lddraf & (jk > kdtop)
            zerate = -pmfd[:, jk] + pmfd[:, ik] + pddrate[:, jk]
            zmfa = 1.0 / torch.minimum(torch.full_like(pmfd[:, jk], -zrmfcmin), pmfd[:, jk])
            numerator = (pmfd[:, ik] * zcd[:, ik, jn] - zerate * pcen[:, ik, jn]
                         + (pddrate[:, jk] + pdmfdp[:, jk] * float(pscav[jn])) * zcd[:, ik, jn])
            zcd[:, jk, jn] = torch.where(dd_mask, numerator * zmfa, zcd[:, jk, jn])

        # ---- Adjust ZCD at KLEV to avoid negatives ----
        jk = klev - 1
        ik = jk - 1
        posi = -zdp[:, jk] * (pmfu[:, jk] * zcu[:, jk, jn] + pmfd[:, jk] * zcd[:, jk, jn]
                              - (pmfu[:, jk] + pmfd[:, jk]) * pcen[:, ik, jn])
        neg_check = lddraf & (pcen[:, jk, jn] + posi * ptsphy < 0.0)
        zmfa = 1.0 / torch.minimum(torch.full_like(pmfd[:, jk], -zrmfcmin), pmfd[:, jk])
        zcd_adj = (((pmfu[:, jk] + pmfd[:, jk]) * pcen[:, ik, jn] - pmfu[:, jk] * zcu[:, jk, jn]
                    + pcen[:, jk, jn] / (ptsphy * zdp[:, jk])) * zmfa)
        zcd[:, jk, jn] = torch.where(neg_check, zcd_adj, zcd[:, jk, jn])

    # ---- 4.0 + 5.0 Fluxes and tendencies ----
    for jn in range(ktrac):
        for jk in range(1, klev):
            ik = jk - 1
            mask = llcumask[:, jk]
            if not mask.any():
                continue
            zmfa_flux = pmfu[:, jk] + pmfd[:, jk]
            zmfc[:, jk, jn] = torch.where(
                mask,
                pmfu[:, jk] * zcu[:, jk, jn] + pmfd[:, jk] * zcd[:, jk, jn]
                - zimp * zmfa_flux * zcen[:, ik, jn],
                zmfc[:, jk, jn])

        # Tendencies JK=2..KLEV-1 (0-based 1..klev-2)
        for jk in range(1, klev - 1):
            ik = jk + 1
            mask = llcumask[:, jk]
            ztenc[:, jk, jn] = torch.where(
                mask, zdp[:, jk] * (zmfc[:, ik, jn] - zmfc[:, jk, jn]), ztenc[:, jk, jn])

        # JK=KLEV (surface)
        jk = klev - 1
        ztenc[:, jk, jn] = torch.where(
            ldcum, -zdp[:, jk] * zmfc[:, jk, jn], ztenc[:, jk, jn])

    # ---- 6.0/7.0 Update tendencies ----
    if zrmfsolct == 0.0:
        # Explicit: PTENC += ZTENC
        for jn in range(ktrac):
            for jk in range(1, klev):
                ptenc[:, jk, jn] = torch.where(
                    llcumask[:, jk],
                    ptenc[:, jk, jn] + ztenc[:, jk, jn],
                    ptenc[:, jk, jn])
    else:
        # Implicit: bidiagonal solve.
        zb = torch.ones(nlon, klev, dtype=dt, device=dev)
        llcumbas = llcumask.clone()
        for jn in range(ktrac):
            # Fill A, B, RHS.
            for jk in range(1, klev):
                ik = jk + 1
                im = jk - 1
                mask = llcumbas[:, jk]
                zzp = zrmfsolct * zdp[:, jk] * ptsphy
                zmfc[:, jk, jn] = torch.where(mask, -zzp * (pmfu[:, jk] + pmfd[:, jk]),
                                              zmfc[:, jk, jn])
                if jk < klev - 1:
                    zb[:, jk] = torch.where(mask, 1.0 + zzp * (pmfu[:, ik] + pmfd[:, ik]), zb[:, jk])
                else:
                    zb[:, jk] = torch.where(mask, torch.ones_like(zb[:, jk]), zb[:, jk])
                # Advective correction term.
                zzp_adv = (_RG * (pmfuo[:, jk] + rmfadvwdd * pmfdo[:, jk])
                           / (pap[:, jk] - pap[:, im]).clamp(min=1e-10) * ptsphy * zadvw)
                zc_adv = zzp_adv * (pcen[:, im, jn] - pcen[:, jk, jn])
                ztenc[:, jk, jn] = torch.where(
                    mask, ztenc[:, jk, jn] * ptsphy + pcen[:, jk, jn] - zc_adv,
                    ztenc[:, jk, jn])
            # Solve bidiagonal.
            zr1 = _cubidiag(kctop, llcumbas, zmfc[:, :, jn], zb, ztenc[:, :, jn])
            # Tendencies.
            for jk in range(1, klev):
                ptenc[:, jk, jn] = torch.where(
                    llcumbas[:, jk],
                    ptenc[:, jk, jn] + (zr1[:, jk] - pcen[:, jk, jn]) * ztsphy,
                    ptenc[:, jk, jn])

    return ptenc


__all__ = ["cuctracer"]
