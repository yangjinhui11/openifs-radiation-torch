"""Batched torch LW driver: RRTM_RRTM_140GP.

Chains ECRT → SETCOEF → GASABS1A → RTRN1A, mirroring the Fortran
``rrtm_rrtm_140gp.F90`` production call sequence.
"""
from __future__ import annotations
import math
import torch
from .ecrt import prepare_rrtm_profile
from .setcoef import rrtm_setcoef_140gp
from .taumol import rrtm_gasabs1a_140gp
from .rtrn1a import rrtm_rtrn1a_140gp
from ..radlsw_inputs import RadLSWInputs


def rrtm_rrtm_140gp(
    inputs: RadLSWInputs,
    novlp: int = 1,
) -> dict:
    """Full LW radiation step: inputs → fluxes.

    Parameters
    ----------
    inputs : RadLSWInputs
        Upward state from ``assemble_radlsw_inputs`` (top-down).
    novlp : int
        Cloud overlap scheme (1 = max-random, production default).

    Returns
    -------
    dict with:
        flux_up   : (nlon, nlev+1)  upward LW flux (W/m²)
        flux_down : (nlon, nlev+1)  downward LW flux (W/m², positive down)
        flux_up_clear   : (nlon, nlev+1)  clear-sky upward
        flux_down_clear : (nlon, nlev+1)  clear-sky downward
        emissivity : (nlon,)  surface LW emissivity
        heating_rate : (nlon, nlev)  K/day
    """
    # ---- 1. ECRT: pack RadLSWInputs → RRTM bottom-up profile ----
    prof = prepare_rrtm_profile(inputs, novlp=novlp)
    nlev, nlon = prof.pavel.shape
    dt = prof.pavel.dtype
    dev = prof.pavel.device

    # ---- 2. SETCOEF: interpolation indices + column amounts ----
    scoef = rrtm_setcoef_140gp(prof.pavel, prof.tavel, prof.coldry, prof.wbrodl, prof.wkl)

    # ---- 3. GASABS1A: per-g-point optical depths + Planck fractions ----
    oneminus = 0.999999
    patr1, pod, ptf1, pfrac_140 = rrtm_gasabs1a_140gp(
        prof.pavel, prof.coldry, scoef["colbrd"], prof.wx, prof.tauaer,
        scoef["fac00"], scoef["fac01"], scoef["fac10"], scoef["fac11"],
        scoef["forfac"], scoef["forfrac"], scoef["indfor"],
        scoef["jp"], scoef["jt"], scoef["jt1"], oneminus,
        scoef["colh2o"], scoef["colco2"], scoef["colo3"],
        scoef["coln2o"], scoef["colch4"], scoef["colo2"], scoef["co2mult"],
        scoef["laytrop"], scoef["layswtch"], scoef["laylow"],
        scoef["selffac"], scoef["selffrac"], scoef["indself"],
        scoef["minorfrac"], scoef["indminor"],
        scoef["scaleminor"], scoef["scaleminorn2"],
        scoef["rat_h2oco2"], scoef["rat_h2oco2_1"],
        scoef["rat_h2oo3"], scoef["rat_h2oo3_1"],
        scoef["rat_h2on2o"], scoef["rat_h2on2o_1"],
        scoef["rat_h2och4"], scoef["rat_h2och4_1"],
        scoef["rat_n2oco2"], scoef["rat_n2oco2_1"],
        scoef["rat_o3co2"], scoef["rat_o3co2_1"],
    )

    # ---- 4. Cloud detection ----
    icldlyr = (prof.cldfrac > 1e-6).to(torch.int64)

    # ---- 5. RTRN1A: radiative transfer ----
    # Permute arrays from (nlev, nlon, *) → (nlon, nlev, *) for RTRN1A
    def _p2d(x):
        """(nlev, nlon) → (nlon, nlev)"""
        return x.permute(1, 0)
    def _p3d(x):
        """(nlev, nlon, ng) → (nlon, ng, nlev)"""
        return x.permute(1, 2, 0)
    def _p2d_cld(x):
        """(nlev, nlon, nb) → (nlon, nlev, nb)"""
        return x.permute(1, 0, 2)

    rt_out = rrtm_rtrn1a_140gp(
        nlev=nlev,
        istart=1, iend=16,
        icldlyr=_p2d(icldlyr),
        cldfrac=_p2d(prof.cldfrac),
        taucld=_p2d_cld(prof.taucld),
        atr1=_p3d(patr1),
        od=_p3d(pod),
        tf1=_p3d(ptf1),
        tavel=_p2d(prof.tavel),
        tz=_p2d(prof.tz),
        tbound=prof.tbound,
        pfrac=_p3d(pfrac_140),
        semiss=prof.semiss,
    )

    # ---- 6. Flux scaling (radiance → W/m²) ----
    flux_fac = math.pi * 2.0e4

    # RTRN1A returns (nlon, nlev+1) already
    return {
        "flux_up": rt_out["totuflux"] * flux_fac,
        "flux_down": rt_out["totdflux"] * flux_fac,
        "flux_up_clear": rt_out["totufluc"] * flux_fac,
        "flux_down_clear": rt_out["totdfluc"] * flux_fac,
        "emissivity": rt_out["semislw"],
        "heating_rate": _heating_rate(
            rt_out["totuflux"], rt_out["totdflux"],
            inputs.pdp.permute(1, 0), flux_fac
        ),
    }


def _heating_rate(flux_up, flux_down, dp, flux_fac):
    """Compute LW heating rate (K/day) from flux divergence.

    flux_up/down: (nlon, nlev+1) radiance
    dp: (nlon, nlev) Pa layer thickness
    """
    # Net flux at each half-level (positive = upward)
    net_top = flux_up[:, :-1] - flux_down[:, :-1]   # top of layer
    net_bot = flux_up[:, 1:] - flux_down[:, 1:]      # bottom of layer
    # Convergence = net entering layer - net leaving layer
    # = (net_top - net_bot) — positive = heating
    g = 9.80665; cp = 1004.707795; sec_per_day = 86400.0
    return (net_top - net_bot) * flux_fac * g / (cp * dp.clamp(min=1.0)) * sec_per_day



__all__ = ["rrtm_rrtm_140gp"]
