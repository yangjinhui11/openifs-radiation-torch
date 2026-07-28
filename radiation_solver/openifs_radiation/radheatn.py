"""Port of RADHEATN — radiation flux interpolation every timestep.

Fortran reference: openifs-48r1/ifs-source/arpifs/phys_radi/radheatn.F90

RADHEATN is called every physics timestep (from CALLPAR → RADFLUX_LAYER).
It uses the transmissivities/emissivities stored from the last full
radiation call (NRADFR steps ago) and combines them with the current
atmospheric state (T, Q, surface skin temperature, solar zenith angle)
to compute:
  * Shortwave fluxes: PFRSO = ZI0 * PTRSOL (+ Manners correction)
  * Longwave fluxes:  PFRTH = PEMTD (+ optional surface-T correction)
  * Heating rates:    divergence of net flux

This is the "cheap" radiation branch that runs every step, as opposed to
the "full" radiation call (RADINA→RADLSW) which runs every NRADFR steps.

Production defaults:
  LAPPROXLWUPDATE = .TRUE.  (update LW for surface temperature change)
  LMANNERSSWUPDATE = .TRUE. (correct SW for solar zenith angle change)

Convention: all arrays (nlon, nlev) or (nlon, nlev+1), top-down (TOA=0).
"""
from __future__ import annotations
import torch
from dataclasses import dataclass

# Physical constants (YOMCST).
_RG = 9.80665       # gravity
_RSIGMA = 5.670374e-8  # Stefan-Boltzmann
_RCPD = 1004.7      # specific heat of dry air
_RVTMP2 = 461.6 / 287.0 - 1.0  # Rv/Rd - 1
_RII0 = 1361.0      # solar constant (TBC from config)
_RRAE = 0.637       # Earth curvature parameter

_ZLWDNDIFFRATIO = 0.2  # fraction of LW up change applied to LW down


@dataclass
class RadHeatState:
    """Radiation state stored from the last full radiation call (RADINA).

    These are the transmissivities/emissivities that RADHEATN uses to
    interpolate fluxes every timestep. They are updated only when the
    full radiation scheme (RADINA → RADLSW) runs (every NRADFR steps).
    """
    # Shortwave transmissivities (stored from RADLSW output)
    ptrsol: torch.Tensor       # (nlon, nlev+1) net SW transmissivity
    ptrsoc: torch.Tensor       # (nlon, nlev+1) clear-sky net SW transmissivity
    ptrsod: torch.Tensor       # (nlon,) surface downwelling SW transmissivity
    ptrsodc: torch.Tensor      # (nlon,) clear-sky surface downwelling SW
    pfdiri: torch.Tensor       # (nlon,) surface direct-beam SW transmissivity
    pcdiri: torch.Tensor       # (nlon,) clear-sky surface direct-beam SW

    # Longwave net fluxes (stored from RADLSW output)
    pemtd: torch.Tensor        # (nlon, nlev+1) net LW flux (W/m2)
    pemtec: torch.Tensor       # (nlon, nlev+1) clear-sky net LW flux
    ptrthd: torch.Tensor       # (nlon,) surface downwelling LW (broadband)
    ptrthdc: torch.Tensor      # (nlon,) clear-sky surface downwelling LW

    # LW derivative (for approximate LW update)
    plwderivative: torch.Tensor  # (nlon, nlev+1) d(LW_up)/d(T_surf)

    # Solar zenith used at last radiation call
    pmu0m: torch.Tensor        # (nlon,) cos(sza) at last full radiation

    # Surface diagnostics (stored from RADLSW)
    puvdfi: torch.Tensor       # (nlon,) UV diffuse transmissivity
    pparfi: torch.Tensor       # (nlon,) PAR transmissivity
    pparcfi: torch.Tensor      # (nlon,) clear-sky PAR
    ptincfi: torch.Tensor      # (nlon,) total direct transmissivity

    # Emissivity
    pemis: torch.Tensor        # (nlon,) surface broadband LW emissivity


@dataclass
class RadHeatOutput:
    """Output of RADHEATN: fluxes and heating rates for this timestep."""
    pfrso: torch.Tensor        # (nlon, nlev+1) net SW flux
    pfrth: torch.Tensor        # (nlon, nlev+1) net LW flux (updated)
    phrsw: torch.Tensor        # (nlon, nlev) SW heating rate
    phrlw: torch.Tensor        # (nlon, nlev) LW heating rate
    pfrsoc: torch.Tensor       # (nlon, 2) clear-sky net SW [TOA, sfc]
    pfrthc: torch.Tensor       # (nlon, 2) clear-sky net LW [TOA, sfc]
    pfrsod: torch.Tensor       # (nlon,) surface SW downwelling
    pfrthd: torch.Tensor       # (nlon,) surface LW downwelling
    pdsrp: torch.Tensor        # (nlon,) surface direct beam SW


def radheatn(
    state: RadHeatState,
    paphm1: torch.Tensor,      # (nlon, nlev+1) half-level pressure [Pa]
    pqm1: torch.Tensor,        # (nlon, nlev) specific humidity
    ptsm1m: torch.Tensor,      # (nlon,) skin temperature [K]
    pmu0: torch.Tensor,        # (nlon,) current cos(solar zenith angle)
    pte: torch.Tensor | None = None,  # (nlon, nlev) temperature tendency (in/out)
    approxlw_update: bool = True,
    manners_sw_update: bool = True,
    rii0: float = _RII0,
) -> RadHeatOutput:
    """Radiation flux interpolation — port of RADHEATN.

    Uses stored transmissivities from the last full radiation call and
    the current atmospheric state to compute fluxes and heating rates.

    Args:
        state: RadHeatState with stored radiation fields.
        paphm1: half-level pressures (Pa), top-down.
        pqm1: specific humidity (kg/kg), top-down.
        ptsm1m: skin temperature (K).
        pmu0: current cosine of solar zenith angle.
        pte: temperature tendency to update (K/s). If None, a zero tensor is used.
        approxlw_update: if True, update LW flux for surface temperature change.
        manners_sw_update: if True, correct SW for solar zenith angle change.

    Returns:
        RadHeatOutput with fluxes and heating rates.
    """
    nlon, nlev = pqm1.shape
    dt = pqm1.dtype
    dev = pqm1.device
    if pte is None:
        pte = torch.zeros(nlon, nlev, dtype=dt, device=dev)

    # ── §2. Incident solar radiation ──
    eps = 100.0 * torch.finfo(dt).eps
    zcons3 = _RG / _RCPD

    # ZI0 = RII0 * PMU0 (if daytime, else 0)
    zi0 = torch.where(pmu0 >= eps, rii0 * pmu0, torch.zeros_like(pmu0))
    pdsrp = torch.where(pmu0 >= eps, torch.ones_like(pmu0), torch.zeros_like(pmu0))

    # ── §4.0.1 Optional LW update ──
    if approxlw_update:
        # Change to upwelling LW at surface due to skin-T change
        zth4 = _RSIGMA * ptsm1m ** 4
        zlwupsurfdiff = state.pemtd[:, -1] + state.pemis * (zth4 - state.ptrthd)
        zlwdnsurfdiff = _ZLWDNDIFFRATIO * zlwupsurfdiff

        # Update net LW at each half-level using the LW derivative
        # ZLWDNDERIVATIVE = (PLWDERIVATIVE[:,jk] - PLWDERIVATIVE[:,0]) / (1 - PLWDERIVATIVE[:,0])
        plw = state.plwderivative
        denom = (1.0 - plw[:, 0:1]).clamp(min=1e-10)
        zlwdnderivative = (plw - plw[:, 0:1]) / denom
        pfrth = state.pemtd + zlwdnsurfdiff.unsqueeze(1) * zlwdnderivative \
                - zlwupsurfdiff.unsqueeze(1) * plw
        # Surface LW downwelling
        pfrthd = state.ptrthd + zlwdnsurfdiff
    else:
        pfrth = state.pemtd.clone()
        pfrthd = state.ptrthd.clone()

    # ── §4.0.2 Optional SW Manners update ──
    if manners_sw_update:
        zcrae = _RRAE * (_RRAE + 2.0)
        # Current and radiation-call zenith angles with curvature
        mu0c = pmu0.clamp(min=eps)
        zmu0curv = _RRAE / (torch.sqrt(mu0c ** 2 + zcrae) - mu0c)
        mu0mc = state.pmu0m.clamp(min=eps)
        zmu0mcrcurv = _RRAE / (torch.sqrt(mu0mc ** 2 + zcrae) - mu0mc)
        zmu0ratio = zmu0mcrcurv / zmu0curv

        # Direct beam transmissivity correction: exp(-tau/mu0_new) = old^(mu0_old/mu0_new)
        daytime = (pmu0 >= eps) & (state.pfdiri >= eps) & (state.ptrsod >= eps)
        zswdirecttrans = torch.where(daytime, state.pfdiri ** zmu0ratio, state.pfdiri)
        zswnetdiff = torch.where(
            daytime,
            0.5 * zi0 * (zswdirecttrans - state.pfdiri) * state.ptrsol[:, -1] / state.ptrsod.clamp(min=1e-30),
            torch.zeros_like(zi0))
        zswsur_scaling = torch.where(
            daytime,
            1.0 + 0.5 * (zswdirecttrans - state.pfdiri) / state.ptrsod.clamp(min=1e-30),
            torch.ones_like(zi0))
        zswdirect_scaling = torch.where(daytime, zswdirecttrans / state.pfdiri.clamp(min=1e-30), torch.ones_like(zi0))

        # Clear-sky equivalents
        daytime_c = (pmu0 >= eps) & (state.pcdiri >= eps) & (state.ptrsodc >= eps)
        zswdirecttransc = torch.where(daytime_c, state.pcdiri ** zmu0ratio, state.pcdiri)
        zswnetdiffc = torch.where(
            daytime_c,
            0.5 * zi0 * (zswdirecttransc - state.pcdiri) * state.ptrsoc[:, -1] / state.ptrsodc.clamp(min=1e-30),
            torch.zeros_like(zi0))
        zswsur_scalingc = torch.where(
            daytime_c,
            1.0 + 0.5 * (zswdirecttransc - state.pcdiri) / state.ptrsodc.clamp(min=1e-30),
            torch.ones_like(zi0))
        zswdirect_scalingc = torch.where(daytime_c, zswdirecttransc / state.pcdiri.clamp(min=1e-30), torch.ones_like(zi0))
    else:
        zswnetdiff = torch.zeros_like(zi0)
        zswnetdiffc = torch.zeros_like(zi0)
        zswsur_scaling = torch.ones_like(zi0)
        zswsur_scalingc = torch.ones_like(zi0)
        zswdirect_scaling = torch.ones_like(zi0)
        zswdirect_scalingc = torch.ones_like(zi0)

    # ── §4. Scale SW transmissivities to fluxes ──
    pfrso = zi0.unsqueeze(1) * state.ptrsol + zswnetdiff.unsqueeze(1)  # (nlon, nlev+1)
    pfrsoc = torch.stack([
        zi0 * state.ptrsoc[:, 0] + zswnetdiffc,       # TOA
        zi0 * state.ptrsoc[:, -1] + zswnetdiffc,       # surface
    ], dim=1)  # (nlon, 2)
    pfrsod = zi0 * state.ptrsod * zswsur_scaling
    pfrsodc = zi0 * state.ptrsodc * zswsur_scalingc

    # Floor SW surface at epsilon for stability
    pfrso[:, -1] = torch.clamp(pfrso[:, -1], min=eps)

    # ── §4.1-4.2 Heating rates from flux divergence ──
    # Top-of-layer fluxes
    zfso_t = pfrso[:, 0]       # SW top
    zfte_t = pfrth[:, 0]       # LW top
    zfswt_c = zi0 * state.ptrsoc[:, 0]
    zflwt_c = state.pemtec[:, 0]
    pfrthc_toa = state.pemtec[:, 0]

    phrsw = torch.zeros(nlon, nlev, dtype=dt, device=dev)
    phrlw = torch.zeros(nlon, nlev, dtype=dt, device=dev)
    phrsc = torch.zeros(nlon, nlev, dtype=dt, device=dev)
    phrlc = torch.zeros(nlon, nlev, dtype=dt, device=dev)

    zfso = zfso_t.clone()
    zfte = zfte_t.clone()
    zfswt_c_cur = zfswt_c.clone()
    zflwt_c_cur = zflwt_c.clone()

    for jk in range(nlev):
        # Bottom-of-layer fluxes
        zfsob = pfrso[:, jk + 1]
        zfteb = pfrth[:, jk + 1]
        # Flux-to-heating-rate conversion
        dp = (paphm1[:, jk + 1] - paphm1[:, jk]).clamp(min=1.0)
        zfac = -zcons3 / (dp * (1.0 + _RVTMP2 * pqm1[:, jk]))
        # Heating rates
        pte[:, jk] = pte[:, jk] + zfac * ((zfsob + zfteb) - (zfso + zfte))
        phrsw[:, jk] = zfac * (zfsob - zfso)
        phrlw[:, jk] = zfac * (zfteb - zfte)
        # Clear-sky
        zfswb_c = zi0 * state.ptrsoc[:, jk + 1]
        zflwb_c = state.pemtec[:, jk + 1]
        phrsc[:, jk] = zfac * (zfswb_c - zfswt_c_cur)
        phrlc[:, jk] = zfac * (zflwb_c - zflwt_c_cur)
        # Swap for next layer
        zfso = zfsob
        zfte = zfteb
        zfswt_c_cur = zfswb_c
        zflwt_c_cur = zflwb_c

    pfrthc_sfc = state.pemtec[:, -1]
    pfrthc = torch.stack([pfrthc_toa, pfrthc_sfc], dim=1)  # (nlon, 2)

    # ── §4.5 Surface diagnostics ──
    pdsrp = pdsrp * rii0 * state.pfdiri * zswdirect_scaling

    return RadHeatOutput(
        pfrso=pfrso, pfrth=pfrth,
        phrsw=phrsw, phrlw=phrlw,
        pfrsoc=pfrsoc, pfrthc=pfrthc,
        pfrsod=pfrsod, pfrthd=pfrthd,
        pdsrp=pdsrp,
    )


__all__ = ["radheatn", "RadHeatState", "RadHeatOutput"]
