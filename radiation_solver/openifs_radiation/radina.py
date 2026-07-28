"""RADINA — radiation interface from callpar to LW + SW schemes.

Fortran reference: radina.F90 (424 lines)

Orchestrates:
  1. Half-level temperature interpolation
  2. Solar zenith angle Earth-curvature correction
  3. Humidity / cloud safety clamps
  4. Ozone (radozc) + aerosol (radaca)
  5. LW chain: ECRT → SETCOEF → GASABS → RTRN1A
  6. SW chain: 6-band clear-sky solver
  7. Flux packaging

All arrays are TOA→surface (vertical axis first), matching OpenIFS convention.
"""
from __future__ import annotations

import math
import torch
from dataclasses import dataclass

from .radlsw_inputs import RadLSWInputs
from .rrtm_lw.driver import rrtm_rrtm_140gp
from .classic_sw.driver import sw_solver


@dataclass
class RadinaState:
    """All inputs to RADINA, matching the Fortran argument list.

    All 2-D state arrays are (nlon, nlev) TOA→surface.
    Half-level arrays are (nlon, nlev+1).
    """
    # ── Grid ──
    nlev: int
    nlon: int
    pgemu: torch.Tensor          # (nlon,) sine of latitude
    pslm: torch.Tensor           # (nlon,) land-sea mask [0=sea, 1=land]

    # ── Pressure ──
    paprs: torch.Tensor          # (nlon, nlev+1) half-level pressure [Pa]
    paprsf: torch.Tensor         # (nlon, nlev)   full-level pressure [Pa]
    pdp: torch.Tensor            # (nlon, nlev)   layer thickness [Pa]

    # ── Thermodynamics ──
    pt: torch.Tensor             # (nlon, nlev) full-level T [K]
    pts: torch.Tensor            # (nlon,)       surface T [K]
    pq: torch.Tensor             # (nlon, nlev) specific humidity [kg/kg]
    pqs: torch.Tensor            # (nlon, nlev) saturation q [kg/kg]

    # ── Cloud condensate ──
    pclfr: torch.Tensor          # (nlon, nlev) cloud fraction
    pqiwp: torch.Tensor          # (nlon, nlev) ice water [kg/kg]
    pqlwp: torch.Tensor          # (nlon, nlev) liquid water [kg/kg]
    pqrwp: torch.Tensor          # (nlon, nlev) rain water [kg/kg]
    pqswp: torch.Tensor          # (nlon, nlev) snow water [kg/kg]

    # ── Surface ──
    palbd: torch.Tensor          # (nlon, 6) diffuse albedo per SW band
    palbp: torch.Tensor          # (nlon, 6) direct albedo per SW band
    pemir: torch.Tensor          # (nlon,) LW emissivity (non-window)
    pemiw: torch.Tensor          # (nlon,) LW emissivity (window)
    pmu0: torch.Tensor           # (nlon,) cosine of solar zenith

    # ── CCN ──
    pccnl: torch.Tensor          # (nlon,) CCN over land
    pccno: torch.Tensor          # (nlon,) CCN over ocean

    # ── Well-mixed gases (vmr) ──
    co2_vmr: torch.Tensor | None = None
    ch4_vmr: torch.Tensor | None = None
    n2o_vmr: torch.Tensor | None = None

    # ── Ozone (optional; if None, uses radozc climatology) ──
    o3_mmr: torch.Tensor | None = None


@dataclass
class RadinaOutput:
    """RADINA outputs matching Fortran intent(out)."""
    # LW fluxes (nlon, nlev+1), W/m2, TOA→surface
    pemtd: torch.Tensor          # total downward LW flux
    pemtc: torch.Tensor          # clear-sky downward LW flux
    # SW fluxes (nlon, nlev+1), W/m2, TOA→surface
    ptrso: torch.Tensor          # total downward SW flux
    ptrsoc: torch.Tensor         # clear-sky downward SW flux
    # Scalars
    pemit: torch.Tensor          # (nlon,) surface broadband LW emissivity
    pdsrp: torch.Tensor          # (nlon,) direct SW at surface
    pfrted: torch.Tensor         # (nlon,) surface downwelling LW flux
    ptrsod: torch.Tensor         # (nlon,) surface SW transmissivity
    pth: torch.Tensor            # (nlon, nlev+1) half-level temperature
    # Heating rate proxy (K/day from flux divergence)
    heat_lw: torch.Tensor        # (nlon, nlev)
    heat_sw: torch.Tensor        # (nlon, nlev)


# ── Physical constants ─────────────────────────────────────────────────
_RRAE = 0.637    # Earth radius / (radius + atmosphere height) ≈ 12742/20000
_RII0 = 1361.0   # Solar constant W/m2
_REPH2O = 1e-12  # minimum humidity
_REPCLC = 1e-5   # minimum cloud fraction for rain/snow division
_RCCARDI = 4.0e-4   # default CO2 vmr
_RCH4 = 1.7e-6
_RN2O = 3.2e-7


def _interp_half_level_temp(
    pt: torch.Tensor,       # (nlon, nlev) full-level T
    paprsf: torch.Tensor,   # (nlon, nlev) full-level pressure
    paprs: torch.Tensor,    # (nlon, nlev+1) half-level pressure
    pts: torch.Tensor,      # (nlon,) surface T
) -> torch.Tensor:
    """Interpolate temperature to half-levels (Fortran lines 217-235).

    Returns pth: (nlon, nlev+1) half-level T, TOA→surface.
    """
    nlon, nlev = pt.shape
    dt = pt.dtype
    dev = pt.device
    pth = torch.zeros(nlon, nlev + 1, dtype=dt, device=dev)

    # Interior: JK=2..nlev → Fortran JK=2..KLEV
    for jk in range(1, nlev):
        z1s = paprsf[:, jk - 1] * (paprsf[:, jk] - paprs[:, jk])
        z2s = paprsf[:, jk] * (paprs[:, jk] - paprsf[:, jk - 1])
        z3s = paprs[:, jk] * (paprsf[:, jk] - paprsf[:, jk - 1])
        pth[:, jk] = (pt[:, jk - 1] * z1s + pt[:, jk] * z2s) / z3s.clamp(min=1e-10)

    # TOA extrapolation (JK=1)
    pth[:, 0] = pt[:, 0] - paprsf[:, 0] * (pt[:, 0] - pth[:, 1]) / (
        paprsf[:, 0] - paprs[:, 1]
    ).clamp(min=1e-10)

    # Surface = PTS
    pth[:, nlev] = pts
    return pth


def _earth_curvature_mu0(pmu0: torch.Tensor) -> torch.Tensor:
    """Earth-curvature-corrected solar zenith (Fortran lines 244-250)."""
    zcrae = _RRAE * (_RRAE + 2.0)
    return torch.where(
        pmu0 > 1e-10,
        _RRAE / (torch.sqrt(pmu0 ** 2 + zcrae) - pmu0),
        _RRAE / torch.sqrt(torch.tensor(zcrae, dtype=pmu0.dtype, device=pmu0.device)),
    )


def _safety_clamps(
    state: RadinaState,
) -> dict:
    """Apply humidity/cloud safety clamps (Fortran lines 254-288).

    Returns clamped copies of q, qs, clfr, qiwp, qlwp, qrwp, qswp.
    """
    zq = state.pq.clamp(min=2.0 * _REPH2O)
    zq = torch.minimum(zq, state.pqs * (1.0 - _REPH2O))
    zqs = state.pqs.clamp(min=2.0 * _REPH2O)
    zclfr = state.pclfr.clamp(0.0, 1.0)
    zqlwp = state.pqlwp.clamp(min=_REPH2O)
    zqiwp = state.pqiwp.clamp(min=_REPH2O)
    zqrwp = torch.where(
        zclfr > _REPCLC, state.pqrwp / zclfr.clamp(min=1e-10),
        torch.zeros_like(state.pqrwp))
    zqswp = torch.where(
        zclfr > _REPCLC, state.pqswp / zclfr.clamp(min=1e-10),
        torch.zeros_like(state.pqswp))
    return dict(q=zq, qs=zqs, clfr=zclfr, qiwp=zqiwp,
                qlwp=zqlwp, qrwp=zqrwp, qswp=zqswp)


def _build_sw_inputs(state: "RadinaState", clamped: dict,
                     o3_mmr: torch.Tensor, zamu0: torch.Tensor,
                     cloud_optics: bool = True) -> dict:
    """Assemble the per-band optical inputs that sw_solver expects.

    Mirrors the RADLSW dump contract: PCLDSW, PAER, POZ, and the per-band
    cloud optics ZTAU/ZOMEGA/ZCG (combined liquid+ice via Slingo+Fu96 from
    sw_cloud_optical_properties), plus PQS.

    Returns a kwargs dict for ``classic_sw.driver.sw_solver``. All tensors
    are top-down (nlon, ...) matching the sw_solver contract.
    """
    nlev = state.nlev
    nlon = state.nlon
    dt = state.pt.dtype
    dev = state.pt.device
    nsw = 6

    # PCLDSW: SW cloud fraction (top-down). radina's clfr is (nlon, nlev) top-down.
    pcldsw = clamped["clfr"]
    if not cloud_optics:
        pcldsw = torch.zeros_like(pcldsw)

    # POZ: O3 column in cm-atm (radozc converts mmr*Pa to cm-atm via 46.6968/RG).
    # radlsw: ZOZ = POZON * 46.6968 / RG, where POZON = o3_mmr * pdp (kg/kg*Pa).
    RG = 9.80665
    poz = (o3_mmr * state.pdp) * 46.6968 / RG          # (nlon, nlev) cm-atm

    # PAER: Tegen 6-type aerosol. Default to a small background (radaca not yet
    # wired in the torch path); shape (nlon, 6, nlev) top-down.
    aer = torch.full((nlon, 6, nlev), 1e-30, dtype=dt, device=dev)

    # Cloud optics: combined liquid+ice per band.
    if cloud_optics:
        from .radlsw_sw_cloud_optics import sw_cloud_optical_properties
        # sw_cloud_optical_properties expects leading vertical axis (nlev, nlon).
        # Our state is (nlon, nlev); transpose, call, then transpose back.
        sw_opt = sw_cloud_optical_properties(
            q_liquid=clamped["qlwp"].T, q_ice=clamped["qiwp"].T,
            pressure=state.paprsf.T, temperature=state.pt.T, dp=state.pdp.T,
            land_sea_mask=state.pslm, dtype=dt,
        )
        # sw_opt returns (nlev, nlon, nsw); permute to (nlon, nlev, nsw).
        tau_l = sw_opt["tau_l"].permute(1, 0, 2)
        tau_i = sw_opt["tau_i"].permute(1, 0, 2)
        ol = sw_opt["omega_l"].permute(1, 0, 2)
        oi = sw_opt["omega_i"].permute(1, 0, 2)
        gl = sw_opt["g_l"].permute(1, 0, 2)
        gi = sw_opt["g_i"].permute(1, 0, 2)
        # Combine liquid + ice: ZTAU = tau_l + tau_i
        ztau = tau_l + tau_i
        # ZOMEGA = (tau_l*ol + tau_i*oi) / ZTAU  (weighted SSA)
        ztau_safe = ztau.clamp(min=1e-30)
        zomega = (tau_l * ol + tau_i * oi) / ztau_safe
        # ZCG = (tau_l*ol*gl + tau_i*oi*gi) / (tau_l*ol + tau_i*oi)
        scat = (tau_l * ol + tau_i * oi).clamp(min=1e-30)
        zcg = (tau_l * ol * gl + tau_i * oi * gi) / scat
        # Reshape to (nlon, nsw, nlev) for sw_solver (band axis second).
        pcg = zcg.permute(0, 2, 1).contiguous()
        pomega = zomega.permute(0, 2, 1).contiguous()
        ptau = ztau.permute(0, 2, 1).contiguous()
    else:
        pcg = torch.zeros(nlon, nsw, nlev, dtype=dt, device=dev)
        pomega = torch.zeros(nlon, nsw, nlev, dtype=dt, device=dev)
        ptau = torch.zeros(nlon, nsw, nlev, dtype=dt, device=dev)

    # PQS: saturation specific humidity (top-down).
    pqs = clamped["qs"]

    return dict(
        p_half=state.paprs, temp=state.pt, q_h2o=clamped["q"],
        q_co2_vmr=(state.co2_vmr if state.co2_vmr is not None
                   else torch.full((nlon, nlev), _RCCARDI, dtype=dt, device=dev)),
        mu0=zamu0, albd=state.palbd, albp=state.palbp,
        pcldsw=pcldsw, aer=aer, poz=poz,
        pcg=pcg, pomega=pomega, ptau=ptau, pqs=pqs,
    )


def radina(state: RadinaState, novlp: int = 1) -> RadinaOutput:
    """Main radiation interface — from model state to LW + SW fluxes.

    Args:
        state: RadinaState with all model fields (TOA→surface ordering).
        novlp: cloud overlap scheme (1=random, 2=maximum-random, ...).

    Returns:
        RadinaOutput with LW/SW fluxes at half-levels.
    """
    nlev = state.nlev
    nlon = state.nlon
    dt = state.pt.dtype
    dev = state.pt.device

    # ── 1. Half-level temperature ──
    pth = _interp_half_level_temp(state.pt, state.paprsf, state.paprs, state.pts)

    # ── 2. Earth-curvature solar zenith ──
    zamu0 = _earth_curvature_mu0(state.pmu0)

    # ── 3. Safety clamps ──
    clamped = _safety_clamps(state)

    # ── 4. Ozone ──
    if state.o3_mmr is not None:
        o3_mmr = state.o3_mmr  # (nlon, nlev)
    else:
        from .radozc import radozc
        # Convert to (nlon, nlev) layer ozone
        o3_layer = radozc(state.paprs, state.pgemu)        # (nlon, nlev) Pa·kg/kg
        o3_mmr = o3_layer / state.pdp.clamp(min=1.0)       # → kg/kg

    # ── 5. Gas mixing ratios ──
    co2 = state.co2_vmr if state.co2_vmr is not None else torch.full(
        (nlon, nlev), _RCCARDI, dtype=dt, device=dev)
    ch4 = state.ch4_vmr if state.ch4_vmr is not None else torch.full(
        (nlon, nlev), _RCH4, dtype=dt, device=dev)
    n2o = state.n2o_vmr if state.n2o_vmr is not None else torch.full(
        (nlon, nlev), _RN2O, dtype=dt, device=dev)

    # ── 6. LW chain ──
    inputs = RadLSWInputs(
        emis=state.pemir, emiw=state.pemiw,
        albd=state.palbd, albp=state.palbp,
        pts=state.pts,
        cloud_fraction=clamped["clfr"].T,           # (nlev, nlon)
        pap=state.paprsf.T,                          # (nlev, nlon)
        paph=state.paprs.T,                          # (nlev+1, nlon)
        pdp=state.pdp.T,                             # (nlev, nlon)
        pt=state.pt.T,                               # (nlev, nlon)
        pth=pth.T,                                   # (nlev+1, nlon)
        pq=clamped["q"].T,                           # (nlev, nlon)
        co2=co2.T, ch4=ch4.T, n2o=n2o.T,
        no2=torch.zeros(nlev, nlon, dtype=dt, device=dev),
        cfc11=torch.zeros(nlev, nlon, dtype=dt, device=dev),
        cfc12=torch.zeros(nlev, nlon, dtype=dt, device=dev),
        cfc22=torch.zeros(nlev, nlon, dtype=dt, device=dev),
        ccl4=torch.zeros(nlev, nlon, dtype=dt, device=dev),
        o3_mmr=o3_mmr.T,                            # (nlev, nlon)
        aer=torch.full((nlev, nlon, 6), 1e-30, dtype=dt, device=dev),
        tau_lw=torch.zeros(nlev, nlon, 16, dtype=dt, device=dev),
        sw_optics=None,
    )
    lw_out = rrtm_rrtm_140gp(inputs, novlp=novlp)

    # LW fluxes: lw_out returns TOA-first (nlon, nlev+1)
    pemtd = lw_out["flux_down"]                     # total downward LW
    pemtc = lw_out["flux_down_clr"] if "flux_down_clr" in lw_out else pemtd
    pemit = lw_out["emissivity"]
    pfrted = pemtd[:, -1]                           # surface downwelling LW

    # ── 7. SW chain (full callpar path: sw_solver with cloud optics) ──
    sw_kwargs = _build_sw_inputs(state, clamped, o3_mmr, zamu0,
                                 cloud_optics=True)
    sw_out = sw_solver(**sw_kwargs)
    ptrso = sw_out["fd_total"]                       # total downward SW (nlon, nlev+1)
    # Clear-sky SW from the per-band clear fluxes (fc_band) summed.
    ptrsoc = sw_out["fc_band"].sum(dim=0)            # (nlon, nlev+1)
    ptrsod = ptrso[:, -1] / (_RII0 * zamu0.clamp(min=1e-4))
    pdsrp = sw_out["fd_band"].sum(dim=0)[:, -1]      # direct at surface (approx)

    # ── 8. Heating rates (K/day) ──
    # Net flux (downward positive) at half-levels, then divergence into each
    # layer. Matches the bridge (_heating_rate): net = dn - up; div = net[k+1]-
    # net[k]; dT/dt = g/cp * div/dp. dp clamped to avoid TOA-boundary blow-up.
    g = 9.80665
    cp = 1004.7
    sec_per_day = 86400.0
    dp_3d = torch.clamp(state.pdp, min=1.0)          # (nlon, nlev)

    fd_lw, fu_lw = lw_out["flux_down"], lw_out["flux_up"]
    fd_sw, fu_sw = sw_out["fd_total"], sw_out["fu_total"]

    lw_net = fd_lw - fu_lw                            # (nlon, nlev+1), down positive
    sw_net = fd_sw - fu_sw
    # Divergence per layer = F_net[top] - F_net[bottom] (energy into the layer).
    # With net downward-positive, absorption makes F_net decrease with depth,
    # so top - bottom > 0 => heating. (Matches the bridge's div sign after its
    # net convention; here we use the direct top-minus-bottom form.)
    div_lw = lw_net[:, :-1] - lw_net[:, 1:]           # (nlon, nlev)
    div_sw = sw_net[:, :-1] - sw_net[:, 1:]
    heat_lw = g / cp * div_lw / dp_3d * sec_per_day
    heat_sw = g / cp * div_sw / dp_3d * sec_per_day

    return RadinaOutput(
        pemtd=pemtd, pemtc=pemtc,
        ptrso=ptrso, ptrsoc=ptrsoc,
        pemit=pemit, pdsrp=pdsrp,
        pfrted=pfrted, ptrsod=ptrsod,
        pth=pth,
        heat_lw=heat_lw, heat_sw=heat_sw,
    )
