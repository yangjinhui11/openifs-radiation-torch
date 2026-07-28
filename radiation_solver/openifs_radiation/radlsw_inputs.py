"""RADLSW solver-input assembly -- packs upstream state into the arrays the
LW (RRTM_RRTM_140GP) and SW solvers consume.

After M1a (LW cloud optics) and M1b (SW cloud optics) the remaining RADLSW
inline work is *not* computation -- it is straightforward input packing.
The findings:

  * **Effective cloudiness ZCLDLD/ZCLDLU** (radlsw.F90 L760-877): gated by
    ``IF (.NOT.LRRTM)`` and therefore **dead code in production** (which uses
    LRRTM=.TRUE.). Not ported.
  * **Surface emissivity / albedo** (L366-372): ``ZEMIS=PEMIS``, ``ZEMIW=PEMIW``,
    ``ZALBD=PALBD``, ``ZALBP=PALBP`` -- pure pass-through of upstream inputs.
  * **Cloud fraction for SW** (L745): ``ZCLDSW = PCLFR`` -- pass-through.
  * **Gas VMR assembly** (L1473-1484, RRTM branch): scalar/well-mixed gases
    are constants (RCH4, RN2O, RCFC11/12/22, RCCL4) broadcast to (KLON, KLEV);
    CO2 = PCCO2; ozone is converted from mass-mixing-ratio*dp to mmr via
    ``ZOZN = POZON / PDP``.

This module performs that packing in one batched call so M2 (the LW entry
point) and M4/M5 (the solvers) can consume a single tidy struct.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    VMR_CO2, VMR_CH4, VMR_N2O, VMR_NO2,
    VMR_CFC11, VMR_CFC12, VMR_CFC22, VMR_CCL4,
)


@dataclass
class RadLSWInputs:
    """Everything RADLSW hands to the LW/SW solvers, after its inline setup.

    All arrays have leading vertical axis ``(nlev, nlon)`` except the
    surface fields which are per-column ``(nlon,)`` and the LW band axis
    fields which are ``(nlev, nlon, 16)`` / SW ``(nlev, nlon, 6)``.
    """
    # Surface (per column)
    emis: torch.Tensor          # LW emissivity (diffuse), (nlon,)
    emiw: torch.Tensor          # LW emissivity (window), (nlon,)
    albd: torch.Tensor          # SW albedo diffuse, (nlon, nsw)
    albp: torch.Tensor          # SW albedo direct,  (nlon, nsw)
    pts: torch.Tensor           # surface temperature, (nlon,)
    # Per-layer (nlev, nlon)
    cloud_fraction: torch.Tensor   # PCLFR (also = ZCLDSW)
    pap: torch.Tensor              # full-level pressure (Pa)
    paph: torch.Tensor             # half-level pressure (Pa)
    pdp: torch.Tensor              # layer thickness (Pa)
    pt: torch.Tensor               # temperature (K)
    pth: torch.Tensor              # half-level temperature (K)
    pq: torch.Tensor               # specific humidity (kg/kg)
    # Gas VMRs (nlev, nlon)
    co2: torch.Tensor
    ch4: torch.Tensor
    n2o: torch.Tensor
    no2: torch.Tensor
    cfc11: torch.Tensor
    cfc12: torch.Tensor
    cfc22: torch.Tensor
    ccl4: torch.Tensor
    o3_mmr: torch.Tensor           # ozone mass-mixing-ratio (= POZON/PDP)
    # Aerosol optical depth (nlev, nlon, naer) -- pass-through from RADACA
    aer: torch.Tensor | None
    # Cloud optics (produced by M1a / M1b)
    tau_lw: torch.Tensor           # (nlev, nlon, 16) ZTAUCLD
    sw_optics: dict | None         # dict from sw_cloud_optical_properties()


def assemble_radlsw_inputs(
    # Per-column surface inputs
    surface_emis: torch.Tensor,
    surface_emiw: torch.Tensor,
    surface_albd: torch.Tensor,
    surface_albp: torch.Tensor,
    surface_temperature: torch.Tensor,
    # Per-layer inputs (leading vertical axis)
    cloud_fraction: torch.Tensor,
    pressure_full: torch.Tensor,
    pressure_half: torch.Tensor,
    dp: torch.Tensor,
    temperature: torch.Tensor,
    temperature_half: torch.Tensor,
    specific_humidity: torch.Tensor,
    ozone_mmr_dp: torch.Tensor,    # POZON (mass-mixing-ratio * dp)
    # Cloud optics (already computed by M1a/M1b)
    tau_lw: torch.Tensor,
    # Optional overrides
    co2_column: torch.Tensor | None = None,  # PCCO2 per-column (defaults to VMR_CO2)
    sw_optics: dict | None = None,
    aer: torch.Tensor | None = None,
) -> RadLSWInputs:
    """Pack upstream state into the RadLSWInputs struct consumed by solvers.

    The well-mixed gas VMRs (CH4, N2O, CFCs) are broadcast as constants from
    ``config.py``; CO2 can be overridden per-column via ``co2_column``. Ozone
    is converted from POZON (= mmr*dp) to mmr by dividing by PDP.
    """
    nlev, nlon = pressure_full.shape
    dtype = pressure_full.dtype
    device = pressure_full.device

    def broadcast_gas(vmr: float) -> torch.Tensor:
        return torch.full((nlev, nlon), float(vmr), dtype=dtype, device=device)

    # Ozone: POZON is mass-mixing-ratio times dp; divide to get mmr.
    o3_mmr = ozone_mmr_dp / dp

    # CO2: per-column override or global default.
    if co2_column is None:
        co2 = broadcast_gas(VMR_CO2)
    else:
        co2 = co2_column.unsqueeze(0).expand(nlev, nlon).clone()

    return RadLSWInputs(
        emis=surface_emis, emiw=surface_emiw,
        albd=surface_albd, albp=surface_albp,
        pts=surface_temperature,
        cloud_fraction=cloud_fraction,
        pap=pressure_full, paph=pressure_half, pdp=dp,
        pt=temperature, pth=temperature_half, pq=specific_humidity,
        co2=co2,
        ch4=broadcast_gas(VMR_CH4),
        n2o=broadcast_gas(VMR_N2O),
        no2=broadcast_gas(VMR_NO2),
        cfc11=broadcast_gas(VMR_CFC11),
        cfc12=broadcast_gas(VMR_CFC12),
        cfc22=broadcast_gas(VMR_CFC22),
        ccl4=broadcast_gas(VMR_CCL4),
        o3_mmr=o3_mmr,
        aer=aer,
        tau_lw=tau_lw,
        sw_optics=sw_optics,
    )


__all__ = ["RadLSWInputs", "assemble_radlsw_inputs"]
