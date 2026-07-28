"""SW cloud optical properties -- production Slingo (1989) liquid + Fu (1996) ice.

Torch port of the shortwave cloud-optics computation in
``openifs-48r1/ifs-source/arpifs/phys_radi/radlsw.F90`` for the production
configuration ``NSWLIQOPT=2`` (Slingo) and ``NSWICEOPT=3`` (Fu 1996), with
``NSW=6`` spectral bands.

For each cloud cell and each SW band the source computes three optical
properties (radlsw.F90 lines 656-728):
  * ``ZTOL`` / ``ZTOI``  -- optical thickness (liquid / ice)
  * ``ZOL``  / ``ZOI``   -- co-single-scatter albedo (1 - omega)
  * ``ZGL``  / ``ZGI``   -- asymmetry parameter

These are then combined in ``radlsw.F90`` (after this kernel, in a section we
do not port here -- it feeds the SW solver) into per-layer band totals.

Reference arithmetic:
  * Slingo liquid (lines 684-688):
        ZTOL = ZFLWP * (RASWCA + RASWCB / re_liq)
        ZGL  = RASWCE + RASWCF * re_liq
        ZOL  = 1 - RASWCC - RASWCD * re_liq
  * Fu-96 ice (lines 720-731):
        Z1RADI = 1 / de_ice
        ZBETAI = RFUAA0 + RFUAA1 * Z1RADI
        ZTOI = ZFIWP * ZBETAI
        ZOMGI = RFUBB0 + de*(RFUBB1 + de*(RFUBB2 + de*RFUBB3))
        ZOI = 1 - ZOMGI
        ZGI = RFUCC0 + de*(RFUCC1 + de*(RFUCC2 + de*RFUCC3))
        ZGI = min(ZGI, 1)

All inputs share the leading vertical axis ``(..., nlev)``; outputs add a
band axis ``(..., nlev, NSW)``.
"""
from __future__ import annotations

import torch

from . import cloud_optics_tables as ct
from .radlsw_cloud_lw_optics import (
    ice_effective_diameter_sun_riku,
    liquid_effective_radius_martin,
)

# NSW = 6 spectral bands (production).
NSW: int = 6

# Cloud-water floor (YOETHF%REPSCW); must match the LW optics module.
REPSCW: float = 1.0e-12


def sw_cloud_optical_properties(
    q_liquid: torch.Tensor, q_ice: torch.Tensor,
    pressure: torch.Tensor, temperature: torch.Tensor, dp: torch.Tensor,
    land_sea_mask: torch.Tensor, dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Per-band SW cloud optical properties for production Slingo + Fu-96.

    Inputs are batched with leading vertical axis ``(..., nlev)`` (except
    land_sea_mask which is per-column ``(...,)``). Returns a dict of tensors
    each with shape ``(..., nlev, NSW)``::

        tau_l, tau_i  -- liquid / ice optical thickness per band
        omega_l, omega_i  -- single-scatter albedo (liquid / ice)
        g_l, g_i  -- asymmetry parameter (liquid / ice)
        cloudy  -- boolean gate (ZFLWP + ZFIWP > REPSCW), shape (..., nlev, 1)

    Cells that are not cloudy are zeroed for tau/omega/g (the source leaves
    them as 0.0 initial values).
    """
    RG = 9.80665
    # ---- in-cloud water paths (g/m^2) ----
    zflwp = (q_liquid * 1000.0) * (dp / RG)
    zfiwp = (q_ice * 1000.0) * (dp / RG)
    # ---- effective sizes (shared with LW optics) ----
    re_liq = liquid_effective_radius_martin(
        q_liquid, pressure, temperature, dp, land_sea_mask, dtype=dtype)
    de_ice, _re_ice = ice_effective_diameter_sun_riku(
        q_ice, pressure, temperature, dtype=dtype)
    # ---- cloudy gate ----
    cloudy = (zflwp + zfiwp) > REPSCW
    # ---- broadcast over band axis ----
    def cell_to_band(t):
        return t.unsqueeze(-1)  # (..., nlev, 1)
    zflwp_b = cell_to_band(zflwp)
    zfiwp_b = cell_to_band(zfiwp)
    re_liq_b = cell_to_band(re_liq)
    de_ice_b = cell_to_band(de_ice)

    # ---- Slingo (1989) liquid, NSWLIQOPT=2 (radlsw.F90 685-687) ----
    # Tables are (NSW,) -> view as (1, ..., 1, NSW) with the right number of
    # leading 1s to broadcast against the cell tensors. zflwp_b already has
    # the band axis appended (..., nlev, 1), so its leading dims are the cell
    # dims -- we need exactly cell.ndim - 1 leading 1s before the NSW axis.
    n_cell_dims = zflwp_b.ndim - 1   # subtract the appended band axis
    _dev = zflwp_b.device
    def band_view(t):
        return t.to(dtype).to(_dev).view(*([1] * n_cell_dims), NSW)
    raswca = band_view(ct.RASWCA)
    raswcb = band_view(ct.RASWCB)
    raswcc = band_view(ct.RASWCC)
    raswcd = band_view(ct.RASWCD)
    raswce = band_view(ct.RASWCE)
    raswcf = band_view(ct.RASWCF)
    ztol = zflwp_b * (raswca + raswcb / re_liq_b)
    zgl = raswce + raswcf * re_liq_b
    zol = 1.0 - raswcc - raswcd * re_liq_b
    # single-scatter albedo omega = 1 - ZOL
    omega_l = 1.0 - zol

    # ---- Fu (1996) ice, NSWICEOPT=3 (radlsw.F90 720-731) ----
    rfaa0 = band_view(ct.RFUAA0)
    rfaa1 = band_view(ct.RFUAA1)
    rfbb0 = band_view(ct.RFUBB0)
    rfbb1 = band_view(ct.RFUBB1)
    rfbb2 = band_view(ct.RFUBB2)
    rfbb3 = band_view(ct.RFUBB3)
    rfcc0 = band_view(ct.RFUCC0)
    rfcc1 = band_view(ct.RFUCC1)
    rfcc2 = band_view(ct.RFUCC2)
    rfcc3 = band_view(ct.RFUCC3)
    z1radi = 1.0 / de_ice_b
    zbetai = rfaa0 + rfaa1 * z1radi
    ztoi = zfiwp_b * zbetai
    zomgi = (rfbb0 + de_ice_b * (rfbb1 + de_ice_b * (rfbb2 + de_ice_b * rfbb3)))
    zoi = 1.0 - zomgi
    omega_i = zomgi
    zgi = (rfcc0 + de_ice_b * (rfcc1 + de_ice_b * (rfcc2 + de_ice_b * rfcc3)))
    zgi = torch.clamp(zgi, max=1.0)

    # ---- zero out clear cells ----
    cloudy_b = cell_to_band(cloudy)
    zero = torch.zeros_like(ztol)
    return {
        "tau_l":   torch.where(cloudy_b, ztol,   zero),
        "tau_i":   torch.where(cloudy_b, ztoi,   zero),
        "omega_l": torch.where(cloudy_b, omega_l, zero),
        "omega_i": torch.where(cloudy_b, omega_i, zero),
        "g_l":     torch.where(cloudy_b, zgl,    zero),
        "g_i":     torch.where(cloudy_b, zgi,    zero),
        "cloudy":  cloudy_b,
    }


__all__ = ["sw_cloud_optical_properties", "NSW"]
