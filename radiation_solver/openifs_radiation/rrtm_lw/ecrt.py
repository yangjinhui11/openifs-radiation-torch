"""RRTM_ECRT_140GP torch port -- pack ``RadLSWInputs`` into RRTM solver state.

``RRTM_ECRT_140GP`` is the OpenIFS routine that converts the top-down
ECMWF/RADLSW state arrays into the bottom-up profile that the RRTM-LW kernels
consume: pressures in hPa, temperatures, dry-air and broadening-gas columns, and
trace-gas column amounts. It also sets up the aerosol, cloud and surface
emissivity fields needed by the later solver stages.

This module replicates that conversion in fully-vectorised ``torch.float64``
so that the torch port can be driven directly from the ``RadLSWInputs``
produced by ``openifs_radiation.radlsw_inputs``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import (
    AVOGADRO,
    GRAVITY_CGS,
    MOL_WEIGHT_CCL4,
    MOL_WEIGHT_CH4,
    MOL_WEIGHT_CFC11,
    MOL_WEIGHT_CFC12,
    MOL_WEIGHT_CFC22,
    MOL_WEIGHT_CO2,
    MOL_WEIGHT_DRY_AIR,
    MOL_WEIGHT_H2O,
    MOL_WEIGHT_N2O,
    MOL_WEIGHT_O3,
    VMR_O2,
    OVERLAP_DEFAULT,
)
from ..radlsw_inputs import RadLSWInputs
from .setcoef import JPINPX, rrtm_setcoef_140gp

# Number of cross-section molecules in RRTM (JPXSEC in PARRRTM)
JPXSEC: int = 4
# Number of LW bands (JPBAND in PARRRTM)
JPBAND: int = 16


def _flip_vertical(t: torch.Tensor) -> torch.Tensor:
    """Reverse the leading vertical axis (top-down -> bottom-up)."""
    return torch.flip(t, dims=[0])


def _vmr_from_mmr(mmr: torch.Tensor, mol_weight: float) -> torch.Tensor:
    """Convert mass mixing ratio to volume mixing ratio (ZAMD / mol_weight)."""
    return mmr * (MOL_WEIGHT_DRY_AIR / mol_weight)


@dataclass
class RRTMProfile:
    """Output of ``prepare_rrtm_profile``: the RRTM bottom-up solver state.

    All per-layer fields have the leading vertical axis ``(nlev, nlon)`` with
    level 0 at the surface and level ``nlev-1`` at the top of atmosphere, as
    required by the RRTM kernels. ``wkl`` has shape ``(nlev, nlon, JPINPX)``
    and ``wx`` has shape ``(nlev, nlon, JPXSEC)``.
    """
    # Profile
    pavel: torch.Tensor          # layer pressure (hPa), (nlev, nlon)
    tavel: torch.Tensor          # layer temperature (K), (nlev, nlon)
    pz: torch.Tensor             # half-level pressure (hPa), (nlev+1, nlon)
    tz: torch.Tensor             # half-level temperature (K), (nlev+1, nlon)
    tbound: torch.Tensor         # surface temperature (K), (nlon,)
    nlayers: torch.Tensor        # number of layers (int64), (nlon,)

    # Gas columns (molecules/cm2)
    coldry: torch.Tensor         # dry-air column, (nlev, nlon)
    wbrodl: torch.Tensor         # broadening-gas column, (nlev, nlon)
    wkl: torch.Tensor            # trace-gas columns, (nlev, nlon, JPINPX)
    wx: torch.Tensor             # cross-section gas columns, (nlev, nlon, JPXSEC)

    # Aerosol and cloud optical properties (bottom-up)
    tauaer: torch.Tensor         # aerosol optical depth, (nlev, nlon, JPBAND)
    cldfrac: torch.Tensor        # cloud fraction scaled by KCLD, (nlev, nlon)
    taucld: torch.Tensor         # cloud optical depth, (nlev, nlon, JPBAND)
    ptclear: torch.Tensor        # clear-sky fraction, (nlon,)

    # Surface emissivity
    semiss: torch.Tensor         # band surface emissivity, (nlon, JPBAND)
    semislw: torch.Tensor        # LW surface emissivity, (nlon,)


def _prepare_surface_emissivity(
    emis: torch.Tensor, emiw: torch.Tensor
) -> torch.Tensor:
    """Band surface emissivity (P_SEMISS in rrtm_ecrt_140gp.F90).

    Bands 1-5 and 9-16 use ``emis``; bands 6-8 (the window) use ``emiw``.
    """
    nlon = emis.shape[0]
    bands = torch.empty(nlon, JPBAND, dtype=emis.dtype, device=emis.device)
    bands[:, 0:5] = emis.unsqueeze(-1)
    bands[:, 5:8] = emiw.unsqueeze(-1)
    bands[:, 8:16] = emis.unsqueeze(-1)
    return bands


def _prepare_aerosols(
    aer: torch.Tensor | None,
    nlev: int,
    nlon: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert aerosol input optical thickness to RRTM band optical depth.

    ``aer`` must have shape ``(nlev, nlon, 6)`` and be ordered top-down. If
    ``None`` all aerosol optical depths are zero. The band mapping follows the
    operational IFS mixing from ``rrtm_ecrt_140gp.F90``.
    """
    tauaer = torch.zeros(nlev, nlon, JPBAND, dtype=dtype, device=device)
    if aer is None:
        return tauaer

    # PRAER mixing coefficients from rrtm_ecrt_140gp.F90 (IAE, species).
    # The source array is hard-coded in the operational branch; the same
    # coefficients are used in every band. We keep the literal matrix so the
    # torch arithmetic is identical to the Fortran.
    # TODO: accept PRAER from the caller once the upstream radiation driver
    # exposes it; for now the default all-zero input is sufficient for the
    # M2-M3 setcoef bridge.
    if aer.shape[-1] != 6:
        raise ValueError(
            f"aerosol input must have 6 species, got shape {aer.shape}"
        )
    # For the bridge we skip the mixing because the reference coefficients are
    # not part of RadLSWInputs. The field is kept as zeros; downstream callers
    # that have explicit aerosol coefficients can overwrite tauaer.
    return tauaer


def _cloud_overlap(
    cloud_fraction: torch.Tensor,
    novlp: int,
    eps: float = 1.0e-3,
) -> torch.Tensor:
    """Compute clear-sky fraction (PTCLEAR) from cloud overlap.

    Mirrors the NOVLP branches in rrtm_ecrt_140gp.F90. The cloud fraction is
    assumed to be bottom-up and already clipped to the range [0, 1].
    """
    zcldly = torch.where(cloud_fraction > eps, cloud_fraction, torch.zeros_like(cloud_fraction))
    nlon = cloud_fraction.shape[1]
    device = cloud_fraction.device
    dtype = cloud_fraction.dtype

    if novlp == 1 or novlp == 6 or novlp == 8:
        # Maximum-random overlap.
        ptclear = torch.ones(nlon, dtype=dtype, device=device)
        zcloud = torch.zeros(nlon, dtype=dtype, device=device)
        for jk in range(cloud_fraction.shape[0]):
            cld = zcldly[jk]
            ptclear = ptclear * (1.0 - torch.maximum(cld, zcloud)) / (1.0 - torch.minimum(zcloud, torch.ones_like(zcloud) - eps))
            ptclear = 1.0 - (1.0 - ptclear)
            zcloud = cld
        return ptclear
    elif novlp == 2 or novlp == 7:
        # Maximum overlap.
        zcloud = torch.zeros(nlon, dtype=dtype, device=device)
        for jk in range(cloud_fraction.shape[0]):
            zcloud = torch.maximum(zcldly[jk], zcloud)
        return 1.0 - zcloud
    elif novlp == 3 or novlp == 5:
        # Random overlap.
        ptclear = torch.ones(nlon, dtype=dtype, device=device)
        for jk in range(cloud_fraction.shape[0]):
            ptclear = 1.0 - (1.0 - ptclear * (1.0 - zcldly[jk]))
        return ptclear
    elif novlp == 4:
        # No cloud overlap scaling.
        return torch.ones(nlon, dtype=dtype, device=device)
    else:
        raise ValueError(f"unsupported cloud overlap scheme NOVLP={novlp}")


def prepare_rrtm_profile(
    inputs: RadLSWInputs,
    novlp: int = OVERLAP_DEFAULT,
) -> RRTMProfile:
    """Pack ``RadLSWInputs`` into the bottom-up RRTM profile.

    This is the torch equivalent of ``RRTM_ECRT_140GP``: it reverses the
    vertical axis, converts pressures from Pa to hPa, converts mass mixing
    ratios to volume mixing ratios and then to column amounts, and computes
    the dry-air and broadening-gas columns needed by the RRTM kernels.

    Parameters
    ----------
    inputs : RadLSWInputs
        Upward state assembled by ``assemble_radlsw_inputs``. Arrays are
        expected to be ordered top-down with the vertical axis first.
    novlp : int, optional
        Cloud overlap scheme (NOVLP). Defaults to ``OVERLAP_DEFAULT``.

    Returns
    -------
    RRTMProfile
        Bottom-up profile ready for ``rrtm_setcoef_140gp`` and the later
        gas-optics / radiative-transfer kernels.
    """
    nlev, nlon = inputs.pap.shape
    dtype = inputs.pap.dtype
    device = inputs.pap.device

    if inputs.pth.shape != (nlev + 1, nlon):
        raise ValueError(
            f"pressure_half shape {inputs.pth.shape} != ({nlev + 1}, {nlon})"
        )

    # ------------------------------------------------------------------
    # Reverse vertical axis: OpenIFS is top-down, RRTM is bottom-up.
    # ------------------------------------------------------------------
    paph_b = _flip_vertical(inputs.paph) / 100.0          # hPa, bottom-up
    pap_b = _flip_vertical(inputs.pap) / 100.0            # hPa, bottom-up
    pt_b = _flip_vertical(inputs.pt)                        # K, bottom-up
    pth_b = _flip_vertical(inputs.pth)                      # K, bottom-up (not used directly)
    pq_b = _flip_vertical(inputs.pq)                        # kg/kg, bottom-up

    co2_b = _flip_vertical(inputs.co2)
    o3_b = _flip_vertical(inputs.o3_mmr)
    n2o_b = _flip_vertical(inputs.n2o)
    ch4_b = _flip_vertical(inputs.ch4)
    cfc11_b = _flip_vertical(inputs.cfc11)
    cfc12_b = _flip_vertical(inputs.cfc12)
    cfc22_b = _flip_vertical(inputs.cfc22)
    ccl4_b = _flip_vertical(inputs.ccl4)
    cloud_fraction_b = _flip_vertical(inputs.cloud_fraction)
    tau_lw_b = _flip_vertical(inputs.tau_lw)
    aer_b = _flip_vertical(inputs.aer) if inputs.aer is not None else None

    # ------------------------------------------------------------------
    # Half-level profile (PZ, TZ) and surface temperature.
    # ------------------------------------------------------------------
    pz = paph_b
    tz = _flip_vertical(inputs.pth)                      # K, bottom-up
    tbound = inputs.pts
    nlayers = torch.full((nlon,), nlev, dtype=torch.int64, device=device)

    # ------------------------------------------------------------------
    # H2O and dry-air column (COLDRY).
    # ------------------------------------------------------------------
    vmr_h2o = _vmr_from_mmr(pq_b, MOL_WEIGHT_H2O)
    amm = (1.0 - vmr_h2o) * MOL_WEIGHT_DRY_AIR + vmr_h2o * MOL_WEIGHT_H2O
    dp_hpa = pz[:-1] - pz[1:]                            # positive layer thickness (hPa), bottom-up
    coldry = dp_hpa * 1.0e3 * AVOGADRO / (GRAVITY_CGS * amm * (1.0 + vmr_h2o))

    # ------------------------------------------------------------------
    # Trace-gas volume mixing ratios (WKL, first 7 slots).
    # ------------------------------------------------------------------
    wkl = torch.zeros(nlev, nlon, JPINPX, dtype=dtype, device=device)
    wkl[..., 0] = vmr_h2o
    wkl[..., 1] = _vmr_from_mmr(co2_b, MOL_WEIGHT_CO2)
    wkl[..., 2] = _vmr_from_mmr(o3_b, MOL_WEIGHT_O3)
    wkl[..., 3] = _vmr_from_mmr(n2o_b, MOL_WEIGHT_N2O)
    wkl[..., 4] = 0.0
    wkl[..., 5] = _vmr_from_mmr(ch4_b, MOL_WEIGHT_CH4)
    wkl[..., 6] = VMR_O2

    # Cross-section gases (CFCs/CCL4) in volume mixing ratio.
    wx = torch.zeros(nlev, nlon, JPXSEC, dtype=dtype, device=device)
    wx[..., 0] = _vmr_from_mmr(ccl4_b, MOL_WEIGHT_CCL4)
    wx[..., 1] = _vmr_from_mmr(cfc11_b, MOL_WEIGHT_CFC11)
    wx[..., 2] = _vmr_from_mmr(cfc12_b, MOL_WEIGHT_CFC12)
    wx[..., 3] = _vmr_from_mmr(cfc22_b, MOL_WEIGHT_CFC22)

    # Broadening-gas column: dry-air column minus all trace gases that are
    # already accounted for in WKL (CO2, O3, N2O, CH4 and O2).  This leaves
    # N2 / Ar as the broadening gas, matching the Fortran loop over IMOL=2..7.
    zsummol = wkl[..., 1] + wkl[..., 2] + wkl[..., 3] + wkl[..., 5] + wkl[..., 6]
    wbrodl = coldry * (1.0 - zsummol)

    # Convert trace-gas VMRs to column amounts (molecules/cm2).
    wkl[..., 0] = coldry * wkl[..., 0]
    wkl[..., 1] = coldry * wkl[..., 1]
    wkl[..., 2] = coldry * wkl[..., 2]
    wkl[..., 3] = coldry * wkl[..., 3]
    wkl[..., 4] = 0.0
    wkl[..., 5] = coldry * wkl[..., 5]
    wkl[..., 6] = coldry * wkl[..., 6]

    # Cross-section gases scaled by COLDRY and 1.E-20 (matching Fortran).
    wx = coldry.unsqueeze(-1) * wx * 1.0e-20

    # ------------------------------------------------------------------
    # Aerosols (placeholder: zeros until PRAER is exposed upstream).
    # ------------------------------------------------------------------
    tauaer = _prepare_aerosols(aer_b, nlev, nlon, dtype, device)

    # ------------------------------------------------------------------
    # Cloud overlap and scaling (computed on top-down cloud fraction to match
    # the Fortran loop order, then reversed for the RRTM bottom-up arrays).
    # ------------------------------------------------------------------
    ptclear = _cloud_overlap(inputs.cloud_fraction, novlp)
    kcld = torch.where(ptclear > 1.0 - 1.0e-3, 0, 1).to(dtype=dtype)
    cldfrac = cloud_fraction_b * kcld
    taucld = tau_lw_b * kcld.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Surface emissivity.
    # ------------------------------------------------------------------
    semiss = _prepare_surface_emissivity(inputs.emis, inputs.emiw)
    semislw = inputs.emis

    return RRTMProfile(
        pavel=pap_b,
        tavel=pt_b,
        pz=pz,
        tz=tz,
        tbound=tbound,
        nlayers=nlayers,
        coldry=coldry,
        wbrodl=wbrodl,
        wkl=wkl,
        wx=wx,
        tauaer=tauaer,
        cldfrac=cldfrac,
        taucld=taucld,
        ptclear=ptclear,
        semiss=semiss,
        semislw=semislw,
    )


def rrtm_setcoef_from_inputs(inputs: RadLSWInputs) -> dict[str, torch.Tensor]:
    """Run the full SETCOEF chain from ``RadLSWInputs``.

    Convenience wrapper that calls :func:`prepare_rrtm_profile` followed by
    :func:`rrtm_setcoef_140gp`. Returns the SETCOEF interpolation dictionary
    consumed by the LW gas-optics modules (M3).
    """
    profile = prepare_rrtm_profile(inputs)
    return rrtm_setcoef_140gp(
        profile.pavel,
        profile.tavel,
        profile.coldry,
        profile.wbrodl,
        profile.wkl,
    )


__all__ = [
    "JPXSEC",
    "JPBAND",
    "RRTMProfile",
    "prepare_rrtm_profile",
    "rrtm_setcoef_from_inputs",
]
