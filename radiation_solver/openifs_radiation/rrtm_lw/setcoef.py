"""Batched torch implementation of ``RRTM_SETCOEF_140GP``.

The Fortran reference computes, for each atmospheric layer, the pressure/
temperature interpolation indices and fractions used by the RRTMG-LW gas-
optics routines, together with the column amounts and reference mixing-ratio
ratios required by the 16 TAUMOL bands. This module replicates that arithmetic
in fully-vectorised ``torch.float64`` so that the torch port can be validated
bit-exact against the production OpenIFS code.

Vertical layout follows the OpenIFS convention: the leading axis is the
vertical axis ``(nlev, nlon)``. ``wkl`` has shape ``(nlev, nlon, JPINPX)`` with
Fortran gas index 1 (H2O) at Python index 0.
"""
from __future__ import annotations

import torch

from .tables import PREFLOG, TREF, CHI_MLS

JPINPX: int = 35  # max active trace gases, matches PARRRTM


def _trunc_int(x: torch.Tensor) -> torch.Tensor:
    """Fortran ``INT`` truncation toward zero, returning int64."""
    return x.to(torch.int64)


def rrtm_setcoef_140gp(
    pavel: torch.Tensor,
    tavel: torch.Tensor,
    coldry: torch.Tensor,
    wbrodl: torch.Tensor,
    wkl: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute SETCOEF gas-optics interpolation quantities.

    Parameters
    ----------
    pavel, tavel, coldry, wbrodl
        Layer pressure (hPa), layer temperature (K), dry-air column and
        broadened-density column, all of shape ``(nlev, nlon)``.
    wkl
        Trace-gas amounts, shape ``(nlev, nlon, JPINPX)``. Only the first seven
        gas slots are used: 0=H2O, 1=CO2, 2=O3, 3=N2O, 5=CH4, 6=O2.

    Returns
    -------
    dict
        Dictionary with the same field names as the Fortran reference, including
        ``jp``, ``jt``, ``jt1``, ``fac00/fac01/fac10/fac11``, column amounts,
        continuum factors, and binary-species reference ratios. Integer tensors
        are returned as ``torch.int64``; floating tensors are ``torch.float64``.
        The scalar per-column counters ``laytrop``, ``layswtch`` and ``laylow``
        are returned as 1-D tensors of length ``nlon``.
    """
    if pavel.dtype != torch.float64:
        raise TypeError(f"pavel must be float64, got {pavel.dtype}")
    if pavel.dim() != 2:
        raise ValueError(f"pavel must be 2-D (nlev, nlon), got shape {pavel.shape}")
    nlev, nlon = pavel.shape
    if tavel.shape != pavel.shape:
        raise ValueError(f"tavel shape {tavel.shape} != pavel shape {pavel.shape}")
    if coldry.shape != pavel.shape:
        raise ValueError(f"coldry shape {coldry.shape} != pavel shape {pavel.shape}")
    if wbrodl.shape != pavel.shape:
        raise ValueError(f"wbrodl shape {wbrodl.shape} != pavel shape {pavel.shape}")
    if wkl.shape != (nlev, nlon, JPINPX):
        raise ValueError(f"wkl shape {wkl.shape} != (nlev={nlev}, nlon={nlon}, JPINPX={JPINPX})")

    # Move reference tables to the input device/dtype once.
    pref_log = PREFLOG.to(device=pavel.device, dtype=pavel.dtype)
    tref = TREF.to(device=pavel.device, dtype=pavel.dtype)
    chi_mls = CHI_MLS.to(device=pavel.device, dtype=pavel.dtype)

    plog = torch.log(pavel)

    # ------------------------------------------------------------------
    # Pressure interpolation and clamping (Fortran indices are 1-based).
    # ------------------------------------------------------------------
    jp = _trunc_int(36.0 - 5.0 * (plog + 0.04))
    jp = torch.clamp(jp, 1, 58)
    jp1 = jp + 1

    fp = 5.0 * (pref_log[jp] - plog)
    fp = torch.clamp(fp, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Temperature interpolation at the lower and upper reference pressures.
    # ------------------------------------------------------------------
    jt = _trunc_int(3.0 + (tavel - tref[jp]) / 15.0)
    jt = torch.clamp(jt, 1, 4)
    ft = ((tavel - tref[jp]) / 15.0) - jt.to(torch.float64) + 3.0

    jt1 = _trunc_int(3.0 + (tavel - tref[jp1]) / 15.0)
    jt1 = torch.clamp(jt1, 1, 4)
    ft1 = ((tavel - tref[jp1]) / 15.0) - jt1.to(torch.float64) + 3.0

    # ------------------------------------------------------------------
    # Common water-vapor and pressure-scaling factors.
    # ------------------------------------------------------------------
    water = wkl[..., 0] / coldry
    scalefac = pavel * (296.0 / 1013.0) / tavel
    lower_mask = plog > 4.56

    # ------------------------------------------------------------------
    # Column amounts (common to both branches).
    # ------------------------------------------------------------------
    colh2o = 1.0e-20 * wkl[..., 0]
    colco2 = 1.0e-20 * wkl[..., 1]
    colo3 = 1.0e-20 * wkl[..., 2]
    coln2o = 1.0e-20 * wkl[..., 3]
    colch4 = 1.0e-20 * wkl[..., 5]
    colo2 = 1.0e-20 * wkl[..., 6]
    colbrd = 1.0e-20 * wbrodl

    colco2 = torch.where(colco2 == 0.0, 1.0e-32 * coldry, colco2)
    coln2o = torch.where(coln2o == 0.0, 1.0e-32 * coldry, coln2o)
    colch4 = torch.where(colch4 == 0.0, 1.0e-32 * coldry, colch4)

    co2reg = 3.55e-24 * coldry
    co2mult = (colco2 - co2reg) * (
        272.63 * torch.exp(-1919.4 / tavel) / (8.7604e-4 * tavel)
    )

    # ------------------------------------------------------------------
    # Continuum and minor-gas factors (branch-specific formulas).
    # ------------------------------------------------------------------
    # Foreign continuum: defined in both branches with different indices.
    forfac = scalefac / (1.0 + water)
    factor_for_lower = (332.0 - tavel) / 36.0
    indfor_lower = torch.clamp(_trunc_int(factor_for_lower), 1, 2)
    forfrac_lower = factor_for_lower - indfor_lower.to(torch.float64)

    factor_for_upper = (tavel - 188.0) / 36.0
    indfor_upper = torch.full_like(indfor_lower, 3, dtype=torch.int64)
    forfrac_upper = factor_for_upper - 1.0

    indfor = torch.where(lower_mask, indfor_lower, indfor_upper)
    forfrac = torch.where(lower_mask, forfrac_lower, forfrac_upper)

    # Self-continuum: only defined in the lower atmosphere.
    selffac = water * forfac
    factor_self = (tavel - 188.0) / 7.2
    int_self = _trunc_int(factor_self)
    indself_lower = torch.clamp(int_self - 7, 1, 9)
    selffrac_lower = factor_self - (indself_lower + 7).to(torch.float64)
    indself = torch.where(lower_mask, indself_lower, torch.zeros_like(indself_lower))
    selffrac = torch.where(lower_mask, selffrac_lower, torch.zeros_like(selffrac_lower))

    # Minor-gas scaling: same expression in both branches.
    scaleminor = pavel / tavel
    scaleminorn2 = (pavel / tavel) * (wbrodl / (coldry + wkl[..., 0]))
    factor_minor = (tavel - 180.8) / 7.2
    indminor = torch.clamp(_trunc_int(factor_minor), 1, 18)
    minorfrac = factor_minor - indminor.to(torch.float64)

    # ------------------------------------------------------------------
    # Reference binary-species ratios (branch-specific which ones are used).
    # ------------------------------------------------------------------
    # Lower-atmosphere ratios.
    rat_h2oco2 = chi_mls[1, jp] / chi_mls[2, jp]
    rat_h2oco2_1 = chi_mls[1, jp1] / chi_mls[2, jp1]
    rat_h2oo3 = torch.where(lower_mask, chi_mls[1, jp] / chi_mls[3, jp], torch.zeros_like(pavel))
    rat_h2oo3_1 = torch.where(lower_mask, chi_mls[1, jp1] / chi_mls[3, jp1], torch.zeros_like(pavel))
    rat_h2on2o = torch.where(lower_mask, chi_mls[1, jp] / chi_mls[4, jp], torch.zeros_like(pavel))
    rat_h2on2o_1 = torch.where(lower_mask, chi_mls[1, jp1] / chi_mls[4, jp1], torch.zeros_like(pavel))
    rat_h2och4 = torch.where(lower_mask, chi_mls[1, jp] / chi_mls[6, jp], torch.zeros_like(pavel))
    rat_h2och4_1 = torch.where(lower_mask, chi_mls[1, jp1] / chi_mls[6, jp1], torch.zeros_like(pavel))
    rat_n2oco2 = torch.where(lower_mask, chi_mls[4, jp] / chi_mls[2, jp], torch.zeros_like(pavel))
    rat_n2oco2_1 = torch.where(lower_mask, chi_mls[4, jp1] / chi_mls[2, jp1], torch.zeros_like(pavel))

    # Upper-atmosphere ratios.
    rat_o3co2 = torch.where(~lower_mask, chi_mls[3, jp] / chi_mls[2, jp], torch.zeros_like(pavel))
    rat_o3co2_1 = torch.where(~lower_mask, chi_mls[3, jp1] / chi_mls[2, jp1], torch.zeros_like(pavel))

    # ------------------------------------------------------------------
    # Interpolation fractions for the four (p,T) corner points.
    # ------------------------------------------------------------------
    compfp = 1.0 - fp
    fac10 = compfp * ft
    fac00 = compfp * (1.0 - ft)
    fac11 = fp * ft1
    fac01 = fp * (1.0 - ft1)

    # Rescale self/foreign continuum factors by the H2O column amount.
    selffac = colh2o * selffac
    forfac = colh2o * forfac

    # Per-column tropopause bookkeeping.
    laytrop = lower_mask.sum(dim=0, dtype=torch.int64)
    layswtch = torch.zeros(nlon, dtype=torch.int64, device=pavel.device)
    laylow = torch.ones(nlon, dtype=torch.int64, device=pavel.device)

    return {
        "colbrd": colbrd,
        "fac00": fac00,
        "fac01": fac01,
        "fac10": fac10,
        "fac11": fac11,
        "forfac": forfac,
        "forfrac": forfrac,
        "indfor": indfor,
        "jp": jp,
        "jt": jt,
        "jt1": jt1,
        "colh2o": colh2o,
        "colco2": colco2,
        "colo3": colo3,
        "coln2o": coln2o,
        "colch4": colch4,
        "colo2": colo2,
        "co2mult": co2mult,
        "laytrop": laytrop,
        "layswtch": layswtch,
        "laylow": laylow,
        "selffac": selffac,
        "selffrac": selffrac,
        "indself": indself,
        "indminor": indminor,
        "scaleminor": scaleminor,
        "scaleminorn2": scaleminorn2,
        "minorfrac": minorfrac,
        "rat_h2oco2": rat_h2oco2,
        "rat_h2oco2_1": rat_h2oco2_1,
        "rat_h2oo3": rat_h2oo3,
        "rat_h2oo3_1": rat_h2oo3_1,
        "rat_h2on2o": rat_h2on2o,
        "rat_h2on2o_1": rat_h2on2o_1,
        "rat_h2och4": rat_h2och4,
        "rat_h2och4_1": rat_h2och4_1,
        "rat_n2oco2": rat_n2oco2,
        "rat_n2oco2_1": rat_n2oco2_1,
        "rat_o3co2": rat_o3co2,
        "rat_o3co2_1": rat_o3co2_1,
    }
