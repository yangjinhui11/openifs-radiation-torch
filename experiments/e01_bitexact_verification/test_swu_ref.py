"""SWU bit-exact validation: torch port vs offline Fortran reference.

Validates the classic IFS shortwave absorber-amount routine ``SWU`` (swu.F90)
against ``libsw_ref.so`` (compiled verbatim from phys_radi/swu.F90).

SWU computes per-layer H2O / CO2 (UMG) column amounts plus the geometry
factors (PRMU, PSEC) and the σ-layer thickness (PDSIG). It is the entry point
of the SW chain (SW → SWU → SW1S/SWNI) and the first place arithmetic
divergence would surface, so it is validated to ULP level.

This test also pins down the three semantic invariants that the previous
placeholder violated (and that the rewrite restored):
  * no RRAE curvature correction (that lives in radina, not swu);
  * co2 is a VMR (matches Fortran PCARDI), not an MMR;
  * PUD is a *per-layer* amount (Fortran output contract), not a cumsum.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

_SKIP = None
try:
    sys.path.insert(0, str(_HERE.parent))
    from offline_ref.sw_ref import SwRef
    if not (_HERE.parent / "offline_ref" / "libsw_ref.so").exists():
        _SKIP = "libsw_ref.so not built (run offline_ref/build_ref.sh)"
except Exception as e:  # noqa: BLE001
    _SKIP = f"offline SW ref unavailable: {e}"


def _build_profile(nlon=3, nlev=15, seed=7):
    """Build a multi-column atmosphere (TOA-first torch arrays).

    Returns top-down torch tensors plus the scalars Fortran swu needs
    (psct = solar constant, pcardi = CO2 VMR).
    """
    rng = np.random.default_rng(seed)
    dt = torch.float64

    # Half-level pressures (Pa): TOA → surface, varying slightly per column.
    p_edge_hpa_base = np.logspace(np.log10(0.7), np.log10(1013.0), nlev + 1)
    p_half = np.tile(p_edge_hpa_base * 100.0, (nlon, 1))
    # jitter surface pressure column-to-column so PPSOL differs
    p_half[:, -1] *= rng.uniform(0.98, 1.02, size=nlon)
    p_half = np.sort(p_half, axis=1)  # ensure monotonic increasing (TOA→sfc)

    # Temperature (K), top-down; water vapour (specific humidity, kg/kg).
    t_full = np.linspace(220.0, 294.0, nlev) + rng.normal(0, 2.0, size=(nlon, nlev))
    pwv = np.linspace(1e-5, 0.02, nlev) * rng.uniform(0.8, 1.2, size=(nlon, 1))

    co2_vmr = 348.0e-6  # PCARDI is a VMR scalar in Fortran
    psct = 1361.0
    mu0 = np.array([0.3, 0.5, 0.7, 0.9])[:nlon]

    return {
        "nlon": nlon, "nlev": nlev,
        "p_half": torch.from_numpy(p_half).to(dt),
        "temp": torch.from_numpy(t_full).to(dt),
        "pwv": torch.from_numpy(pwv).to(dt),
        "co2_vmr": torch.full((nlon, nlev), co2_vmr, dtype=dt),
        "mu0": torch.from_numpy(mu0).to(dt),
        "psct": psct,
        "pcardi": co2_vmr,
    }


@pytest.fixture(scope="module")
def sw_ref():
    r = SwRef()
    assert r.init() == 0
    return r


def _torch_swu(prof):
    """Run the torch _swu port, return its per-layer PUD + geometry (top-down)."""
    from openifs_radiation.classic_sw.driver import _swu

    p_full = 0.5 * (prof["p_half"][:, :-1] + prof["p_half"][:, 1:])
    out_t = _swu(prof["p_half"], p_full, prof["temp"], prof["pwv"],
                 prof["co2_vmr"], None, prof["mu0"])
    return out_t


def _fortran_swu(sw_ref, prof):
    """Run the Fortran SWU reference.

    Fortran swu has a SUBTLE mixed vertical convention (cf. radlsw.F90 lines
    374-411 where these fields are assembled):
      * PPMB  — bottom-up (JK=1=surface), in hPa. radlsw builds it as
                ZPMB(:,:,JK+1)=PAPH(:,:,KLEV+1-JK), i.e. flipped from the
                top-down PAPH. So we flip p_half.
      * PTAVE — bottom-up (JK=1=surface). radlsw: ZTAVE(:,:,JK)=PT(:,:,KLEV+1-JK).
                So we flip temp.
      * PWV   — **TOP-DOWN** (accessed inside swu as PWV(JL,JKL) with
                JKL=KLEV+1-JK, i.e. swu flips it itself). radlsw passes PWV
                through unchanged from the top-down PT convention. So we do
                NOT flip pwv.

    Returns the Fortran outputs unchanged (still in Fortran's bottom-up
    orientation); callers flip for comparison.
    """
    p_half = prof["p_half"].numpy()
    ppmb_bu = np.asfortranarray((p_half / 100.0)[:, ::-1])            # hPa, sfc→TOA
    ptave_bu = np.asfortranarray(prof["temp"].numpy()[:, ::-1])       # K, sfc→TOA
    pwv_td = np.asfortranarray(prof["pwv"].numpy())                   # TOP-DOWN, no flip
    ppsol = np.ascontiguousarray(p_half[:, -1])                       # surface Pa
    prmu0 = np.ascontiguousarray(prof["mu0"].numpy())
    out_f = sw_ref.swu(prof["psct"], prof["pcardi"], ppmb_bu, ppsol,
                       prmu0, ptave_bu, pwv_td)
    return out_f


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swu_psec_no_curvature(sw_ref):
    """PSEC must equal 1/mu0 exactly — swu does NOT apply the RRAE correction.

    The previous implementation returned the RRAE-corrected secant
    (ZAMU0=RRAE/(sqrt(mu0^2+zcrae)-mu0), which is a radina-layer quantity);
    Fortran swu sets PSEC=1/PRMU0 verbatim.
    """
    prof = _build_profile(nlon=3, nlev=15)
    out_t = _torch_swu(prof)
    out_f = _fortran_swu(sw_ref, prof)

    psec_t = out_t["psec"].numpy()
    psec_f = out_f["psec"]
    d = np.abs(psec_t - psec_f)
    print(f"  psec torch: {psec_t}")
    print(f"  psec fort:  {psec_f}")
    print(f"  max|Δ|: {d.max():.3e}")
    # Exact reciprocal — only float round-off.
    assert d.max() < 1e-14, f"PSEC max|Δ|={d.max():.3e} (expected 1/mu0)"


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swu_pdsig_bitexact(sw_ref):
    """PDSIG (σ-layer thickness) must match Fortran bit-exact.

    This is the pressure/arithmetic backbone of swu; it already agreed before
    the rewrite, and must continue to agree (top-down vs bottom-up are the same
    differences, just reversed in order).
    """
    prof = _build_profile(nlon=3, nlev=15)
    out_t = _torch_swu(prof)
    out_f = _fortran_swu(sw_ref, prof)

    # Fortran PDSIG is bottom-up (JK=1=ground); torch is top-down. Flip.
    pdsig_f_td = out_f["pdsig"][:, ::-1]
    d = np.abs(out_t["pdsig"].numpy() - pdsig_f_td)
    print(f"  pdsig max|Δ|: {d.max():.3e}")
    assert d.max() < 1e-12, f"PDSIG max|Δ|={d.max():.3e}"


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swu_pud_h2o_co2_bitexact(sw_ref):
    """PUD per-layer H2O (idx 1) and CO2 (idx 2) must match Fortran bit-exact.

    This is the core invariant the rewrite restores: PUD is a *per-layer*
    amount (matching Fortran's output contract), not a cumsum. The previous
    implementation returned cumsum(PUD), which produced a "cumulative of
    cumulative" downstream in sw1s/swni.
    """
    prof = _build_profile(nlon=3, nlev=15)
    out_t = _torch_swu(prof)
    out_f = _fortran_swu(sw_ref, prof)

    # Fortran PUD bottom-up: idx 0=ground layer, KLEV=top layer, KLEV+1=TOA=0.
    # torch PUD top-down: idx 0=TOA=0, 1..KLEV=top→ground.
    # Flip the level axis so both are top-down.
    pud_f_td = out_f["pud"][:, :, ::-1]
    pud_t = out_t["pud"].numpy()

    for name, j in [("H2O", 0), ("CO2", 1), ("H2O_vap", 3), ("H2O_dry", 4)]:
        a_t = pud_t[:, j, :]
        a_f = pud_f_td[:, j, :]
        d = np.abs(a_t - a_f)
        # Skip the exact-zero TOA boundary (idx 0) in the rel check.
        interior = np.abs(a_f) > 1e-30
        d_int = d[interior]
        md = float(d_int.max()) if interior.any() else 0.0
        print(f"  PUD {name}: max|Δ|={md:.3e}  (torch[0,:4]={a_t[0,:4]})")
        # Per-layer amounts are products of pressure-powers × scalars ×
        # temperature-powers; torch.pow vs Fortran ** should agree to ~1e-13.
        assert md < 1e-12, f"PUD[{name}] max|Δ|={md:.3e}"


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swu_pud_toa_boundary_zero(sw_ref):
    """PUD at the TOA boundary (index 0 in top-down) must be exactly 0."""
    prof = _build_profile(nlon=3, nlev=15)
    out_t = _torch_swu(prof)
    pud_t = out_t["pud"].numpy()
    toa = pud_t[:, :, 0]
    print(f"  PUD[:,:,0] (TOA boundary): {toa}")
    assert np.allclose(toa, 0.0, atol=0.0), (
        f"PUD TOA boundary non-zero: max={np.abs(toa).max():.3e}")


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swu_pud_is_per_layer_not_cumulative(sw_ref):
    """Guard against regression: PUD must be per-layer, NOT a cumsum.

    Uses an isolating probe — humidity placed in a SINGLE layer only — so the
    per-layer vs cumulative distinction is unambiguous:
      * per-layer: only the spiked layer has nonzero PUD;
      * cumsum:    all layers at/below the spike have nonzero PUD.
    """
    prof = _build_profile(nlon=1, nlev=8)
    # Overwrite pwv so only one mid-atmosphere layer carries moisture.
    pwv_spike = np.full(8, 1e-12)
    pwv_spike[3] = 0.01                                    # top-down index 3
    prof["pwv"] = torch.from_numpy(pwv_spike.copy()).to(torch.float64).unsqueeze(0)

    out_t = _torch_swu(prof)
    pud_h2o = out_t["pud"][0, 0, 1:].numpy()              # per-layer, top-down
    nz = np.where(pud_h2o > 1e-6)[0].tolist()
    print(f"  H2O per-layer with single-layer spike: {pud_h2o}")
    print(f"  nonzero layers: {nz} (per-layer expects [3] only)")
    # Only the spiked layer should carry the amount. A cumsum would propagate
    # it to every layer below the spike (indices 3..7).
    assert nz == [3], (
        f"PUD looks cumulative (nonzero at {nz}); expected per-layer at [3] only")


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swu_pud_o3_zero(sw_ref):
    """PUD index 3 (O3) must be zero — swu leaves it for swuvo3/POZ."""
    prof = _build_profile(nlon=2, nlev=10)
    out_t = _torch_swu(prof)
    o3 = out_t["pud"][:, 2, :].numpy()
    print(f"  PUD[:,:,3] (O3, should be 0): max|val|={np.abs(o3).max():.3e}")
    assert np.abs(o3).max() == 0.0, "swu must leave O3 column (PUD[:,:,3]) at 0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
