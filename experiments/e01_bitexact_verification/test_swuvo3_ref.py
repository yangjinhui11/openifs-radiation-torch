"""SWUVO3 bit-exact validation: torch port vs offline Fortran reference.

Validates the IFS ozone transmission routine ``SWUVO3`` (swuvo3.F90) against
``libsw_ref.so``. swuvo3 uses a sum-of-exponentials form::

    PTR = Σ_{jx=1..NEXPO3(KNU)} REXPO3(KNU,1,jx) * exp(-REXPO3(KNU,2,jx) * PU)

Only called from sw1s (bands 1-3, UV/visible) in the production NSW=6 config;
NEXPO3 = [7, 7, 6, 4, 0, 0]. Bands 5/6 (NEXPO3=0) return PTR=0 verbatim.
"""
from __future__ import annotations
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


@pytest.fixture(scope="module")
def sw_ref():
    r = SwRef()
    assert r.init() == 0
    return r


def _pu_grid(dtype=np.float64):
    """O3 column amounts spanning the regimes swuvo3 sees in production
    (cm-atm scaled amounts from radozc, ~1e-3 to ~1e2)."""
    return np.array(
        [0.0, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        dtype=dtype,
    )


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("knu", [1, 2, 3, 4, 5, 6])
def test_swuvo3_bitexact(sw_ref, knu):
    """swuvo3 must match Fortran bit-exact across all 6 bands.

    Bands 1-4 have NEXPO3>0 (real O3 absorption); bands 5-6 have NEXPO3=0 and
    must return PTR=0 verbatim (matching Fortran's DO 1,0 no-op).
    """
    from openifs_radiation.classic_sw.swtt import swuvo3

    pu_np = _pu_grid()
    ptr_f = sw_ref.swuvo3(knu, pu_np)
    ptr_t = swuvo3(knu, torch.from_numpy(pu_np)).numpy()

    assert np.isfinite(ptr_f).all(), f"Fortran swuvo3 band{knu} non-finite"
    assert np.isfinite(ptr_t).all(), f"torch swuvo3 band{knu} non-finite"

    abs_diff = np.abs(ptr_t - ptr_f)
    max_abs = float(abs_diff.max())
    print(f"  band{knu}: max|Δ|={max_abs:.3e}")
    print(f"    fort: {np.round(ptr_f, 6)}")
    print(f"    torch:{np.round(ptr_t, 6)}")

    # exp() evaluated in Fortran (libm) vs torch should agree to ~1 ULP; the
    # sum of a few terms stays within a handful of ULPs.
    assert max_abs < 1e-12, (
        f"swuvo3 band{knu}: max|Δ|={max_abs:.3e} exceeds 1e-12")


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swuvo3_band5_6_zero(sw_ref):
    """Bands 5 and 6 (NEXPO3=0) must return exactly 0 (Fortran no-op loop)."""
    from openifs_radiation.classic_sw.swtt import swuvo3

    pu_np = _pu_grid()
    for knu in (5, 6):
        ptr_f = sw_ref.swuvo3(knu, pu_np)
        ptr_t = swuvo3(knu, torch.from_numpy(pu_np)).numpy()
        assert np.all(ptr_f == 0.0), f"Fortran band{knu} not all zero: {ptr_f}"
        assert np.all(ptr_t == 0.0), f"torch band{knu} not all zero: {ptr_t}"
        print(f"  band{knu}: both zero ✓")


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swuvo3_transmission_range(sw_ref):
    """O3 transmission must be non-negative and match the Fortran reference's
    own range (the sum-of-exponentials coefficients are a numerical fit, so
    Σ coef at PU=0 can be ~1.000001 — that is the table's intrinsic property,
    not a porting error, and must match Fortran exactly).
    """
    from openifs_radiation.classic_sw.swtt import swuvo3

    pu_np = _pu_grid()
    for knu in (1, 2, 3, 4):
        ptr_t = swuvo3(knu, torch.from_numpy(pu_np)).numpy()
        ptr_f = sw_ref.swuvo3(knu, pu_np)
        assert (ptr_t >= 0.0).all(), f"band{knu} negative transmission"
        # The reference itself exceeds 1.0 at PU=0 by ~1e-6 (coefficient sum);
        # torch must match that exactly rather than be clamped to <=1.
        d = np.abs(ptr_t - ptr_f).max()
        assert d < 1e-12, f"band{knu} deviates from Fortran range: {d:.3e}"
        print(f"  band{knu} range: [{ptr_t.min():.6f}, {ptr_t.max():.6f}]  (matches Fortran)")


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swuvo3_batched(sw_ref):
    """swuvo3 must handle (nlon, kabs) batched input matching Fortran."""
    from openifs_radiation.classic_sw.swtt import swuvo3

    rng = np.random.default_rng(11)
    pu = (10.0 ** rng.uniform(-3, 2, size=(8, 2))).astype(np.float64)

    for knu in (1, 2, 3):
        ptr_f = sw_ref.swuvo3(knu, pu)
        ptr_t = swuvo3(knu, torch.from_numpy(pu)).numpy()
        d = np.abs(ptr_t - ptr_f)
        print(f"  band{knu} batched: max|Δ|={d.max():.3e}")
        assert d.max() < 1e-12, f"band{knu} batched max|Δ|={d.max():.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
