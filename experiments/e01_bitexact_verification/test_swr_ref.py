"""SWR bit-exact validation: torch port vs offline Fortran reference.

Validates the cloudy-sky reflectivity/transmissivity adding routine ``SWR``
(swr.F90) against ``libsw_ref.so``. swr couples the clear-sky layer optics
(from swclr) with the cloud optics (PCG/POMEGA/PTAU/PCLD) via delta-Eddington
(swde) and produces the cloudy-column adding matrix (PRJ/PRK/PRMUE) that
sw1s/swni combine with the gas-transmission terms.

Production config: NOVLP=1, ECMWF branch, NSW=6.
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
    from offline_ref.sw_ref import SwRef
    if not (_HERE.parent / "offline_ref" / "libsw_ref.so").exists():
        _SKIP = "libsw_ref.so not built (run offline_ref/build_ref.sh)"
except Exception as e:  # noqa: BLE001
    _SKIP = f"offline SW ref unavailable: {e}"


def _build_inputs(nlon=3, nlev=12, seed=9):
    """Build swr inputs (bottom-up torch tensors where Fortran is bottom-up).

    swr needs the clear-sky layer props (PCGAZ/PPIZAZ/PTAUAZ from swclr) plus
    the cloud fields (PCG/POMEGA/PTAU/PCLD).
    """
    rng = np.random.default_rng(seed)
    dt = torch.float64
    palbp = torch.from_numpy(rng.uniform(0.05, 0.4, (nlon, 6)).astype(np.float64))
    # Cloud optical props per band (small positive values).
    pcg    = torch.from_numpy(rng.uniform(0.5, 0.9, (nlon, 6, nlev)).astype(np.float64))
    pomega = torch.from_numpy(rng.uniform(0.9, 0.99999, (nlon, 6, nlev)).astype(np.float64))
    ptau   = torch.from_numpy(rng.uniform(0.0, 5.0, (nlon, 6, nlev)).astype(np.float64))
    # Cloud fraction (0..0.8).
    pcld   = torch.from_numpy(rng.uniform(0.0, 0.8, (nlon, nlev)).astype(np.float64))
    # Clear-sky layer props (from swclr stage 1: bottom-up, g/SSA/tau).
    pcgaz  = torch.from_numpy(rng.uniform(0.5, 0.9, (nlon, nlev)).astype(np.float64))
    ppizaz = torch.from_numpy(rng.uniform(0.9, 0.99999, (nlon, nlev)).astype(np.float64))
    ptauz  = torch.from_numpy(rng.uniform(0.01, 1.0, (nlon, nlev)).astype(np.float64))
    mu0 = torch.from_numpy(rng.uniform(0.2, 0.95, (nlon,)).astype(np.float64))
    psec = 1.0 / mu0
    return dict(nlon=nlon, nlev=nlev, palbp=palbp, pcg=pcg, pomega=pomega,
                ptau=ptau, pcld=pcld, pcgaz=pcgaz, ppizaz=ppizaz, ptauz=ptauz,
                psec=psec)


@pytest.fixture(scope="module")
def sw_ref():
    r = SwRef()
    assert r.init() == 0
    return r


_TOL = 1e-10   # swr has exp() with 200-clamp + swde; allow a few ULPs.


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("knu", [1, 2, 3, 4, 5, 6])
def test_swr_outputs(sw_ref, knu):
    """All swr outputs (pray1/2, prefz, prj, prk, prmue, ptra1/2, ptrcld) must
    match the Fortran reference bit-exact across all 6 bands."""
    from openifs_radiation.classic_sw.swr import swr as torch_swr

    prof = _build_inputs(nlon=3, nlev=12)
    # torch swr: cloud fields indexed by Fortran IKL/JKM1 internally; we pass
    # the bottom-up arrays directly (pcld/pcgaz/ppizaz/ptauz are bottom-up).
    out_t = torch_swr(knu, prof["palbp"], prof["pcg"], prof["pcld"],
                      prof["pomega"], prof["psec"], prof["ptau"],
                      prof["pcgaz"], prof["ppizaz"], prof["ptauz"])

    # Fortran: pass bottom-up arrays as-is (Fortran indexes them itself).
    out_f = sw_ref.swr(knu, prof["palbp"].numpy(), prof["pcg"].numpy(),
                       prof["pcld"].numpy(), prof["pomega"].numpy(),
                       prof["psec"].numpy(), prof["ptau"].numpy(),
                       prof["pcgaz"].numpy(), prof["ppizaz"].numpy(),
                       prof["ptauz"].numpy())

    worst = 0.0
    for key in ("pray1", "pray2", "prefz", "prj", "prk", "prmue",
                "ptra1", "ptra2", "ptrcld"):
        t = out_t[key].numpy()
        f = out_f[key]
        d = np.abs(t - f)
        md = float(d.max())
        worst = max(worst, md)
        if md > _TOL:
            print(f"  band{knu} {key}: max|Δ|={md:.3e}  FAIL")
    print(f"  band{knu} worst max|Δ| = {worst:.3e}")
    assert worst < _TOL, f"band{knu} swr worst max|Δ|={worst:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
