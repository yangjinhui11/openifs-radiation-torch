"""SWCLR bit-exact validation: torch port vs offline Fortran reference.

Validates the clear-sky reflectivity/transmissivity adding routine ``SWCLR``
(swclr.F90) against ``libsw_ref.so``. swclr is the shared dependency of sw1s
(UV/visible) and swni (near-IR): it builds the clear-column adding matrix
(PRJ/PRK) and equivalent zenith angles (PRMU0) that the band solvers couple
with the gas-transmission and cloud-adding (swr) terms.

Production config only: NOVLP=1 (maximum-random), ECMWF aerosol branch
(NOVLP<5), KAER=1 (Tegen aerosols active), no dust, NSW=6.

Vertical convention: torch ``swclr`` works in Fortran's bottom-up convention
internally and returns Fortran-indexed arrays (PREFZ index 1 = surface). The
Fortran reference returns the same layout, so comparison is direct.
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


def _build_inputs(nlon=3, nlev=12, seed=3):
    """Build swclr inputs (top-down torch tensors).

    swclr needs: PAER (aerosol), PALBP (albedo), PDSIG (sigma thickness),
    PRAYL (Rayleigh coef), PSEC (secant), PRMU (cos sza).
    """
    rng = np.random.default_rng(seed)
    dt = torch.float64

    # Aerosol optical thickness per type per layer (small, physically ~0.01-0.5).
    paer = rng.uniform(0.0, 0.3, size=(nlon, 6, nlev)).astype(np.float64)
    # Surface albedo per band (0.05-0.4), full NSW=6 array.
    palbp = rng.uniform(0.05, 0.4, size=(nlon, 6)).astype(np.float64)
    # Sigma-layer thickness (positive, ~0.01-0.1).
    pdsig = rng.uniform(0.005, 0.08, size=(nlon, nlev)).astype(np.float64)
    # Rayleigh coefficient (~0.05-0.3 depending on band/pressure).
    prayl = rng.uniform(0.05, 0.3, size=(nlon,)).astype(np.float64)
    # Secant (1/mu0): mu0 in [0.2, 0.95] => psec in [1.05, 5.0].
    mu0 = rng.uniform(0.2, 0.95, size=(nlon,)).astype(np.float64)
    psec = 1.0 / mu0

    return {
        "nlon": nlon, "nlev": nlev,
        "paer": torch.from_numpy(paer),       # top-down
        "palbp": torch.from_numpy(palbp),
        "pdsig": torch.from_numpy(pdsig),     # top-down
        "prayl": torch.from_numpy(prayl),
        "psec": torch.from_numpy(psec),
        "prmu": torch.from_numpy(mu0),
    }


@pytest.fixture(scope="module")
def sw_ref():
    r = SwRef()
    assert r.init() == 0
    return r


def _torch_swclr(knu, prof, rray):
    from openifs_radiation.classic_sw.swclr import swclr as torch_swclr
    return torch_swclr(
        knu=knu, kaer=1,
        paer_td=prof["paer"], palbp=prof["palbp"],
        pdsig_td=prof["pdsig"], prayl=prof["prayl"],
        psec=prof["psec"], rray=rray, prmu=prof["prmu"],
    )


def _fortran_swclr(sw_ref, knu, prof):
    """Run Fortran SWCLR. Inputs must be bottom-up for the fields swclr reads
    bottom-up (PDSIG), top-down for PAER (swclr flips internally). Our torch
    swclr port already handles the flips, but for the Fortran call we pass the
    fields in Fortran's expected layout."""
    nlon, nlev = prof["nlon"], prof["nlev"]
    # PAER: Fortran expects top-down (comment "ENTERED FROM TOP TO BOTTOM").
    paer_f = prof["paer"].numpy()                       # already top-down
    palbp_f = prof["palbp"].numpy()
    # PDSIG: Fortran reads PDSIG(JL,JK) bottom-up (from swu). Flip top-down->bu.
    pdsig_f = np.asfortranarray(prof["pdsig"].numpy()[:, ::-1])
    prayl_f = prof["prayl"].numpy()
    psec_f = prof["psec"].numpy()
    out = sw_ref.swclr(knu, 1, paer_f, palbp_f, pdsig_f, prayl_f, psec_f)
    return out


def _rray_table():
    """Load RRAY (6,6) Rayleigh coefficients from the saved npy."""
    tdir = _PKG_ROOT / "openifs_radiation" / "rrtm_lw" / "tables"
    return torch.from_numpy(np.load(str(tdir / "sw_rray.npy"))).to(torch.float64)


# Per-output tolerance. swclr has exp() and divides; the aerosol-normalise
# and delta-Eddington transforms are sensitive but should match to ~1e-12
# (torch.exp vs libm exp differ by ~1 ULP).
_TOL = 1e-11


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("knu", [1, 2, 3, 4, 5, 6])
def test_swclr_pcgaz_ppizaz_ptauz(sw_ref, knu):
    """Layer optical props (g/SSA/tau after delta-transform) must match."""
    prof = _build_inputs(nlon=3, nlev=12)
    rray = _rray_table()
    out_t = _torch_swclr(knu, prof, rray)
    out_f = _fortran_swclr(sw_ref, knu, prof)

    for key in ("pcgaz", "ppizaz", "ptauz"):
        t = out_t[key].numpy()
        f = out_f[key]
        d = np.abs(t - f)
        md = float(d.max())
        print(f"  band{knu} {key}: max|Δ|={md:.3e}")
        assert md < _TOL, f"band{knu} {key} max|Δ|={md:.3e}"


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("knu", [1, 2, 3, 4, 5, 6])
def test_swclr_pray_ptra_prefz(sw_ref, knu):
    """Adding layer R/T (PRAY1/2, PTRA1/2, PREFZ) must match."""
    prof = _build_inputs(nlon=3, nlev=12)
    rray = _rray_table()
    out_t = _torch_swclr(knu, prof, rray)
    out_f = _fortran_swclr(sw_ref, knu, prof)

    for key in ("pray1", "pray2", "ptra1", "ptra2"):
        t = out_t[key].numpy()
        f = out_f[key]
        d = np.abs(t - f)
        md = float(d.max())
        print(f"  band{knu} {key}: max|Δ|={md:.3e}")
        assert md < _TOL, f"band{knu} {key} max|Δ|={md:.3e}"
    # PREFZ is (nlon, 2, klev+1)
    d_prefz = np.abs(out_t["prefz"].numpy() - out_f["prefz"])
    print(f"  band{knu} prefz: max|Δ|={d_prefz.max():.3e}")
    assert d_prefz.max() < _TOL


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("knu", [1, 2, 3, 4, 5, 6])
def test_swclr_prj_prk(sw_ref, knu):
    """Adding matrix PRJ/PRK must match."""
    prof = _build_inputs(nlon=3, nlev=12)
    rray = _rray_table()
    out_t = _torch_swclr(knu, prof, rray)
    out_f = _fortran_swclr(sw_ref, knu, prof)

    for key in ("prj", "prk"):
        t = out_t[key].numpy()
        f = out_f[key]
        d = np.abs(t - f)
        md = float(d.max())
        print(f"  band{knu} {key}: max|Δ|={md:.3e}")
        assert md < _TOL, f"band{knu} {key} max|Δ|={md:.3e}"


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_swclr_prmu0_ptrclr(sw_ref):
    """Equivalent zenith angle (PRMU0) and clear-column transmissivity."""
    prof = _build_inputs(nlon=3, nlev=12)
    rray = _rray_table()
    for knu in (1, 4):
        out_t = _torch_swclr(knu, prof, rray)
        out_f = _fortran_swclr(sw_ref, knu, prof)
        d_mu0 = np.abs(out_t["prmu0"].numpy() - out_f["prmu0"])
        d_clr = np.abs(out_t["ptrclr"].numpy() - out_f["ptrclr"])
        print(f"  band{knu} prmu0 max|Δ|={d_mu0.max():.3e}, ptrclr max|Δ|={d_clr.max():.3e}")
        assert d_mu0.max() < _TOL
        assert d_clr.max() < _TOL


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
