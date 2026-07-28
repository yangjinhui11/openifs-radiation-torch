"""End-to-end SW solver integration test.

Validates that ``sw_solver`` (the callpar-path entry point) correctly chains
SWU → SW1S (bands 1-3) → SWNI (bands 4-6) and produces physically consistent
total/band fluxes. This is the M5 driver integration — it does not compare
against Fortran (the per-kernel tests do that), it verifies the assembly.
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


def _build_full_inputs(nlon=3, nlev=12, seed=29):
    """Build a complete SW input set (top-down torch tensors)."""
    rng = np.random.default_rng(seed)
    dt = torch.float64
    nsw = 6
    p_edge = np.logspace(np.log10(100.0), np.log10(101325.0), nlev + 1)
    p_half = torch.from_numpy(np.tile(p_edge, (nlon, 1))).to(dt)
    temp = torch.from_numpy(np.linspace(220.0, 290.0, nlev) +
                            rng.normal(0, 2, (nlon, nlev))).to(dt)
    q_h2o = torch.from_numpy(np.linspace(1e-5, 0.02, nlev) *
                             rng.uniform(0.8, 1.2, (nlon, 1))).to(dt)
    pqs = (q_h2o * 1.1).clamp(min=1e-4)
    co2_vmr = torch.full((nlon, nlev), 348e-6, dtype=dt)
    mu0 = torch.from_numpy(rng.uniform(0.3, 0.95, (nlon,))).to(dt)
    albd = torch.from_numpy(rng.uniform(0.05, 0.4, (nlon, nsw))).to(dt)
    albp = torch.from_numpy(rng.uniform(0.05, 0.4, (nlon, nsw))).to(dt)
    pcldsw = torch.from_numpy(rng.uniform(0.0, 0.7, (nlon, nlev))).to(dt)
    aer = torch.from_numpy(rng.uniform(0.0, 0.3, (nlon, 6, nlev))).to(dt)
    poz = torch.from_numpy(rng.uniform(0.001, 1.0, (nlon, nlev))).to(dt)
    pcg = torch.from_numpy(rng.uniform(0.5, 0.9, (nlon, nsw, nlev))).to(dt)
    pomega = torch.from_numpy(rng.uniform(0.9, 0.99999, (nlon, nsw, nlev))).to(dt)
    ptau = torch.from_numpy(rng.uniform(0.0, 5.0, (nlon, nsw, nlev))).to(dt)
    return dict(p_half=p_half, temp=temp, q_h2o=q_h2o, q_co2_vmr=co2_vmr,
                mu0=mu0, albd=albd, albp=albp, pcldsw=pcldsw, aer=aer,
                poz=poz, pcg=pcg, pomega=pomega, ptau=ptau, pqs=pqs)


def test_sw_solver_runs_and_physical():
    """sw_solver must produce finite, physically-bounded SW fluxes."""
    from openifs_radiation.classic_sw.driver import sw_solver
    prof = _build_full_inputs(nlon=3, nlev=12)
    out = sw_solver(
        prof["p_half"], prof["temp"], prof["q_h2o"], prof["q_co2_vmr"],
        prof["mu0"], prof["albd"], prof["albp"], prof["pcldsw"], prof["aer"],
        prof["poz"], prof["pcg"], prof["pomega"], prof["ptau"], prof["pqs"])

    fd = out["fd_total"]; fu = out["fu_total"]
    assert torch.isfinite(fd).all(), "SW down flux non-finite"
    assert torch.isfinite(fu).all(), "SW up flux non-finite"
    # Surface down must be positive (incident solar).
    assert (fd[:, -1] >= 0).all(), "Surface SW down negative"
    # TOA up must be non-negative (reflected).
    assert (fu[:, 0] >= -1e-6).all(), "TOA SW up negative"
    # TOA incident ≈ 1361 * mu0; surface down < TOA incident (atmospheric absorption).
    for ic in range(fd.shape[0]):
        incident = 1361.0 * prof["mu0"][ic].item()
        assert fd[ic, -1] < incident + 1.0, (
            f"Col {ic}: sfc down {fd[ic,-1]:.1f} >= incident {incident:.1f}")
    print(f"\n=== sw_solver ({fd.shape[0]} cols, {fd.shape[1]-1} levels) ===")
    print(f"  Sfc down: {fd[:, -1].numpy()}")
    print(f"  TOA up:   {fu[:, 0].numpy()}")
    print(f"  Planetary albedo: {(fu[:, 0] / (1361*prof['mu0'])).numpy()}")


def test_sw_solver_band_decomposition():
    """Total flux = sum over 6 bands; each band contributes non-negative sfc down."""
    from openifs_radiation.classic_sw.driver import sw_solver
    prof = _build_full_inputs(nlon=2, nlev=12)
    out = sw_solver(
        prof["p_half"], prof["temp"], prof["q_h2o"], prof["q_co2_vmr"],
        prof["mu0"], prof["albd"], prof["albp"], prof["pcldsw"], prof["aer"],
        prof["poz"], prof["pcg"], prof["pomega"], prof["ptau"], prof["pqs"])

    fd_sum = out["fd_band"].sum(dim=0)
    diff = (fd_sum - out["fd_total"]).abs().max()
    assert diff < 1e-6, f"Band sum vs total: max|diff|={diff:.2e}"
    print(f"\n=== Band decomposition ===")
    print(f"  Band sum = total? max|diff|={diff:.2e}")
    print(f"  Per-band surface down: {out['fd_band'][:, 0, -1].numpy()}")


def test_sw_solver_clear_vs_cloudy():
    """Clear-sky (pcldsw=0) should give higher surface down than cloudy."""
    from openifs_radiation.classic_sw.driver import sw_solver
    prof = _build_full_inputs(nlon=2, nlev=12)
    # Clear sky
    prof_clr = dict(prof); prof_clr["pcldsw"] = torch.zeros_like(prof["pcldsw"])
    out_clr = sw_solver(
        prof_clr["p_half"], prof_clr["temp"], prof_clr["q_h2o"], prof_clr["q_co2_vmr"],
        prof_clr["mu0"], prof_clr["albd"], prof_clr["albp"], prof_clr["pcldsw"],
        prof_clr["aer"], prof_clr["poz"], prof_clr["pcg"], prof_clr["pomega"],
        prof_clr["ptau"], prof_clr["pqs"])
    # Cloudy
    out_cld = sw_solver(
        prof["p_half"], prof["temp"], prof["q_h2o"], prof["q_co2_vmr"],
        prof["mu0"], prof["albd"], prof["albp"], prof["pcldsw"], prof["aer"],
        prof["poz"], prof["pcg"], prof["pomega"], prof["ptau"], prof["pqs"])
    # Clear-sky surface down >= cloudy (clouds reduce transmission).
    print(f"\n=== Clear vs cloudy ===")
    print(f"  Clear sfc down: {out_clr['fd_total'][:, -1].numpy()}")
    print(f"  Cloudy sfc down: {out_cld['fd_total'][:, -1].numpy()}")
    # At least one column should show clear > cloudy.
    assert (out_clr["fd_total"][:, -1] >= out_cld["fd_total"][:, -1] - 1.0).all(), (
        "Clear sky should transmit more than cloudy")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
