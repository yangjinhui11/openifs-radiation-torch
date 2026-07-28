#!/usr/bin/env python3
"""FD cross-validation for the global gradient at one sample point.

Picks one column from the subsampled grid, computes ∂F_sfc/∂T_sfc,
∂F_sfc/∂q_sfc, ∂F_sfc/∂μ₀ via both autograd and central finite differences,
and saves the results + timing data to /tmp/global_gradients_fd.json.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

RAD_ROOT = Path("/home/qixiang/yangjinhui/openifs/physics_callpar/openifs_radiation_pytorch")
sys.path.insert(0, str(RAD_ROOT))
os.environ.setdefault("DATA", str(RAD_ROOT))

from openifs_radiation.classic_sw.driver import sw_solver

g = np.load("/tmp/global_sw.npz")
nlat = int(g["nlat"]); nlon = int(g["nlon"]); nlev = int(g["nlev"])

# Pick a mid-latitude daytime column for FD validation
LAT_STEP = 8; LON_STEP = 16
lat_idx = np.arange(0, nlat, LAT_STEP)
lon_idx = np.arange(0, nlon, LON_STEP)
lat_grid, lon_grid = np.meshgrid(lat_idx, lon_idx, indexing="ij")
sample_flat = (lat_grid * nlon + lon_grid).flatten()

# Pick a mid-latitude daytime column (~30°N, 0°E) for FD validation
# lat_idx has ~46 entries spanning -90 to +90; ~30°N ≈ index 28
target_lat = 28   # ~30°N
target_lon = 0    # first longitude
col_global = sample_flat[target_lat * len(lon_idx) + target_lon]
print(f"Validation column: global index {col_global}, lat={g['lats'][lat_idx[target_lat]]:.1f}°, "
      f"lon={g['lons'][lon_idx[target_lon]]:.1f}°, mu0={max(g['mu0'][col_global], 0.15):.3f}")

dt = torch.float64; PCARDI = 348.0e-6; MU0_FLOOR = 0.15

def build_inputs(T_np, Q_np, mu0_val, requires_grad=False):
    bs = 1
    T_t = torch.from_numpy(T_np).to(dt).reshape(1, -1)
    Q_t = torch.from_numpy(Q_np).to(dt).reshape(1, -1)
    if requires_grad:
        T_t.requires_grad_(True); Q_t.requires_grad_(True)
    mu0_t = torch.tensor([mu0_val], dtype=dt)
    if requires_grad:
        mu0_t.requires_grad_(True)
    p_h = torch.from_numpy(g["p_half"][col_global]).to(dt).reshape(1, -1)
    co2 = torch.full((1, nlev), PCARDI, dtype=dt)
    o3 = torch.from_numpy(g["O3"][col_global]).to(dt).reshape(1, -1)
    albd = torch.full((1, 6), 0.15, dtype=dt); albp = torch.full((1, 6), 0.15, dtype=dt)
    pcldsw = torch.full((1, nlev), 1e-6, dtype=dt)
    aer = torch.full((1, 6, nlev), 1e-3, dtype=dt)
    poz = o3.clone()
    pcg = torch.full((1, 6, nlev), 0.85, dtype=dt)
    pomega = torch.full((1, 6, nlev), 0.99, dtype=dt)
    ptau = torch.full((1, 6, nlev), 1e-3, dtype=dt)
    pqs = torch.full((1, nlev), 1e-3, dtype=dt)
    return dict(p_half=p_h, temp=T_t, q_h2o=Q_t, q_co2_vmr=co2, mu0=mu0_t,
                albd=albd, albp=albp, pcldsw=pcldsw, aer=aer, poz=poz,
                pcg=pcg, pomega=pomega, ptau=ptau, pqs=pqs)

T_base = g["T"][col_global].astype(np.float64)
Q_base = g["Q"][col_global].astype(np.float64)
mu0_base = max(g["mu0"][col_global], MU0_FLOOR)

# ── Autograd ──
t0 = time.time()
inp = build_inputs(T_base, Q_base, mu0_base, requires_grad=True)
out = sw_solver(**inp)
fwd_time = time.time() - t0
fd_surf = out["fd_total"][0, -1]
t0 = time.time()
fd_surf.backward()
bwd_time = time.time() - t0
ag_T = inp["temp"].grad[0, -1].item()
ag_Q = inp["q_h2o"].grad[0, -1].item()
ag_mu0 = inp["mu0"].grad.item()

print(f"\nAutograd: fwd={fwd_time:.2f}s, bwd={bwd_time:.2f}s")
print(f"  ∂F/∂T_sfc = {ag_T:.6e}")
print(f"  ∂F/∂q_sfc = {ag_Q:.4f}")
print(f"  ∂F/∂μ₀    = {ag_mu0:.4f}")

# ── Finite differences ──
def fd_flux(T_np, Q_np, mu0_val):
    inp = build_inputs(T_np, Q_np, mu0_val)
    with torch.no_grad():
        out = sw_solver(**inp)
    return out["fd_total"][0, -1].item()

eps = 1e-3
# ∂F/∂T_sfc
unit = np.zeros(nlev); unit[-1] = 1.0
fd_T = (fd_flux(T_base + eps*unit, Q_base, mu0_base) - fd_flux(T_base - eps*unit, Q_base, mu0_base)) / (2*eps)
# ∂F/∂q_sfc
fd_Q = (fd_flux(T_base, Q_base + eps*unit, mu0_base) - fd_flux(T_base, Q_base - eps*unit, mu0_base)) / (2*eps)
# ∂F/∂μ₀
fd_mu0 = (fd_flux(T_base, Q_base, mu0_base + eps) - fd_flux(T_base, Q_base, mu0_base - eps)) / (2*eps)

print(f"\nFinite difference (eps={eps}):")
print(f"  ∂F/∂T_sfc = {fd_T:.6e}  rel_err={abs(fd_T-ag_T)/abs(ag_T):.2e}")
print(f"  ∂F/∂q_sfc = {fd_Q:.4f}  rel_err={abs(fd_Q-ag_Q)/abs(ag_Q):.2e}")
print(f"  ∂F/∂μ₀    = {fd_mu0:.4f}  rel_err={abs(fd_mu0-ag_mu0)/abs(ag_mu0):.2e}")

# ── Save ──
result = {
    "col_global": int(col_global),
    "lat": float(g["lats"][lat_idx[target_lat]]),
    "lon": float(g["lons"][lon_idx[target_lon]]),
    "mu0": mu0_base,
    "fwd_time": fwd_time,
    "bwd_time": bwd_time,
    "autograd": {"grad_T": ag_T, "grad_Q": ag_Q, "grad_mu0": ag_mu0},
    "finite_diff": {"grad_T": fd_T, "grad_Q": fd_Q, "grad_mu0": fd_mu0, "eps": eps},
    "rel_err": {
        "T": abs(fd_T - ag_T) / abs(ag_T),
        "Q": abs(fd_Q - ag_Q) / abs(ag_Q),
        "mu0": abs(fd_mu0 - ag_mu0) / abs(ag_mu0),
    },
}
with open("/tmp/global_gradients_fd.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to /tmp/global_gradients_fd.json")
