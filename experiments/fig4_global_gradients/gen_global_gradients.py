#!/usr/bin/env python3
"""Experiment 3: Global 2D radiation sensitivity fields via autograd.

Computes ∂F_sfc/∂T_sfc, ∂F_sfc/∂q_sfc, ∂F_sfc/∂μ₀ on a subsampled ERA5 grid,
then reshapes to 2D global maps. Uses the autograd-compatible SW chain
(all JIT kernels disabled, pure-Python fallback).

Strategy: subsample the 259,920-column global grid to ~2000 columns (every
~130th column ≈ 8° spacing) for tractable runtime, then reshape to 2D.

Output: /tmp/global_gradients.npz
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import torch

# ── Setup ────────────────────────────────────────────────────────────────
RAD_ROOT = Path("/home/qixiang/yangjinhui/openifs/physics_callpar/openifs_radiation_pytorch")
sys.path.insert(0, str(RAD_ROOT))
os.environ.setdefault("DATA", str(RAD_ROOT))

from openifs_radiation.classic_sw.driver import sw_solver

# ── Load global ERA5 data ────────────────────────────────────────────────
print("Loading global ERA5 data...")
g = np.load("/tmp/global_sw.npz")
nlat = int(g["nlat"])  # 361
nlon = int(g["nlon"])  # 720
nlev = int(g["nlev"])  # 137
nlev1 = nlev + 1
ncol_total = nlat * nlon  # 259920
lats = g["lats"]
lons = g["lons"]

print(f"Grid: {nlat}×{nlon} = {ncol_total} columns, {nlev} levels")

# ── Subsample for tractable runtime ──────────────────────────────────────
# Sample every Nth column in both lat and lon for a coarser grid
LAT_STEP = 8   # every 8th lat point → ~45 lat points
LON_STEP = 16  # every 16th lon point → ~45 lon points
lat_idx = np.arange(0, nlat, LAT_STEP)
lon_idx = np.arange(0, nlon, LON_STEP)
nlat_s = len(lat_idx)
nlon_s = len(lon_idx)
ncol_s = nlat_s * nlon_s
print(f"Subsampled: {nlat_s}×{nlon_s} = {ncol_s} columns (step {LAT_STEP}×{LON_STEP})")

# Build subsampled 2D index grid
lat_grid, lon_grid = np.meshgrid(lat_idx, lon_idx, indexing="ij")
sample_flat = (lat_grid * nlon + lon_grid).flatten()  # column indices into flat array

# Extract subsampled data
T_all = g["T"][sample_flat]          # (ncol_s, nlev)
Q_all = g["Q"][sample_flat]
O3_all = g["O3"][sample_flat]
p_half_all = g["p_half"][sample_flat]  # (ncol_s, nlev1)
mu0_all = g["mu0"][sample_flat]
SP_all = g["SP"][sample_flat]

# Floor mu0 to avoid division issues (polar night)
MU0_FLOOR = 0.15
mu0_all = np.maximum(mu0_all, MU0_FLOOR)

# Device
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEV}")

print(f"T range: {T_all.min():.1f} - {T_all.max():.1f} K")
print(f"Q range: {Q_all.min():.2e} - {Q_all.max():.2e}")
print(f"mu0 range: {mu0_all.min():.3f} - {mu0_all.max():.3f}")

# ── Batch autograd gradient computation ──────────────────────────────────
dt = torch.float64
PCARDI = 348.0e-6

# Process in chunks to manage memory
CHUNK = 200 if DEV == "cuda" else 50  # columns per batch
n_chunks = (ncol_s + CHUNK - 1) // CHUNK

grad_T_sfc = np.zeros(ncol_s)
grad_Q_sfc = np.zeros(ncol_s)
grad_mu0_sfc = np.zeros(ncol_s)
fd_sfc = np.zeros(ncol_s)

print(f"\nProcessing {ncol_s} columns in {n_chunks} chunks of {CHUNK}...")
t0 = time.time()

for chunk_idx in range(n_chunks):
    i0 = chunk_idx * CHUNK
    i1 = min(i0 + CHUNK, ncol_s)
    bs = i1 - i0

    if chunk_idx % 5 == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(chunk_idx, 1) * (n_chunks - chunk_idx)
        print(f"  Chunk {chunk_idx+1}/{n_chunks} (cols {i0}-{i1-1}), "
              f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    T_chunk = torch.from_numpy(T_all[i0:i1]).to(dt).to(DEV)
    Q_chunk = torch.from_numpy(Q_all[i0:i1]).to(dt).to(DEV)
    T_chunk.requires_grad_(True)
    Q_chunk.requires_grad_(True)

    mu0_chunk = torch.from_numpy(mu0_all[i0:i1]).to(dt).to(DEV)
    mu0_chunk.requires_grad_(True)

    p_h = torch.from_numpy(p_half_all[i0:i1]).to(dt).to(DEV)
    co2 = torch.full((bs, nlev), PCARDI, dtype=dt, device=DEV)
    o3 = torch.from_numpy(O3_all[i0:i1]).to(dt).to(DEV)
    albd = torch.full((bs, 6), 0.15, dtype=dt, device=DEV)
    albp = torch.full((bs, 6), 0.15, dtype=dt, device=DEV)
    pcldsw = torch.full((bs, nlev), 1e-6, dtype=dt, device=DEV)
    aer = torch.full((bs, 6, nlev), 1e-3, dtype=dt, device=DEV)
    poz = o3.clone()
    pcg = torch.full((bs, 6, nlev), 0.85, dtype=dt, device=DEV)
    pomega = torch.full((bs, 6, nlev), 0.99, dtype=dt, device=DEV)
    ptau = torch.full((bs, 6, nlev), 1e-3, dtype=dt, device=DEV)
    pqs = torch.full((bs, nlev), 1e-3, dtype=dt, device=DEV)

    try:
        out = sw_solver(
            p_half=p_h, temp=T_chunk, q_h2o=Q_chunk, q_co2_vmr=co2,
            mu0=mu0_chunk, albd=albd, albp=albp, pcldsw=pcldsw,
            aer=aer, poz=poz, pcg=pcg, pomega=pomega, ptau=ptau, pqs=pqs,
        )
        fd_surf = out["fd_total"][:, -1]  # (bs,)
        fd_sfc[i0:i1] = fd_surf.detach().cpu().numpy()

        # Sum flux for scalar backward
        fd_surf.sum().backward()

        # Extract surface-level gradients
        grad_T_sfc[i0:i1] = T_chunk.grad[:, -1].cpu().numpy()
        grad_Q_sfc[i0:i1] = Q_chunk.grad[:, -1].cpu().numpy()
        grad_mu0_sfc[i0:i1] = mu0_chunk.grad.cpu().numpy()
    except Exception as e:
        print(f"    ERROR at chunk {chunk_idx}: {e}")
        grad_T_sfc[i0:i1] = np.nan
        grad_Q_sfc[i0:i1] = np.nan
        grad_mu0_sfc[i0:i1] = np.nan
        fd_sfc[i0:i1] = np.nan

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/ncol_s:.2f}s/column)")

# ── Reshape to 2D maps ───────────────────────────────────────────────────
fd_2d = fd_sfc.reshape(nlat_s, nlon_s)
grad_T_2d = grad_T_sfc.reshape(nlat_s, nlon_s)
grad_Q_2d = grad_Q_sfc.reshape(nlat_s, nlon_s)
grad_mu0_2d = grad_mu0_sfc.reshape(nlat_s, nlon_s)
lat_s = lats[lat_idx]
lon_s = lons[lon_idx]

print(f"\n2D maps: {nlat_s}×{nlon_s}")
print(f"F_sfc range: {np.nanmin(fd_2d):.1f} - {np.nanmax(fd_2d):.1f} W/m²")
print(f"∂F/∂T range: {np.nanmin(grad_T_2d):.4f} - {np.nanmax(grad_T_2d):.4f}")
print(f"∂F/∂q range: {np.nanmin(grad_Q_2d):.2f} - {np.nanmax(grad_Q_2d):.2f}")
print(f"∂F/∂μ₀ range: {np.nanmin(grad_mu0_2d):.1f} - {np.nanmax(grad_mu0_2d):.1f}")

# ── Save ─────────────────────────────────────────────────────────────────
out_path = "/tmp/global_gradients.npz"
np.savez(out_path,
    fd_sfc=fd_2d, grad_T=grad_T_2d, grad_Q=grad_Q_2d, grad_mu0=grad_mu0_2d,
    lats=lat_s, lons=lon_s, nlat=nlat_s, nlon=nlon_s,
)
print(f"\nSaved to {out_path}")
