#!/usr/bin/env python3
"""Performance benchmark: CPU vs GPU, SW chain, varying column counts.

Tests the production JIT-compiled path (the fast one used for flux computation)
on both CPU and GPU at multiple column counts.

Output: /tmp/perf_benchmark.json
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

# ── Load ERA5 data ──
g = np.load("/tmp/global_sw.npz")
nlev = int(g["nlev"])
T_template = g["T"][:1].astype(np.float64)   # template (1, nlev)
Q_template = g["Q"][:1].astype(np.float64)
O3_template = g["O3"][:1].astype(np.float64)
p_half_template = g["p_half"][:1].astype(np.float64)
mu0_template = np.maximum(g["mu0"][:1], 0.5).astype(np.float64)
dt = torch.float64
PCARDI = 348.0e-6

def build_inputs(ncol, dev):
    """Build sw_solver inputs by tiling the template column."""
    T = torch.from_numpy(np.tile(T_template, (ncol, 1))).to(dt).to(dev)
    Q = torch.from_numpy(np.tile(Q_template, (ncol, 1))).to(dt).to(dev)
    p_h = torch.from_numpy(np.tile(p_half_template, (ncol, 1))).to(dt).to(dev)
    mu0 = torch.from_numpy(np.tile(mu0_template, (ncol,))).to(dt).to(dev)
    co2 = torch.full((ncol, nlev), PCARDI, dtype=dt, device=dev)
    o3 = torch.from_numpy(np.tile(O3_template, (ncol, 1))).to(dt).to(dev)
    albd = torch.full((ncol, 6), 0.15, dtype=dt, device=dev)
    albp = torch.full((ncol, 6), 0.15, dtype=dt, device=dev)
    pcldsw = torch.full((ncol, nlev), 1e-6, dtype=dt, device=dev)
    aer = torch.full((ncol, 6, nlev), 1e-3, dtype=dt, device=dev)
    poz = o3.clone()
    pcg = torch.full((ncol, 6, nlev), 0.85, dtype=dt, device=dev)
    pomega = torch.full((ncol, 6, nlev), 0.99, dtype=dt, device=dev)
    ptau = torch.full((ncol, 6, nlev), 1e-3, dtype=dt, device=dev)
    pqs = torch.zeros((ncol, nlev), dtype=dt, device=dev)
    return dict(p_half=p_h, temp=T, q_h2o=Q, q_co2_vmr=co2, mu0=mu0,
                albd=albd, albp=albp, pcldsw=pcldsw, aer=aer, poz=poz,
                pcg=pcg, pomega=pomega, ptau=ptau, pqs=pqs)

def benchmark(dev, ncols_list, n_repeat=3):
    """Benchmark sw_solver on a device for various column counts."""
    results = []
    for ncol in ncols_list:
        inp = build_inputs(ncol, dev)
        # Warmup
        with torch.no_grad():
            _ = sw_solver(**inp)
            if dev == "cuda":
                torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(n_repeat):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = sw_solver(**inp)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)

        median_time = sorted(times)[len(times) // 2]
        flux = out["fd_total"][:, -1].mean().item()
        results.append({
            "ncol": ncol,
            "time_ms": median_time * 1000,
            "per_col_ms": median_time * 1000 / ncol,
            "flux": flux,
        })
        print(f"  {dev} ncol={ncol:>6d}: {median_time*1000:.0f} ms "
              f"({median_time*1000/ncol:.3f} ms/col), flux={flux:.1f}")
    return results

# ── Run benchmarks ──
ncols_list = [1, 16, 64, 256, 1024, 4096, 8192]

print("=== CPU Benchmark (JIT production path) ===")
cpu_results = benchmark("cpu", ncols_list)

print("\n=== GPU Benchmark (JIT production path, NVIDIA A100) ===")
gpu_results = benchmark("cuda", ncols_list)

# ── Compute speedups ──
print("\n=== CPU vs GPU Speedup ===")
for cpu_r, gpu_r in zip(cpu_results, gpu_results):
    speedup = cpu_r["time_ms"] / gpu_r["time_ms"]
    print(f"  ncol={cpu_r['ncol']:>6d}: CPU={cpu_r['time_ms']:.0f}ms, "
          f"GPU={gpu_r['time_ms']:.0f}ms, speedup={speedup:.1f}x")

# ── Save ──
result = {
    "cpu": cpu_results,
    "gpu": gpu_results,
    "nlev": nlev,
    "device": "NVIDIA A100 80GB" if torch.cuda.is_available() else "N/A",
    "cpu_info": "single thread",
}
with open("/tmp/perf_benchmark.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to /tmp/perf_benchmark.json")
