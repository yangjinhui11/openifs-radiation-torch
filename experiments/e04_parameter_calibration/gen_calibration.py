#!/usr/bin/env python3
"""Experiment 4: Gradient-based calibration of a radiation parameter.

Demonstrates the key advantage of the differentiable scheme: a physical
parameter can be calibrated by gradient descent against an observed flux,
without hand-coding an adjoint.

Design (controlled-recovery test on diverse global columns):
  1. Pick a "true" surface albedo alpha_true (known).
  2. Generate synthetic "observations": F_obs = sw_solver(alpha_true) on
     32 ERA5 columns spanning the global daylit belt (mu0 = 0.10-0.75).
  3. Initialize alpha_guess far from alpha_true (requires_grad=True).
  4. Minimize L(alpha) = MSE(F_sim(alpha), F_obs) via L-BFGS, using the
     autograd gradient dL/dalpha from a single backward pass per step.
  5. Compare cost vs. finite-difference (grid-search) calibration.

Output: /tmp/calibration_results.json
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_config import setup_paths, find_era5
setup_paths()
_HERE = Path(__file__).resolve().parent

from openifs_radiation.classic_sw.driver import sw_solver

# Disable all JIT kernels — they break autograd (in-place ops).
for mod_name in ["swtt", "sw1s", "swni", "swclr", "swr"]:
    m = __import__(f"openifs_radiation.classic_sw.{mod_name}", fromlist=[mod_name])
    m._jit = False

# ── Load diverse ERA5 columns from the global grid ───────────────────────
print("Loading global ERA5 data...", flush=True)
g = np.load(find_era5("global_sw.npz"))
NCOL = 32
day = np.where(g["mu0"] > 0.1)[0]
idx = day[np.linspace(0, len(day) - 1, NCOL).astype(int)]
nlev = int(g["nlev"])
print("  NCOL=%d, nlev=%d, mu0 range %.3f-%.3f, T_sfc %.1f-%.1f K"
      % (NCOL, nlev, g["mu0"][idx].min(), g["mu0"][idx].max(),
         g["T"][idx, -1].min(), g["T"][idx, -1].max()), flush=True)

dt = torch.float64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("  Device:", DEV, flush=True)

T = torch.from_numpy(g["T"][idx]).to(dt).to(DEV)
Q = torch.from_numpy(g["Q"][idx]).to(dt).to(DEV)
O3 = torch.from_numpy(g["O3"][idx]).to(dt).to(DEV)
PH = torch.from_numpy(g["p_half"][idx]).to(dt).to(DEV)
MU0 = torch.from_numpy(g["mu0"][idx]).to(dt).to(DEV)
CO2 = torch.full((NCOL, nlev), 348e-6, dtype=dt, device=DEV)
PCLD = torch.full((NCOL, nlev), 1e-6, dtype=dt, device=DEV)
AER = torch.full((NCOL, 6, nlev), 1e-3, dtype=dt, device=DEV)
PCG = torch.full((NCOL, 6, nlev), 0.85, dtype=dt, device=DEV)
POM = torch.full((NCOL, 6, nlev), 0.99, dtype=dt, device=DEV)
PTAU = torch.full((NCOL, 6, nlev), 1e-3, dtype=dt, device=DEV)
PQS = torch.full((NCOL, nlev), 1e-3, dtype=dt, device=DEV)


def run_flux(alpha):
    albd = alpha.expand(NCOL, 6)
    albp = alpha.expand(NCOL, 6)
    out = sw_solver(p_half=PH, temp=T, q_h2o=Q, q_co2_vmr=CO2, mu0=MU0,
                    albd=albd, albp=albp, pcldsw=PCLD, aer=AER, poz=O3,
                    pcg=PCG, pomega=POM, ptau=PTAU, pqs=PQS)
    return out["fd_total"][:, -1]


# ── Step 1: synthetic observations with known truth ──────────────────────
ALB_TRUE = 0.30
print("\nGenerating observations with alpha_true=%.2f" % ALB_TRUE, flush=True)
with torch.no_grad():
    F_obs = run_flux(torch.tensor(ALB_TRUE, dtype=dt, device=DEV))
    print("  F_obs: %.1f to %.1f W/m2" % (F_obs.min().item(), F_obs.max().item()), flush=True)

# ── Step 2: L-BFGS gradient-based calibration ────────────────────────────
print("\n=== Gradient-based calibration (autograd + L-BFGS) ===", flush=True)
ALB_INIT = 0.10
alpha = torch.tensor(ALB_INIT, dtype=dt, device=DEV, requires_grad=True)
opt = torch.optim.LBFGS([alpha], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")
hist_ag = []
n_evals = [0]
t0 = time.time()


def closure():
    n_evals[0] += 1
    opt.zero_grad()
    F_sim = run_flux(alpha)
    loss = ((F_sim - F_obs) ** 2).mean()
    loss.backward()
    hist_ag.append({"eval": n_evals[0], "alpha": alpha.item(),
                    "loss": loss.item(), "grad": alpha.grad.item()})
    return loss


for step in range(8):
    opt.step(closure)
    h = hist_ag[-1]
    print("  step %d: alpha=%.5f loss=%.3e grad=%.3e n_eval=%d"
          % (step, h["alpha"], h["loss"], h["grad"], n_evals[0]), flush=True)
    if h["loss"] < 1e-8:
        break

t_ag = time.time() - t0
alpha_ag = alpha.item()
print("  RECOVERED alpha=%.6f (truth %.2f), time=%.1fs, evals=%d"
      % (alpha_ag, ALB_TRUE, t_ag, n_evals[0]), flush=True)

# ── Step 3: grid-search baseline ─────────────────────────────────────────
print("\n=== Grid-search calibration (baseline) ===", flush=True)
alphas_grid = np.linspace(0.05, 0.50, 19)
losses_grid = []
t0 = time.time()
with torch.no_grad():
    for a in alphas_grid:
        at = torch.tensor(a, dtype=dt, device=DEV)
        F_sim = run_flux(at)
        losses_grid.append(((F_sim - F_obs) ** 2).mean().item())
t_gs = time.time() - t0
i_best = int(np.argmin(losses_grid))
alpha_gs = float(alphas_grid[i_best])
print("  best grid alpha=%.4f (residual %.4f), time=%.1fs, evals=%d"
      % (alpha_gs, ALB_TRUE - alpha_gs, t_gs, len(alphas_grid)), flush=True)

# ── Save ─────────────────────────────────────────────────────────────────
result = {
    "nlev": nlev, "ncol": NCOL, "device": DEV,
    "alpha_true": ALB_TRUE, "alpha_init": ALB_INIT,
    "autograd": {"alpha_recovered": alpha_ag, "final_loss": hist_ag[-1]["loss"],
                 "time_s": t_ag, "n_evals": n_evals[0], "history": hist_ag},
    "grid_search": {"alpha_recovered": alpha_gs, "final_loss": losses_grid[i_best],
                    "time_s": t_gs, "n_evals": len(alphas_grid),
                    "alphas": alphas_grid.tolist(), "losses": losses_grid},
}
_out = _HERE / "calibration_results.json"
with open(_out, "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved %s" % _out, flush=True)
print("DONE: autograd alpha=%.5f in %d evals/%.1fs | grid alpha=%.5f in %d evals/%.1fs"
      % (alpha_ag, n_evals[0], t_ag, alpha_gs, len(alphas_grid), t_gs), flush=True)
