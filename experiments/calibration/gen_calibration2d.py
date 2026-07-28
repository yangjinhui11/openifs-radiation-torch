#!/usr/bin/env python3
"""Experiment 6: 2-D calibration (surface albedo + cloud fraction scalar).

A genuinely identifiable 2-parameter inverse problem: broadband surface flux
depends DIFFERENTLY on albedo (controls surface reflection) and on cloud
fraction (controls how much the cloudy vs clear column is mixed), so the two
parameters can be jointly recovered from broadband flux observations across
diverse columns. This demonstrates multi-parameter gradient calibration and
provides the dimension where grid search is already 25x more expensive.

Output: /tmp/calibration2d_results.json
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch

RAD_ROOT = Path("/home/qixiang/yangjinhui/openifs/physics_callpar/openifs_radiation_pytorch")
sys.path.insert(0, str(RAD_ROOT))
os.environ.setdefault("DATA", str(RAD_ROOT))
from openifs_radiation.classic_sw.driver import sw_solver
for mod_name in ["swtt", "sw1s", "swni", "swclr", "swr"]:
    __import__(f"openifs_radiation.classic_sw.{mod_name}", fromlist=[mod_name])._jit = False

print("Loading ERA5...", flush=True)
g = np.load("/tmp/global_sw.npz")
NCOL = 16
day = np.where(g["mu0"] > 0.2)[0]
idx = day[np.linspace(0, len(day) - 1, NCOL).astype(int)]
nlev = int(g["nlev"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float64
T = torch.from_numpy(g["T"][idx]).to(dt).to(DEV)
Q = torch.from_numpy(g["Q"][idx]).to(dt).to(DEV)
O3 = torch.from_numpy(g["O3"][idx]).to(dt).to(DEV)
PH = torch.from_numpy(g["p_half"][idx]).to(dt).to(DEV)
MU0 = torch.from_numpy(g["mu0"][idx]).to(dt).to(DEV)
CO2 = torch.full((NCOL, nlev), 348e-6, dtype=dt, device=DEV)
AER = torch.full((NCOL, 6, nlev), 1e-3, dtype=dt, device=DEV)
PQS = torch.full((NCOL, nlev), 1e-3, dtype=dt, device=DEV)
# a fixed, realistic cloud optical thickness field (varies per column)
rng = np.random.default_rng(7)
PTAU_BASE = torch.from_numpy(
    rng.uniform(2.0, 8.0, size=(NCOL, 6, nlev)).astype(np.float64)
).to(DEV)
# only mid/lower troposphere has clouds
mask = torch.zeros(NCOL, 6, nlev, dtype=dt, device=DEV)
for j in range(nlev):
    mask[:, :, j] = 1.0 if 30 <= j <= 95 else 0.0
PTAU_BASE = PTAU_BASE * mask
POMEGA_C = torch.full((NCOL, 6, nlev), 0.9999, dtype=dt, device=DEV)
PCG_C = torch.full((NCOL, 6, nlev), 0.85, dtype=dt, device=DEV)
print("  NCOL=%d mu0=%.2f-%.2f" % (NCOL, MU0.min(), MU0.max()), flush=True)


def run_flux(theta):
    """theta = (alpha, cf): scalar albedo broadcast + scalar cloud fraction.
    Returns broadband surface downward flux (NCOL,)."""
    alpha, cf = theta[0], theta[1]
    alb = alpha.expand(NCOL, 6)
    pcld = cf.expand(NCOL, nlev)
    out = sw_solver(p_half=PH, temp=T, q_h2o=Q, q_co2_vmr=CO2, mu0=MU0,
                    albd=alb, albp=alb, pcldsw=pcld, aer=AER, poz=O3,
                    pcg=PCG_C, pomega=POMEGA_C, ptau=PTAU_BASE, pqs=PQS)
    return out["fd_total"][:, -1]


# truth and init
THETA_TRUE = torch.tensor([0.25, 0.6], dtype=dt, device=DEV)  # alpha=0.25, cf=0.6
THETA_INIT = torch.tensor([0.15, 0.2], dtype=dt, device=DEV)   # far from truth
print("\ntruth (alpha, cf) =", THETA_TRUE.tolist(), flush=True)
with torch.no_grad():
    F_obs = run_flux(THETA_TRUE)
    print("F_obs: %.1f-%.1f W/m2" % (F_obs.min(), F_obs.max()), flush=True)
    # report identifiability: flux sensitivity to each param at init
    for name, pert in [("alpha", torch.tensor([0.01, 0.0], dtype=dt, device=DEV)),
                       ("cf", torch.tensor([0.0, 0.05], dtype=dt, device=DEV))]:
        Fp = run_flux(THETA_INIT + pert)
        Fm = run_flux(THETA_INIT - pert)
        print("  dF/d%s ~ %.2f W/m2 per unit" % (name, ((Fp - Fm) / (2 * pert.norm())).mean()), flush=True)

# L-BFGS
print("\n=== 2-D gradient-based calibration ===", flush=True)
theta = THETA_INIT.clone().requires_grad_()
opt = torch.optim.LBFGS([theta], lr=0.05, max_iter=20, line_search_fn="strong_wolfe")
hist = []
n_evals = [0]
t0 = time.time()


def closure():
    n_evals[0] += 1
    opt.zero_grad()
    F_sim = run_flux(theta)
    loss = ((F_sim - F_obs) ** 2).mean()
    loss.backward()
    serr = (theta - THETA_TRUE).norm().item() / THETA_TRUE.norm().item()
    hist.append({"eval": n_evals[0], "loss": loss.item(),
                 "theta": theta.detach().tolist(),
                 "grad": theta.grad.tolist(), "s_rel_err": serr})
    return loss


for step in range(10):
    opt.step(closure)
    h = hist[-1]
    print("  step %d: loss=%.3e theta=[%.4f,%.4f] s_err=%.3e n_eval=%d"
          % (step, h["loss"], h["theta"][0], h["theta"][1], h["s_rel_err"], n_evals[0]), flush=True)
    if h["loss"] < 1e-9:
        break

t_ag = time.time() - t0
theta_rec = theta.detach()
s_rel_err = (theta_rec - THETA_TRUE).norm().item() / THETA_TRUE.norm().item()
print("  RECOVERED (alpha,cf) = [%.5f, %.5f]  (truth [%.2f,%.2f])"
      % (theta_rec[0], theta_rec[1], THETA_TRUE[0], THETA_TRUE[1]), flush=True)
print("  rel param err=%.2e, time=%.1fs, evals=%d" % (s_rel_err, t_ag, n_evals[0]), flush=True)

# Grid search cost at 2-D
with torch.no_grad():
    t0 = time.time()
    _ = run_flux(THETA_INIT)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.time() - t0
print("\n=== Grid-search cost (2-D) ===", flush=True)
for n in [5, 10, 15]:
    print("  n=%d: %d evals = %.0f s" % (n, n * n, n * n * t_fwd), flush=True)

result = {"ncol": NCOL, "device": DEV,
          "theta_true": THETA_TRUE.tolist(), "theta_init": THETA_INIT.tolist(),
          "autograd": {"theta_recovered": theta_rec.tolist(), "s_rel_err": s_rel_err,
                       "final_loss": hist[-1]["loss"], "time_s": t_ag,
                       "n_evals": n_evals[0], "history": hist},
          "grid_cost": {"t_fwd_s": t_fwd, "evals": {str(n): n*n for n in [5,10,15]},
                        "time_s": {str(n): n*n*t_fwd for n in [5,10,15]}}}
with open("/tmp/calibration2d_results.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nDONE: 2-D rel_err=%.2e in %d evals" % (s_rel_err, n_evals[0]), flush=True)
