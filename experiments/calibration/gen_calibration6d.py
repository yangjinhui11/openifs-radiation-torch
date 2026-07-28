#!/usr/bin/env python3
"""Experiment 5: Multi-parameter gradient-based calibration (6-band cloud
optical thickness scaling factors).

This extends the single-parameter albedo recovery of Experiment 4 to a
6-dimensional inverse problem, demonstrating that the differentiable scheme
can calibrate multiple physical parameters jointly and that the gradient
approach scales to dimensions where grid search is infeasible.

Controlled-recovery design:
  Forward model: F_sim(s) = sw_solver(s_nu * ptau_ref, pomega, pcg, ...)
    where s = (s_1,...,s_6) are per-band cloud-optical-thickness scaling
    factors, and ptau_ref is a fixed reference cloud optical thickness field
    (injected realistic liquid clouds so the cloudy path is exercised).
  1. Pick s_true (known, perturbed around 1.0).
  2. F_obs = F_sim(s_true) on N diverse ERA5 columns.
  3. s_0 = (1,...,1) (parameterization default).
  4. Minimize L(s) = MSE(F_sim(s), F_obs) via L-BFGS + autograd.
  5. Report recovery, convergence, cost; compare grid-search scaling.

Output: /tmp/calibration6d_results.json
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

# Disable JIT for autograd
for mod_name in ["swtt", "sw1s", "swni", "swclr", "swr"]:
    __import__(f"openifs_radiation.classic_sw.{mod_name}", fromlist=[mod_name])._jit = False

# ── Load diverse ERA5 columns ────────────────────────────────────────────
print("Loading ERA5...", flush=True)
g = np.load("/tmp/global_sw.npz")
NCOL = 16  # keep small for speed; 16 diverse columns suffice for 6-param ID
day = np.where(g["mu0"] > 0.2)[0]
idx = day[np.linspace(0, len(day) - 1, NCOL).astype(int)]
nlev = int(g["nlev"])
print("  NCOL=%d mu0=%.2f-%.2f T_sfc=%.0f-%.0f"
      % (NCOL, g["mu0"][idx].min(), g["mu0"][idx].max(),
         g["T"][idx, -1].min(), g["T"][idx, -1].max()), flush=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float64
T = torch.from_numpy(g["T"][idx]).to(dt).to(DEV)
Q = torch.from_numpy(g["Q"][idx]).to(dt).to(DEV)
O3 = torch.from_numpy(g["O3"][idx]).to(dt).to(DEV)
PH = torch.from_numpy(g["p_half"][idx]).to(dt).to(DEV)
MU0 = torch.from_numpy(g["mu0"][idx]).to(dt).to(DEV)
CO2 = torch.full((NCOL, nlev), 348e-6, dtype=dt, device=DEV)
PCLD = torch.full((NCOL, nlev), 0.5, dtype=dt, device=DEV)
AER = torch.full((NCOL, 6, nlev), 1e-3, dtype=dt, device=DEV)
PQS = torch.full((NCOL, nlev), 1e-3, dtype=dt, device=DEV)
ALB = torch.full((NCOL, 6), 0.15, dtype=dt, device=DEV)

# Reference cloud optical thickness: each band has a DISTINCT vertical
# structure that VARIES across columns, so that per-band scaling factors
# are jointly identifiable. A single uniform cloud makes s degenerate
# (many s give the same flux); varying cloud structure per column provides
# independent constraints. We model each column's band-b cloud as a Gaussian
# profile whose peak level and amplitude vary across columns (mimicking the
# spatial heterogeneity of real cloud fields).
PTAU_REF = torch.zeros(NCOL, 6, nlev, dtype=dt, device=DEV)
# Per-band base (peak, hw, amp); each column gets a random perturbation of
# these so columns differ. Seed for reproducibility.
rng = np.random.default_rng(42)
band_profiles = [
    (50, 25, 2.0),   # band 1: UV
    (60, 20, 3.5),   # band 2: visible
    (45, 30, 4.5),   # band 3: visible
    (75, 15, 6.0),   # band 4: NIR
    (40, 35, 5.0),   # band 5: NIR
    (65, 25, 4.0),   # band 6: NIR
]
levels = torch.arange(nlev, dtype=dt, device=DEV)
for c in range(NCOL):
    for b, (peak, hw, amp) in enumerate(band_profiles):
        # per-column perturbation: peak ±15, amp ×[0.5,1.5]
        p_c = peak + rng.uniform(-15, 15)
        a_c = amp * rng.uniform(0.5, 1.5)
        prof = a_c * torch.exp(-((levels - p_c) ** 2) / (2 * hw ** 2))
        PTAU_REF[c, b, :] = prof
POMEGA_REF = torch.full((NCOL, 6, nlev), 0.9999, dtype=dt, device=DEV)
PCG_REF = torch.full((NCOL, 6, nlev), 0.85, dtype=dt, device=DEV)
print("  ptau_ref: %.2f-%.2f, cloudy cells=%d/%d"
      % (PTAU_REF.min().item(), PTAU_REF.max().item(),
         (PTAU_REF > 0.5).sum().item(), PTAU_REF.numel()), flush=True)


def run_forcing(s):
    """Cloud-induced surface flux anomaly = F(cloudy) - F(clear), per band.

    This removes the clear-sky background, against which the cloud signal is
    small. The cloud forcing is directly sensitive to ptau, making the per-band
    scaling factors identifiable. Returns (6, NCOL).

    NOTE: each call performs TWO forward solves (cloudy + clear). The clear-sky
    run does not depend on s, so it could be cached; we recompute it here for
    simplicity. Eval counts in the cost accounting reflect this 2-forward
    structure (one cloudy + one clear per loss evaluation)."""
    # cloudy run
    ptau = PTAU_REF * s.view(1, 6, 1)
    out_c = sw_solver(p_half=PH, temp=T, q_h2o=Q, q_co2_vmr=CO2, mu0=MU0,
                      albd=ALB, albp=ALB, pcldsw=PCLD, aer=AER, poz=O3,
                      pcg=PCG_REF, pomega=POMEGA_REF, ptau=ptau, pqs=PQS)
    # clear-sky run: zero cloud optical depth and cloud fraction
    ptau_zero = torch.zeros_like(PTAU_REF)
    out_0 = sw_solver(p_half=PH, temp=T, q_h2o=Q, q_co2_vmr=CO2, mu0=MU0,
                      albd=ALB, albp=ALB, pcldsw=torch.zeros_like(PCLD),
                      aer=AER, poz=O3, pcg=PCG_REF, pomega=POMEGA_REF,
                      ptau=ptau_zero, pqs=PQS)
    # per-band surface downward flux, anomaly = cloudy - clear
    return out_c["fd_band"][:, :, -1] - out_0["fd_band"][:, :, -1]


# ── Step 1: synthetic observations ───────────────────────────────────────
S_TRUE = torch.tensor([1.3, 0.8, 1.1, 0.9, 1.2, 0.85], dtype=dt, device=DEV)
print("\ntruth s_true =", S_TRUE.tolist(), flush=True)
with torch.no_grad():
    F_obs = run_forcing(S_TRUE)
    print("F_obs (cloud forcing): %.1f-%.1f W/m2" % (F_obs.min().item(), F_obs.max().item()), flush=True)

# ── Step 2: L-BFGS gradient-based calibration ────────────────────────────
print("\n=== Gradient-based calibration (6-D, autograd + L-BFGS) ===", flush=True)
s = torch.ones(6, dtype=dt, device=DEV, requires_grad=True)  # init = default
opt = torch.optim.LBFGS([s], lr=0.05, max_iter=20, line_search_fn="strong_wolfe")
hist = []
n_evals = [0]
t0 = time.time()


def closure():
    n_evals[0] += 1
    opt.zero_grad()
    F_sim = run_forcing(s)
    loss = ((F_sim - F_obs) ** 2).mean()
    loss.backward()
    gnorm = s.grad.norm().item()
    serr = (s - S_TRUE).norm().item() / S_TRUE.norm().item()
    hist.append({"eval": n_evals[0], "loss": loss.item(),
                 "s": s.detach().tolist(), "grad_norm": gnorm,
                 "s_rel_err": serr})
    return loss


for step in range(12):
    opt.step(closure)
    h = hist[-1]
    print("  step %d: loss=%.3e |grad|=%.3e s_err=%.3e n_eval=%d"
          % (step, h["loss"], h["grad_norm"], h["s_rel_err"], n_evals[0]), flush=True)
    if h["loss"] < 1e-9:
        break

t_ag = time.time() - t0
s_rec = s.detach()
s_rel_err = (s_rec - S_TRUE).norm().item() / S_TRUE.norm().item()
print("  RECOVERED s =", [round(x, 4) for x in s_rec.tolist()], flush=True)
print("  rel param error = %.2e, time=%.1fs, evals=%d" % (s_rel_err, t_ag, n_evals[0]), flush=True)

# ── Step 3: estimate grid-search cost at this dimensionality ─────────────
# Full grid in 6-D is infeasible; we cost one coordinate-descent baseline:
# 5 pts × 6 coords = 30 evals per sweep, a few sweeps. And note n^6 scaling.
print("\n=== Grid-search cost projection ===", flush=True)
# Empirically time 1 forward to project grid costs.
with torch.no_grad():
    t0 = time.time()
    _ = run_forcing(torch.ones(6, dtype=dt, device=DEV))
    if DEV == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.time() - t0
# For a grid of n points per dimension:
print("  1 forward = %.2fs" % t_fwd, flush=True)
for n in [3, 5, 7]:
    neval = n ** 6
    print("  full grid n=%d: %d evals = %.0f hours" % (n, neval, neval * t_fwd / 3600), flush=True)
# Coordinate descent (practical non-gradient baseline): ~6*n per sweep
for n in [5, 10]:
    neval = 6 * n * 3  # 3 sweeps
    print("  coord descent n=%d: %d evals = %.0f s" % (n, neval, neval * t_fwd), flush=True)

# ── Save ─────────────────────────────────────────────────────────────────
result = {
    "nlev": nlev, "ncol": NCOL, "device": DEV, "ndim": 6,
    "s_true": S_TRUE.tolist(), "s_init": [1.0] * 6,
    "autograd": {"s_recovered": s_rec.tolist(), "s_rel_err": s_rel_err,
                 "final_loss": hist[-1]["loss"], "time_s": t_ag,
                 "n_evals": n_evals[0], "history": hist},
    "grid_cost": {"t_fwd_s": t_fwd, "full_grid_hours": {str(n): n**6 * t_fwd / 3600 for n in [3,5,7]},
                  "coord_descent_s": {str(n): 6*n*3*t_fwd for n in [5,10]}},
}
with open("/tmp/calibration6d_results.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved /tmp/calibration6d_results.json", flush=True)
print("DONE: 6-D recovery rel_err=%.2e in %d evals/%.1fs"
      % (s_rel_err, n_evals[0], t_ag), flush=True)
