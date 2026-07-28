#!/usr/bin/env python3
"""Recompute flux residuals at theta_init and theta_recovered for the 2-D
experiment, to produce a before-vs-after scatter of F_sim vs F_obs."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

RAD_ROOT = Path("/home/qixiang/yangjinhui/openifs/physics_callpar/openifs_radiation_pytorch")
sys.path.insert(0, str(RAD_ROOT))
os.environ.setdefault("DATA", str(RAD_ROOT))
from openifs_radiation.classic_sw.driver import sw_solver
for mod_name in ["swtt", "sw1s", "swni", "swclr", "swr"]:
    __import__(f"openifs_radiation.classic_sw.{mod_name}", fromlist=[mod_name])._jit = False

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
rng = np.random.default_rng(7)
PTAU_BASE = torch.from_numpy(
    rng.uniform(2.0, 8.0, size=(NCOL, 6, nlev)).astype(np.float64)).to(DEV)
mask = torch.zeros(NCOL, 6, nlev, dtype=dt, device=DEV)
for j in range(nlev):
    mask[:, :, j] = 1.0 if 30 <= j <= 95 else 0.0
PTAU_BASE = PTAU_BASE * mask
POMEGA_C = torch.full((NCOL, 6, nlev), 0.9999, dtype=dt, device=DEV)
PCG_C = torch.full((NCOL, 6, nlev), 0.85, dtype=dt, device=DEV)


def run_flux(theta):
    theta_t = torch.tensor(theta, dtype=dt, device=DEV)
    alpha, cf = theta_t[0], theta_t[1]
    alb = alpha.expand(NCOL, 6)
    pcld = cf.expand(NCOL, nlev)
    out = sw_solver(p_half=PH, temp=T, q_h2o=Q, q_co2_vmr=CO2, mu0=MU0,
                    albd=alb, albp=alb, pcldsw=pcld, aer=AER, poz=O3,
                    pcg=PCG_C, pomega=POMEGA_C, ptau=PTAU_BASE, pqs=PQS)
    return out["fd_total"][:, -1].detach().cpu().numpy()


# load recovered theta
r2 = json.load(open("/tmp/calibration2d_results.json"))
theta_true = r2["theta_true"]
theta_init = r2["theta_init"]
theta_rec = r2["autograd"]["theta_recovered"]

with torch.no_grad():
    F_obs = run_flux(theta_true)
    F_init = run_flux(theta_init)
    F_rec = run_flux(theta_rec)

out = {
    "theta_true": theta_true, "theta_init": theta_init, "theta_recovered": theta_rec,
    "F_obs": F_obs.tolist(), "F_init": F_init.tolist(), "F_rec": F_rec.tolist(),
    "resid_init": (F_init - F_obs).tolist(),
    "resid_rec": (F_rec - F_obs).tolist(),
    "rmse_init": float(np.sqrt(np.mean((F_init - F_obs) ** 2))),
    "rmse_rec": float(np.sqrt(np.mean((F_rec - F_obs) ** 2))),
}
with open("/tmp/flux_residual.json", "w") as f:
    json.dump(out, f, indent=2)
print("RMSE init: %.4f W/m2" % out["rmse_init"])
print("RMSE rec : %.4e W/m2" % out["rmse_rec"])
print("improvement factor: %.1e" % (out["rmse_init"] / out["rmse_rec"]))
