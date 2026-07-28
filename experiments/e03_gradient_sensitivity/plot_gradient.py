"""Figure 3: Gradient correctness verification — autograd vs finite difference.

Three panels:
  (a) Convergence of FD to autograd as eps → 0 (V-shaped curve, log-log)
  (b) Vertical profile ∂F_surf/∂T_k: autograd vs FD
  (c) Convergence of ∂F/∂μ₀
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

with open("gradient_verification.json") as f:
    d = json.load(f)

p_full = np.array(d["p_full_hpa"])
nlev = d["nlev"]
ag_T = np.array(d["autograd"]["grad_T"])
ag_mu0 = d["autograd"]["grad_mu0"]
conv_T = d["convergence_T"]
conv_mu0 = d["convergence_mu0"]
fd_sampled = {int(k): v for k, v in d["finite_diff"]["grad_T_sampled"].items()}
sample_levels = sorted(fd_sampled.keys())

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# ── (a) Convergence of ∂F/∂T at mid-level ──
ax = axes[0]
eps_vals = [c["eps"] for c in conv_T]
rel_errs = [c["rel_err"] for c in conv_T]
ax.loglog(eps_vals, rel_errs, "bo-", linewidth=2, markersize=8, label="rel. error")
# Add O(eps²) reference line
eps_arr = np.array(eps_vals)
ax.loglog(eps_arr, 1e-6 * eps_arr**2, "k--", linewidth=1, alpha=0.5, label="$O(\\varepsilon^2)$ reference")
ax.set_xlabel("Perturbation $\\varepsilon$ (K)", fontsize=12)
ax.set_ylabel("Relative error $|FD - autograd| / |autograd|$", fontsize=12)
ax.set_title("(a) FD Convergence: $\\partial F_\\mathrm{sfc}/\\partial T_k$\n"
             "(mid-troposphere, level %d)" % (nlev // 2), fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(1e-8, 1e0)

# ── (b) Vertical gradient profile ──
ax = axes[1]
# Plot full autograd profile
ax.plot(ag_T, p_full, "r-", linewidth=2, label="Autograd", alpha=0.8)
# Plot FD at sampled levels
fd_vals = [fd_sampled[k] for k in sample_levels]
p_samples = [p_full[k] for k in sample_levels]
ax.plot(fd_vals, p_samples, "bx", markersize=8, markeredgewidth=1.5, label="Finite diff.")
ax.set_ylim(1020, 0.5)
ax.set_xlabel("$\\partial F_\\mathrm{sfc}/\\partial T_k$ (W m$^{-2}$ K$^{-1}$)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(b) Vertical Gradient Profile\n"
             "$\\partial F_\\mathrm{sfc}/\\partial T_k$ at each level", fontsize=12)
ax.legend(fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

# ── (c) Convergence of ∂F/∂μ₀ ──
ax = axes[2]
eps_mu0 = [c["eps"] for c in conv_mu0]
rel_mu0 = [c["rel_err"] for c in conv_mu0]
ax.loglog(eps_mu0, rel_mu0, "gs-", linewidth=2, markersize=8, label="rel. error")
eps_arr2 = np.array(eps_mu0)
ax.loglog(eps_arr2, 1e-5 * eps_arr2**2, "k--", linewidth=1, alpha=0.5, label="$O(\\varepsilon^2)$ reference")
ax.set_xlabel("Perturbation $\\varepsilon$ (dimensionless)", fontsize=12)
ax.set_ylabel("Relative error", fontsize=12)
ax.set_title("(c) FD Convergence: $\\partial F_\\mathrm{sfc}/\\partial \\mu_0$\n"
             "(autograd = %.1f W m$^{-2}$)" % ag_mu0, fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(1e-13, 1e-1)

plt.suptitle("Gradient Verification: Autograd vs Central Finite Differences\n"
             "(Clear-sky SW surface flux, ERA5 column, $\\mu_0 = 0.5$, 137 levels)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig3_gradient_verification.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig3_gradient_verification.png", dpi=200, bbox_inches="tight")
print("Saved fig3_gradient_verification")
