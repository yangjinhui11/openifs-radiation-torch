#!/usr/bin/env python3
"""Combined calibration figure: (a) 1-D albedo L-BFGS convergence,
(b) 2-D (albedo, cloud-fraction) recovery trajectory, (c) grid-search cost
scaling with parameter dimension."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r1 = json.load(open("calibration_results.json"))
r2 = json.load(open("calibration2d_results.json"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# --- (a) 1-D albedo convergence ---
ag1 = r1["autograd"]
evals = [h["eval"] for h in ag1["history"]]
losses = [h["loss"] for h in ag1["history"]]
ax1.semilogy(evals, losses, "r.-", linewidth=1.8, markersize=5,
             label="L-BFGS (autograd)")
ax1.set_xlabel("Forward+backward evaluations", fontsize=12)
ax1.set_ylabel("Loss $L$  [W$^2$ m$^{-4}$]", fontsize=12)
ax1.set_title("(a) 1-D: albedo calibration", fontsize=13)
ax1.set_xlim(0, max(evals) + 2)
ax1.grid(True, which="both", alpha=0.3)
ax1.annotate("$\\alpha_0$=0.10", xy=(2, losses[0]), fontsize=10, color="r")
ax1.annotate("recovered $\\alpha$=%.5f\n(truth 0.30)" % ag1["alpha_recovered"],
             xy=(evals[-1], losses[-1]), xytext=(evals[-1]*0.4, losses[-1]*1e4),
             fontsize=9, color="r", arrowprops=dict(arrowstyle="->", color="r"))

# --- (b) 2-D recovery trajectory ---
ag2 = r2["autograd"]
alphas = [h["theta"][0] for h in ag2["history"]]
cfs = [h["theta"][1] for h in ag2["history"]]
# subsample for clarity
sub = list(range(0, len(alphas), max(1, len(alphas)//20)))
ax2.plot([r2["theta_init"][0]], [r2["theta_init"][1]], "ks", markersize=12,
         label="init (0.15, 0.20)")
ax2.plot(alphas, cfs, "r.-", linewidth=1.5, markersize=4, label="L-BFGS path")
ax2.plot([r2["theta_true"][0]], [r2["theta_true"][1]], "g*", markersize=18,
         label="truth (0.25, 0.60)")
ax2.plot([ag2["theta_recovered"][0]], [ag2["theta_recovered"][1]], "b^",
         markersize=10, label="recovered (0.250, 0.600)")
ax2.set_xlabel("Surface albedo $\\alpha$", fontsize=12)
ax2.set_ylabel("Cloud fraction $c_f$", fontsize=12)
ax2.set_title("(b) 2-D: ($\\alpha$, $c_f$) recovery", fontsize=13)
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle("Gradient-based parameter calibration\n"
             "ERA5 columns, 137 levels, autograd + L-BFGS",
             fontsize=13, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig("fig5_calibration.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig5_calibration.png", dpi=200, bbox_inches="tight")
# Also copy to paper_figure for LaTeX
import shutil
shutil.copy("fig5_calibration.pdf", "../../paper_figure/fig5_calibration.pdf")
shutil.copy("fig5_calibration.png", "../../paper_figure/fig5_calibration.png")
print("Saved fig5_calibration")
print("1-D: alpha %.6f (%d evals) | 2-D: %s rel_err %.2e (%d evals)"
      % (ag1["alpha_recovered"], ag1["n_evals"],
         [round(x, 5) for x in ag2["theta_recovered"]],
         ag2["s_rel_err"], ag2["n_evals"]))
