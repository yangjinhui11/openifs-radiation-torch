#!/usr/bin/env python3
"""Figure 6 (simplified): single-panel per-column flux residual collapse.
Shows the physical result of calibration (flux fit quality), complementing
fig5 which shows the optimization process (loss/parameter convergence).
This panel is NOT redundant with fig5 — it is the only place the physical
flux residual is shown."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
fr = json.load(open(os.path.join(_HERE, "flux_residual.json")))
resid_init = np.abs(np.array(fr["resid_init"]))
resid_rec = np.abs(np.array(fr["resid_rec"]))

fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

cols = np.arange(1, len(resid_init) + 1)
width = 0.4
ax.bar(cols - width/2, resid_init, width, color="red", alpha=0.75,
       label="before calibration ($\\theta_0$)")
ax.bar(cols + width/2, resid_rec, width, color="blue", alpha=0.9,
       label="after calibration ($\\theta_\\mathrm{rec}$)")
ax.set_yscale("log")
ax.set_xlabel("ERA5 column index", fontsize=12)
ax.set_ylabel(r"$|F_\mathrm{sim} - F_\mathrm{obs}|$ (W m$^{-2}$)", fontsize=12)
ax.set_title("Per-column surface SW flux residual: before vs after 2-D calibration",
             fontsize=13)
ax.set_xticks(cols[::2])
ax.grid(True, axis="y", which="both", alpha=0.3)
ax.legend(loc="lower right", fontsize=11)

# reference lines for the two bands
ax.axhline(resid_init.mean(), color="red", linestyle=":", linewidth=0.9, alpha=0.6)
ax.axhline(resid_rec.mean(), color="blue", linestyle=":", linewidth=0.9, alpha=0.6)
ax.annotate("before: $\\sim$$10^{2}$ W m$^{-2}$\n(column-mean RMSE 198.9)",
            xy=(14, resid_init.mean()), xytext=(9, resid_init.mean() * 1.8),
            fontsize=10, color="red",
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.6))
ax.annotate("after: $\\sim$$10^{-5}$ W m$^{-2}$\n(RMSE $1.8\\times10^{-5}$)",
            xy=(14, resid_rec.mean()), xytext=(9, resid_rec.mean() * 0.04),
            fontsize=10, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.6))

plt.tight_layout()
plt.savefig("fig6b_flux_fit.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig6b_flux_fit.png", dpi=200, bbox_inches="tight")
print("Saved fig6b_flux_fit (single panel, residual collapse)")
print("RMSE: %.4f -> %.4e (factor %.1e)" %
      (fr["rmse_init"], fr["rmse_rec"], fr["rmse_init"]/fr["rmse_rec"]))
