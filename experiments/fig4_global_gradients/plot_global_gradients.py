"""Figure 4: Global 2D radiation sensitivity fields via autograd.

4-panel global map:
  (a) F_sfc (surface downward SW flux, reference)
  (b) ∂F_sfc/∂T_sfc (temperature sensitivity)
  (c) ∂F_sfc/∂q_sfc (water vapor sensitivity)
  (d) ∂F_sfc/∂μ₀ (solar zenith angle sensitivity)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

d = np.load("global_gradients.npz")
lats = d["lats"]
lons = d["lons"]
fd_sfc = d["fd_sfc"]
grad_T = d["grad_T"]
grad_Q = d["grad_Q"]
grad_mu0 = d["grad_mu0"]

# Divergent blue-white-red colormap for signed quantities
div_cmap = "RdBu_r"

fig, axes = plt.subplots(2, 2, figsize=(18, 10), subplot_kw={"projection": None})

panels = [
    (axes[0, 0], fd_sfc, "(a) SW Surface Downward Flux",
     "W m$^{-2}$", "YlOrRd", 0, 1000),
    (axes[0, 1], grad_T, "(b) $\\partial F_\\mathrm{sfc}/\\partial T_\\mathrm{sfc}$",
     "W m$^{-2}$ K$^{-1}$", div_cmap, None, None),
    (axes[1, 0], grad_Q, "(c) $\\partial F_\\mathrm{sfc}/\\partial q_\\mathrm{sfc}$",
     "W m$^{-2}$ (kg/kg)$^{-1}$", div_cmap, None, None),
    (axes[1, 1], grad_mu0, "(d) $\\partial F_\\mathrm{sfc}/\\partial \\mu_0$",
     "W m$^{-2}$", "YlOrRd", 0, 1400),
]

for ax, field, title, label, cmap, vmin, vmax in panels:
    # Mask NaN
    field_ma = np.ma.array(field, mask=np.isnan(field))
    if vmin is not None:
        im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
    else:
        # Symmetric range for divergent
        mx = np.nanpercentile(np.abs(field), 98)
        im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap,
                           vmin=-mx, vmax=mx, shading="auto")
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=12)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(label, fontsize=9)

plt.suptitle("Global Radiation Sensitivity Fields (Autograd Gradients)\n"
             "ERA5 00Z 1 Jan 2024, clear sky, $\\mu_0 \\geq 0.15$, 137 levels",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fig4_global_gradients.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig4_global_gradients.png", dpi=200, bbox_inches="tight")
import shutil
shutil.copy("fig4_global_gradients.pdf", "../../paper_figure/fig4_global_gradients.pdf")
shutil.copy("fig4_global_gradients.png", "../../paper_figure/fig4_global_gradients.png")
print("Saved fig4_global_gradients")
