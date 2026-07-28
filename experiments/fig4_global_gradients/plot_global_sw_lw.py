"""SW+LW joint global feedback kernel maps from real ERA5 data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("global_sw_lw_full.npz")
lats = d["lats"]; lons = d["lons"]

fig, axes = plt.subplots(3, 2, figsize=(20, 18))

panels = [
    (axes[0,0], d["sw_sfc"], "(a) SW Surface Downward Flux", "W m$^{-2}$", "YlOrRd", 0, 1200),
    (axes[0,1], d["lw_olr"], "(b) LW OLR (TOA Upward)", "W m$^{-2}$", "YlOrRd", 150, 350),
    (axes[1,0], d["sw_grad_q"], "(c) $\\partial F_{\\mathrm{SW}}/\\partial q_{\\mathrm{sfc}}$", "W m$^{-2}$ (kg/kg)$^{-1}$", "RdBu_r", None, None),
    (axes[1,1], d["lw_grad_Ts"], "(d) $\\partial F_{\\mathrm{OLR}}/\\partial T_{\\mathrm{sfc}}$", "W m$^{-2}$ K$^{-1}$", "YlOrRd", 0, 2.5),
    (axes[2,0], d["sw_grad_mu0"], "(e) $\\partial F_{\\mathrm{SW}}/\\partial \\mu_0$", "W m$^{-2}$", "YlOrRd", 0, 1500),
    (axes[2,1], d["sw_grad_T"], "(f) $\\partial F_{\\mathrm{SW}}/\\partial T_{\\mathrm{sfc}}$", "W m$^{-2}$ K$^{-1}$", "RdBu_r", None, None),
]

for ax, field, title, label, cmap, vmin, vmax in panels:
    field_ma = np.ma.array(field, mask=np.isnan(field))
    if vmin is not None:
        im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    else:
        mx = np.nanpercentile(np.abs(field), 97)
        im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, vmin=-mx, vmax=mx, shading="auto")
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=13)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(label, fontsize=9)

plt.suptitle("SW+LW Joint Feedback Kernel (Real ERA5 0.5° Grid, 00Z 1 Jan 2024)\n"
             "Both SW and LW gradients via automatic differentiation on GPU",
             fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("global_sw_lw_kernels.pdf", dpi=200, bbox_inches="tight")
plt.savefig("global_sw_lw_kernels.png", dpi=200, bbox_inches="tight")
print("Saved global_sw_lw_kernels")
