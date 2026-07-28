"""Journal-clean global SW+LW feedback kernel maps (with coastlines).

Replaces plot_global_sw_lw.py output.  Changes:
  - NO suptitle.
  - Cartopy coastlines + gridlines on every panel.
  - Caption-accurate: states it is a subsampled grid.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from plot_style import colorbar_shrink

d = np.load(os.path.join(os.path.dirname(__file__), "global_sw_lw_full.npz"))
lats = d["lats"]; lons = d["lons"]
DATA_CRS = ccrs.PlateCarree()

fig, axes = plt.subplots(3, 2, figsize=(15, 16),
                         subplot_kw={"projection": ccrs.PlateCarree()},
                         constrained_layout=True)

panels = [
    (axes[0,0], d["sw_sfc"], False, "(a) SW Surface Downward Flux",
        "W m$^{-2}$", "YlOrRd", 0, 1200),
    (axes[0,1], d["lw_olr"], False, "(b) LW OLR (TOA Upward)",
        "W m$^{-2}$", "YlOrRd", 150, 350),
    (axes[1,0], d["sw_grad_q"], True, r"(c) $\partial F_{\mathrm{SW}}/\partial q_{\mathrm{sfc}}$",
        "W m$^{-2}$ (kg/kg)$^{-1}$", "RdBu_r", None, None),
    (axes[1,1], d["lw_grad_Ts"], False, r"(d) $\partial F_{\mathrm{OLR}}/\partial T_{\mathrm{sfc}}$",
        "W m$^{-2}$ K$^{-1}$", "YlOrRd", 0, 2.5),
    (axes[2,0], d["sw_grad_mu0"], False, r"(e) $\partial F_{\mathrm{SW}}/\partial \mu_0$",
        "W m$^{-2}$", "YlOrRd", 0, 1500),
    (axes[2,1], d["sw_grad_T"], True, r"(f) $\partial F_{\mathrm{SW}}/\partial T_{\mathrm{sfc}}$",
        "W m$^{-2}$ K$^{-1}$", "RdBu_r", None, None),
]

for ax, field, diverging, title, label, cmap, vmin, vmax in panels:
    field_ma = np.ma.array(field, mask=np.isnan(field))
    if diverging:
        mx = np.nanpercentile(np.abs(field), 97)
        norm = TwoSlopeNorm(vmin=-mx, vcenter=0, vmax=mx)
        im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, norm=norm,
                           shading="auto", transform=DATA_CRS)
    else:
        im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", transform=DATA_CRS)
    ax.coastlines(resolution="110m", linewidth=0.5, color="#444444")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#bbbbbb",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
    colorbar_shrink(im, ax, label, shrink=0.85, pad=0.02)
    ax.set_title(title, fontsize=10, loc="left")

out_dir = os.path.dirname(__file__)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, "global_sw_lw_kernels.%s" % ext))
plt.close(fig)
print("Saved global_sw_lw_kernels (with coastlines)")
