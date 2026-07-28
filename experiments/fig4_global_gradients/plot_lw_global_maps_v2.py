"""Journal-clean clear-sky LW global maps (with coastlines).

Replaces plot_lw_global_maps.py output.  Changes:
  - NO suptitle.
  - Cartopy coastlines + gridlines.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from plot_style import colorbar_shrink

d = np.load(os.path.join(os.path.dirname(__file__), "cloudy_lw_global.npz"))
lats = d["lats"]; lons = d["lons"]
lw_olr = d["lw_clr_olr"]
lw_sfc = d["lw_clr_sfc"]
DATA_CRS = ccrs.PlateCarree()

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                         subplot_kw={"projection": ccrs.PlateCarree()},
                         constrained_layout=True)

panels = [
    (axes[0], lw_olr, "(a) Outgoing Longwave Radiation (Clear Sky)",
        "W m$^{-2}$", "viridis", 140, 320),
    (axes[1], lw_sfc, "(b) Surface Downward Longwave Flux (Clear Sky)",
        "W m$^{-2}$", "magma", 80, 440),
]

for ax, field, title, label, cmap, vmin, vmax in panels:
    field_ma = np.ma.array(field, mask=np.isnan(field))
    im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto", transform=DATA_CRS)
    ax.coastlines(resolution="110m", linewidth=0.5, color="white")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#888888",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
    colorbar_shrink(im, ax, label, shrink=0.85, pad=0.02)
    ax.set_title(title, fontsize=11, loc="left")

out_dir = os.path.dirname(__file__)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, "lw_global_maps.%s" % ext))
plt.close(fig)
print("Saved lw_global_maps (with coastlines)")
print("OLR range: [%.1f, %.1f]" % (np.nanmin(lw_olr), np.nanmax(lw_olr)))
print("SFC range: [%.1f, %.1f]" % (np.nanmin(lw_sfc), np.nanmax(lw_sfc)))
