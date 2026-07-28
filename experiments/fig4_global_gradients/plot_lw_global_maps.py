"""Regenerate lw_global_maps.pdf with corrected clear-sky LW data.

The clear-sky LW OLR and surface downward flux are taken from the same
ERA5 0.5-degree run that produced the cloudy-sky comparison, ensuring
internal consistency. Uses the patched rtrn1a.py (surface down now physical).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("cloudy_lw_global.npz")
lats = d["lats"]; lons = d["lons"]
lw_olr = d["lw_clr_olr"]      # clear-sky OLR
lw_sfc = d["lw_clr_sfc"]      # clear-sky surface downward

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

panels = [
    (axes[0], lw_olr, "(a) Outgoing Longwave Radiation (Clear Sky)",
        "W m$^{-2}$", "viridis", 140, 320),
    (axes[1], lw_sfc, "(b) Surface Downward Longwave Flux (Clear Sky)",
        "W m$^{-2}$", "magma", 80, 440),
]

for ax, field, title, label, cmap, vmin, vmax in panels:
    field_ma = np.ma.array(field, mask=np.isnan(field))
    im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(title, fontsize=13)
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(label, fontsize=10)

plt.suptitle("Clear-Sky Longwave Radiation Fields (Real ERA5 0.5$\\degree$, 00Z 1 Jan 2024, 137 levels)",
             fontsize=14, fontweight="bold", y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("lw_global_maps.pdf", dpi=200, bbox_inches="tight")
plt.savefig("lw_global_maps.png", dpi=200, bbox_inches="tight")
print("Saved lw_global_maps")
print("OLR  range: [%.1f, %.1f] mean=%.1f" % (np.nanmin(lw_olr), np.nanmax(lw_olr), np.nanmean(lw_olr)))
print("SFC  range: [%.1f, %.1f] mean=%.1f" % (np.nanmin(lw_sfc), np.nanmax(lw_sfc), np.nanmean(lw_sfc)))
