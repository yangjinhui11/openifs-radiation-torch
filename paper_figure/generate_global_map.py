"""Figure 4: Global maps of SW radiation fields."""
import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

d = np.load("global_sw.npz")
nlat = int(d["nlat"]); nlon = int(d["nlon"]); nlev = int(d["nlev"])
lats = d["lats"]; lons = d["lons"]
fd_sw = d["fd_sw"]; fu_sw = d["fu_sw"]; mu0 = d["mu0"]

def to_2d(flat):
    return flat.reshape(nlat, nlon)

sw_sfc = to_2d(fd_sw[:, -1])
sw_toa_up = to_2d(fu_sw[:, 0])
sw_net_toa = to_2d(fd_sw[:, 0] - fu_sw[:, 0])
mu0_2d = to_2d(mu0)
# Planetary albedo (only where mu0 > 0)
albedo = np.where(mu0_2d > 0.01, sw_toa_up / (1361 * mu0_2d + 1e-10), np.nan)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))

panels = [
    (axes[0,0], sw_sfc, "(a) SW Surface Downward", "W m$^{-2}$", "YlOrRd", 0, 1200),
    (axes[0,1], sw_toa_up, "(b) SW TOA Upward (Reflected)", "W m$^{-2}$", "YlOrRd", 0, 400),
    (axes[1,0], albedo, "(c) Planetary Albedo", "", "RdYlBu_r", 0, 0.6),
    (axes[1,1], mu0_2d, "(d) Cosine Solar Zenith ($\\mu_0$)", "", "YlGn", 0, 1.0),
]

for ax, field, title, label, cmap, vmin, vmax in panels:
    if title.startswith("(c)"):
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    else:
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=12)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    if label:
        cb.set_label(label, fontsize=9)

plt.suptitle("Shortwave Radiation", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("fig4_global_sw_maps.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig4_global_sw_maps.png", dpi=200, bbox_inches="tight")
print("Saved fig4_global_sw_maps")
