"""Figure 4: Global maps of SW radiation fields."""
import numpy as np, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# Look for global_sw.npz in (1) this directory, (2) ../../data/, (3) /tmp (legacy server)
_HERE = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_HERE, "global_sw.npz"),
    os.path.normpath(os.path.join(_HERE, "..", "..", "data", "global_sw.npz")),
    "/tmp/global_sw.npz",
]
_npz_path = next((p for p in _candidates if os.path.exists(p)), None)
if _npz_path is None:
    print("ERROR: global_sw.npz not found. See experiments/e02_era5_fluxes/gen_data.py "
          "or data/README.md to regenerate.", file=sys.stderr)
    sys.exit(1)
print("Loading %s" % _npz_path)
d = np.load(_npz_path)
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

plt.suptitle("Global Shortwave Radiation Maps (ERA5-driven, 00Z 1 Jan 2024)\n"
             "OpenIFS 48r1 PyTorch Port, $0.5^\\circ$ grid, 137 levels, clear sky",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("/tmp/fig4_global_sw_maps.pdf", dpi=200)
plt.savefig("/tmp/fig4_global_sw_maps.png", dpi=200)
print("Saved fig4_global_sw_maps")
