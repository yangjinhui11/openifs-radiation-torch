"""Cloud Radiative Effect (CRE) global maps — journal-clean version.

Changes vs previous version (per reviewer feedback):
  - NO suptitle. All context moved to the LaTeX caption.
  - Cartopy PlateCarree coastlines + gridlines on every panel.
  - Night hemisphere shown in light gray, with the convention stated in caption.
  - TOA-CRE sign convention stated explicitly in each panel title.
  - Global mean reported in each panel title (night contributes CRE=0).

Sign conventions used here:
  SW CRE (surface)   = F_sw,cld − F_sw,clr   (negative: clouds cool the surface)
  LW CRE (TOA, as dF_up) = F_OLR,cld − F_OLR,clr   (negative: clouds reduce OLR)
  LW CRE (TOA, as net down) = −(F_OLR,cld − F_OLR,clr)  (positive: clouds warm)
  We report the OLR-difference form and label it "ΔOLR" to avoid ambiguity.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from plot_style import make_global_axes, colorbar_shrink, GEO_DATA_NOTE
try:
    import cartopy.crs as ccrs
    DATA_CRS = ccrs.PlateCarree()   # transform for pcolormesh data
except ImportError:
    DATA_CRS = None

d = np.load(os.path.join(os.path.dirname(__file__), "cloudy_lw_global.npz"))
lats = d["lats"]; lons = d["lons"]

sw_clr = d["sw_clear"]; sw_cld = d["sw_cloudy"]
lw_clr_olr = d["lw_clr_olr"]; lw_cld_olr = d["lw_cld_olr"]
lw_clr_sfc = d["lw_clr_sfc"]; lw_cld_sfc = d["lw_cld_sfc"]

# CRE definitions (all as cloudy - clear)
sw_cre_sfc  = sw_cld - sw_clr                       # <0 : clouds reflect SW
dOLR_cld    = lw_cld_olr - lw_clr_olr               # <0 : clouds reduce OLR  (= LW CRE in OLR form)
lw_cre_sfc  = lw_cld_sfc - lw_clr_sfc               # >0 : clouds warm surface
net_cre_toa = sw_cre_sfc + dOLR_cld                 # net at TOA (using OLR-form LW CRE)

daylit = sw_clr > 1.0   # daylit mask

# Subsampling note: this run uses 2700 columns (60 lat × 45 lon, ~6°×8° spacing)
# subsampled from the full ERA5 0.5° grid.  Stated in the caption.
nlat_g, nlon_g = len(lats), len(lons)

fig, axes = make_global_axes(2, 2, figsize=(16, 9))

panels = [
    # (ax, field, daylit_only, title, cmap)
    (axes[0], sw_cre_sfc,  True,
     r"(a) SW CRE$_\mathrm{sfc}$ = $F^\downarrow_\mathrm{cld}-F^\downarrow_\mathrm{clr}$"
     "\n(global mean %+.0f W m$^{-2}$)" % np.nanmean(sw_cre_sfc),
     "RdBu_r"),
    (axes[1], dOLR_cld,    False,
     r"(b) LW CRE$_\mathrm{TOA}$ = $\Delta\mathrm{OLR}$ = OLR$_\mathrm{cld}-$OLR$_\mathrm{clr}$"
     "\n(global mean %+.0f W m$^{-2}$)" % np.nanmean(dOLR_cld),
     "RdBu_r"),
    (axes[2], lw_cre_sfc,  False,
     r"(c) LW CRE$_\mathrm{sfc}$ = $LW^\downarrow_\mathrm{cld}-LW^\downarrow_\mathrm{clr}$"
     "\n(global mean %+.0f W m$^{-2}$)" % np.nanmean(lw_cre_sfc),
     "RdBu_r"),
    (axes[3], net_cre_toa, True,
     r"(d) Net CRE$_\mathrm{TOA}$ = (a)+(b)"
     "\n(global mean %+.0f W m$^{-2}$)" % np.nanmean(net_cre_toa),
     "RdBu_r"),
]

for ax, field, daylit_only, title, cmap in panels:
    if daylit_only:
        mask = np.isnan(field) | (~daylit)
    else:
        mask = np.isnan(field)
    field_ma = np.ma.array(field, mask=mask)

    valid_abs = np.abs(np.ma.compressed(field_ma))
    mx = np.percentile(valid_abs, 98) if valid_abs.size > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-mx, vcenter=0, vmax=mx)

    cm = plt.get_cmap(cmap).copy()
    # Cartopy requires the 'bad' color to be transparent for pcolormesh to wrap
    # across the date line, so we draw the night mask as a separate gray layer
    # underneath instead of relying on set_bad.
    cm.set_bad(alpha=0)                   # fully transparent bad (for wrapping)
    im = ax.pcolormesh(lons, lats, field_ma, cmap=cm, norm=norm,
                       shading="auto", transform=DATA_CRS)
    # Draw the night/mask region in light gray on top (zorder below the data
    # is not possible with pcolormesh, so we fill masked cells explicitly).
    if mask.any():
        night_ma = np.ma.array(mask, mask=~mask)
        ax.pcolormesh(lons, lats, night_ma.astype(float),
                      cmap=matplotlib.colors.ListedColormap(["#d9d9d9"]),
                      vmin=0, vmax=1, shading="auto",
                      transform=DATA_CRS, alpha=0.85)
    colorbar_shrink(im, ax, "W m$^{-2}$", shrink=0.85, pad=0.02)
    ax.set_title(title, fontsize=9.5, loc="left")

out_dir = os.path.dirname(__file__)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, "cloud_radiative_effect_global.%s" % ext))
plt.close(fig)
print("Saved cloud_radiative_effect_global (journal-clean)")

# Summary print for verification
print("\n=== CRE statistics (global mean, night=0 included) ===")
for name, arr in [("SW CRE sfc", sw_cre_sfc), ("LW CRE TOA (dOLR)", dOLR_cld),
                  ("LW CRE sfc", lw_cre_sfc), ("Net CRE TOA", net_cre_toa)]:
    print("  %-18s mean=%+7.1f  min=%+8.1f  max=%+8.1f" %
          (name, np.nanmean(arr), np.nanmin(arr), np.nanmax(arr)))
print("\nGrid: %d lat x %d lon (subsampled from 0.5deg ERA5)" % (nlat_g, nlon_g))
print("Daylit fraction: %.1f%%" % (100*daylit.mean()))
