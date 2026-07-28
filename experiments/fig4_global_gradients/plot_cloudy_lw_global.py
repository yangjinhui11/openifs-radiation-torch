"""Global clear-sky vs cloudy-sky SW+LW flux maps from real ERA5 data.

Reads cloudy_lw_global.npz produced by gen_cloudy_global.py and produces:
  - Panel (a,b): SW surface downward, clear vs cloudy (cloud radiative effect)
  - Panel (c,d): LW OLR, clear vs cloudy
  - Panel (e,f): LW surface downward, clear vs cloudy

Plus a separate 3-panel cloud radiative effect (CRE) summary:
  - SW CRE = SW_cld - SW_clr  (negative: clouds reflect SW)
  - LW CRE = LW_cld_olr - LW_clr_olr  (negative: clouds reduce OLR)
  - Net CRE = SW_CRE + LW_CRE
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

d = np.load("cloudy_lw_global.npz")
lats = d["lats"]; lons = d["lons"]

# --- Six-panel clear/cloudy comparison ---
fig, axes = plt.subplots(3, 2, figsize=(20, 18))

sw_clr = d["sw_clear"]; sw_cld = d["sw_cloudy"]
lw_clr_olr = d["lw_clr_olr"]; lw_cld_olr = d["lw_cld_olr"]
lw_clr_sfc = d["lw_clr_sfc"]; lw_cld_sfc = d["lw_cld_sfc"]

panels = [
    (axes[0,0], sw_clr,  "(a) SW Surface Downward — Clear Sky", "W m$^{-2}$", "YlOrRd", 0, 1200),
    (axes[0,1], sw_cld,  "(b) SW Surface Downward — Cloudy Sky", "W m$^{-2}$", "YlOrRd", 0, 1200),
    (axes[1,0], lw_clr_olr, "(c) LW OLR — Clear Sky", "W m$^{-2}$", "viridis", 180, 320),
    (axes[1,1], lw_cld_olr, "(d) LW OLR — Cloudy Sky", "W m$^{-2}$", "viridis", 180, 320),
    (axes[2,0], lw_clr_sfc, "(e) LW Surface Down — Clear Sky", "W m$^{-2}$", "magma", 150, 480),
    (axes[2,1], lw_cld_sfc, "(f) LW Surface Down — Cloudy Sky", "W m$^{-2}$", "magma", 150, 480),
]

for ax, field, title, label, cmap, vmin, vmax in panels:
    field_ma = np.ma.array(field, mask=np.isnan(field))
    im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=13)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(label, fontsize=9)

plt.suptitle("Clear vs Cloudy Sky Flux Comparison (Real ERA5 0.5$\\degree$, 00Z 1 Jan 2024)\n"
             "SW surface downward, LW OLR, LW surface downward",
             fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("cloudy_vs_clear_global.pdf", dpi=200, bbox_inches="tight")
plt.savefig("cloudy_vs_clear_global.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved cloudy_vs_clear_global")

# --- Four-panel Cloud Radiative Effect (CRE) summary ---
# CRE = cloudy - clear flux. SW CRE is meaningful only on the daylit
# hemisphere (night => no solar flux => CRE = 0 by definition); mask those
# cells as "no data" (shown in light gray) so the white zero of the divergent
# colormap is reserved for genuine near-zero CRE on the daylit side.
sw_cre = sw_cld - sw_clr                   # clouds reflect SW -> negative
lw_cre_olr = lw_cld_olr - lw_clr_olr       # clouds reduce OLR -> negative
lw_cre_sfc = lw_cld_sfc - lw_clr_sfc       # clouds enhance sfc down -> positive
net_cre = sw_cre + lw_cre_olr              # net TOA CRE

daylit = sw_clr > 1.0                      # daylit mask (W/m^2 threshold)

# Global + daylit-only means for the titles
sw_cre_day_mean = np.nanmean(np.where(daylit, sw_cre, np.nan))
net_cre_day_mean = np.nanmean(np.where(daylit, net_cre, np.nan))

fig2, axes2 = plt.subplots(2, 2, figsize=(20, 13))

# (field, title, daylit-only?, sym-percentile)
cre_panels = [
    (axes2[0,0], sw_cre,     "(a) SW CRE at Surface",
        True,  "W m$^{-2}$"),
    (axes2[0,1], lw_cre_olr, "(b) LW CRE at TOA",
        False, "W m$^{-2}$"),
    (axes2[1,0], lw_cre_sfc, "(c) LW CRE at Surface",
        False, "W m$^{-2}$"),
    (axes2[1,1], net_cre,    "(d) Net TOA CRE = SW + LW",
        True,  "W m$^{-2}$"),
]

for ax, field, title, daylit_only, label in cre_panels:
    # Mask: NaN stays masked; on SW-dependent panels also mask the night side
    # (shown in gray) so the white zero of the divergent colormap is reserved
    # for genuine near-zero CRE on the daylit side. The reported mean is the
    # GLOBAL mean (night cells contribute CRE=0), matching the convention used
    # in climate-model CRE diagnostics and published estimates.
    if daylit_only:
        mask = np.isnan(field) | (~daylit)
        mean_val = np.nanmean(field)        # global mean (night = 0 included)
    else:
        mask = np.isnan(field)
        mean_val = np.nanmean(field)
    sub = "(global mean = %+.0f)" % mean_val
    field_ma = np.ma.array(field, mask=mask)

    # Symmetric range from the 98th percentile of |CRE| on valid cells
    valid_abs = np.abs(np.ma.compressed(field_ma))
    mx = np.percentile(valid_abs, 98) if valid_abs.size > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-mx, vcenter=0, vmax=mx)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#d9d9d9")          # light gray for masked (night/missing)
    im = ax.pcolormesh(lons, lats, field_ma, cmap=cmap, norm=norm, shading="auto")
    cb = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02, extend="both")
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("%s  %s" % (title, sub), fontsize=12.5)

plt.suptitle("Cloud Radiative Effect (CRE = Cloudy $-$ Clear) from the Differentiable Scheme\n"
             "Real ERA5 cloud fields (cc, clwc, ciwc) on the 0.5$\\degree$ grid, 00Z 1 Jan 2024, 137 levels\n"
             "Gray = night hemisphere (SW CRE undefined; no solar flux)",
             fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("cloud_radiative_effect_global.pdf", dpi=200, bbox_inches="tight")
plt.savefig("cloud_radiative_effect_global.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved cloud_radiative_effect_global")

# --- Print summary statistics ---
print("\n=== Global flux statistics ===")
for name, arr in [("SW clr sfc", sw_clr), ("SW cld sfc", sw_cld),
                  ("LW clr OLR", lw_clr_olr), ("LW cld OLR", lw_cld_olr),
                  ("LW clr sfc", lw_clr_sfc), ("LW cld sfc", lw_cld_sfc)]:
    print("  %-12s mean=%7.2f  min=%7.2f  max=%7.2f" %
          (name, np.nanmean(arr), np.nanmin(arr), np.nanmax(arr)))
print("\n=== Cloud Radiative Effect (W/m^2) ===")
for name, arr in [("SW CRE sfc", sw_cre), ("LW CRE TOA", lw_cre_olr),
                  ("LW CRE sfc", lw_cre_sfc), ("Net TOA CRE", net_cre)]:
    print("  %-12s mean=%7.2f  min=%7.2f  max=%7.2f" %
          (name, np.nanmean(arr), np.nanmin(arr), np.nanmax(arr)))
