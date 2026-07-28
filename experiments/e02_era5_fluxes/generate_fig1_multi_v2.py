"""Figure 1: Multi-region flux profiles (journal-clean).

Changes vs original:
  - NO suptitle (context in LaTeX caption).
  - Single shared legend in the empty 8th slot, not repeated on every panel.
  - Consistent axis-label units and formatting.
  - Tighter, smaller fonts; consistent gridline styling.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "multi_region_fluxes.json")) as f:
    d = json.load(f)

regions = d["regions"]
colors = {"70N": "#1f77b4", "45N": "#2ca02c", "EQ": "#d62728", "70S": "#9467bd"}
labels = {"70N": r"70$^\circ$N (Arctic)", "45N": r"45$^\circ$N (mid-lat)",
          "EQ": r"Equator (0$^\circ$)", "70S": r"70$^\circ$S (Antarctic)"}

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                     "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})

fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

def plot_profile(ax, key, half=False, ylabel="Pressure (hPa)", xlabel="",
                 title="", mul=1.0, zeroline=False):
    pkey = "p_half_hpa" if half else "p_full_hpa"
    for r in regions:
        y = np.array(r[key]) * mul
        p = np.array(r[pkey])
        ax.plot(y, p, color=colors[r["name"]], linewidth=1.8)
    ax.set_ylim(1020, 0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    if zeroline:
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)

plot_profile(axes[0,0], "T", xlabel="Temperature (K)", title="(a) Temperature")
plot_profile(axes[0,1], "sw_down", half=True, ylabel="",
             xlabel=r"SW downward (W m$^{-2}$)", title="(b) SW downward flux")
plot_profile(axes[0,2], "lw_up", half=True, ylabel="",
             xlabel=r"LW upward (W m$^{-2}$)", title="(c) LW upward flux (OLR)")
plot_profile(axes[0,3], "Q", ylabel="", xlabel="q (g kg$^{-1}$)",
             title="(d) Specific humidity", mul=1000.0)

plot_profile(axes[1,0], "sw_heat", ylabel="Pressure (hPa)",
             xlabel=r"SW heating (K day$^{-1}$)", title="(e) SW heating rate", zeroline=True)
plot_profile(axes[1,1], "lw_heat", ylabel="",
             xlabel=r"LW heating (K day$^{-1}$)", title="(f) LW heating rate", zeroline=True)
# (g) net = sw + lw
ax = axes[1,2]
for r in regions:
    sw = np.array(r["sw_heat"]); lw = np.array(r["lw_heat"]); p = np.array(r["p_full_hpa"])
    ax.plot(sw + lw, p, color=colors[r["name"]], linewidth=1.8)
ax.set_ylim(1020, 0.5); ax.set_xlabel(r"Net heating (K day$^{-1}$)", fontsize=10)
ax.set_ylabel("", fontsize=10)
ax.set_title("(g) Net radiative heating", fontsize=11, loc="left")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
ax.invert_yaxis(); ax.grid(True, alpha=0.25, linewidth=0.5); ax.tick_params(labelsize=9)

# (h) shared legend in the empty slot
ax_leg = axes[1,3]; ax_leg.axis("off")
handles = [plt.Line2D([0],[0], color=colors[k], linewidth=2.2) for k in ["70N","45N","EQ","70S"]]
ax_leg.legend(handles, [labels[k] for k in ["70N","45N","EQ","70S"]],
              loc="center", fontsize=11, frameon=False, title="Climate regime",
              title_fontsize=11)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(_HERE, "fig1_flux_profiles.%s" % ext))
plt.close(fig)
print("Saved fig1_flux_profiles (journal-clean, shared legend)")
