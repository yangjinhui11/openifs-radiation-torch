"""Figure 2: 9-panel vertical profiles (journal-clean).

Changes vs original:
  - NO suptitle (context in LaTeX caption).
  - max|Δ| annotation restyled: subtle white box with thin gray edge
    (was a loud yellow box), placed top-right so it does not collide with the
    legend on panel (a).
  - Single legend on panel (a), smaller fonts, consistent gridline style.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "vertical_profiles_multi.json")) as f:
    d = json.load(f)

columns = d["columns"]
p_half = np.array(d["p_half_hpa"])   # nlev+1
p_full = np.array(d["p_full_hpa"])   # nlev

region_colors = {"70N": "#1f77b4", "45N": "#2ca02c", "EQ": "#d62728", "70S": "#9467bd"}
region_labels = {"70N": r"70$^\circ$N", "45N": r"45$^\circ$N",
                 "EQ": "Equator", "70S": r"70$^\circ$S"}

quantities = [
    (r"PUD H$_2$O",        "pud_h2o",   "half", r"kg m$^{-2}$"),
    (r"PUD CO$_2$",        "pud_co2",    "half", r"kg m$^{-2}$"),
    (r"PCGAZ ($g$)",       "pcgaz",      "full", "asymmetry factor"),
    (r"PPIZAZ ($\omega$)", "ppizaz",    "full", "single-scatter albedo"),
    (r"PTAUZ ($\tau$)",    "ptauz",      "full", "optical depth"),
    (r"PRAY1 (direct)",    "pray1",      "half", "reflectivity"),
    (r"PRAY2 (diffuse)",   "pray2",      "half", "reflectivity"),
    (r"PTRA1 (direct)",    "ptra1",      "half", "transmissivity"),
    (r"PTRA2 (diffuse)",   "ptra2",      "half", "transmissivity"),
]

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                     "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})

fig, axes = plt.subplots(3, 3, figsize=(16, 16), constrained_layout=True)
axes_flat = axes.flatten()

for i, (label, key_base, ptype, xlabel) in enumerate(quantities):
    ax = axes_flat[i]
    t_key = key_base + "_torch"
    f_key = key_base + "_fortran"
    p = p_full if ptype == "full" else p_half

    for col in columns:
        cname = col["name"]
        color = region_colors[cname]
        t_arr = np.array(col[t_key])
        f_arr = np.array(col[f_key])
        ax.plot(t_arr, p, "-", color=color, linewidth=1.6,
                label=region_labels[cname] if i == 0 else None)
        ax.plot(f_arr, p, "x", color=color, markersize=3, alpha=0.5)

    ax.set_ylim(1020, 0.5)
    ax.set_title("(%s) %s" % (chr(97+i), label), fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=10)
    if i % 3 == 0:
        ax.set_ylabel("Pressure (hPa)", fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Subtle max|Δ| annotation: white box, thin gray edge, top-right corner
    all_diffs = []
    for col in columns:
        diff = np.abs(np.array(col[t_key]) - np.array(col[f_key]))
        all_diffs.append(np.nanmax(diff))
    max_diff = max(all_diffs)
    label_text = "bitwise identical" if max_diff == 0 else r"max$|\Delta|$ = %.1e" % max_diff
    ax.text(0.98, 0.97, label_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#999999", linewidth=0.6, alpha=0.9))

    if i == 0:
        ax.legend(fontsize=8.5, loc="lower left", frameon=True,
                  edgecolor="#cccccc", framealpha=0.9)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(_HERE, "fig2_vertical_comparison.%s" % ext))
plt.close(fig)
print("Saved fig2_vertical_comparison (journal-clean, subtle annotations)")
