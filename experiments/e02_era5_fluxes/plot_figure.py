"""Figure 2: 9-panel vertical profiles (3x3 layout), 4 ERA5 regimes overlaid.

Each panel shows one radiation intermediate quantity. For each quantity,
4 ERA5 climate regimes (70N/45N/EQ/70S — same as Figure 1) are overlaid.
Within each regime, torch (solid line) and Fortran (x markers) overlap
perfectly (bit-exact).

The remaining 2 quantities (PRMU0, PTRCLR) are scalar / geometry-only and
are reported in the text rather than plotted.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

with open("vertical_profiles_multi.json") as f:
    d = json.load(f)

columns = d["columns"]
p_half = np.array(d["p_half_hpa"])   # nlev+1
p_full = np.array(d["p_full_hpa"])   # nlev

# 4 regions with colors matching Figure 1
region_colors = {"70N": "blue", "45N": "green", "EQ": "red", "70S": "purple"}
region_labels = {"70N": "70$^\\circ$N", "45N": "45$^\\circ$N",
                 "EQ": "Equator", "70S": "70$^\\circ$S"}

# 9 quantities for 3x3 grid (omit PRMU0 and PTRCLR — reported in text)
quantities = [
    ("PUD H$_2$O",        "pud_h2o",   "half", "kg m$^{-2}$"),
    ("PUD CO$_2$",        "pud_co2",    "half", "kg m$^{-2}$"),
    ("PCGAZ ($g$)",       "pcgaz",      "full", "asymmetry factor"),
    ("PPIZAZ ($\\omega$)","ppizaz",    "full", "single-scatter albedo"),
    ("PTAUZ ($\\tau$)",   "ptauz",      "full", "optical depth"),
    ("PRAY1 (direct)",    "pray1",      "half", "reflectivity"),
    ("PRAY2 (diffuse)",   "pray2",      "half", "reflectivity"),
    ("PTRA1 (direct)",    "ptra1",      "half", "transmissivity"),
    ("PTRA2 (diffuse)",   "ptra2",      "half", "transmissivity"),
]

n = len(quantities)
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
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

        # torch: solid line; fortran: x markers (overlapping)
        ax.plot(t_arr, p, "-", color=color, linewidth=2,
                label=region_labels[cname])
        ax.plot(f_arr, p, "x", color=color, markersize=3, alpha=0.5)

    ax.set_ylim(1020, 0.5)
    ax.set_title("(%s) %s" % (chr(97+i), label), fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    if i % 3 == 0:
        ax.set_ylabel("Pressure (hPa)", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Annotate max diff across all columns
    all_diffs = []
    for col in columns:
        diff = np.abs(np.array(col[t_key]) - np.array(col[f_key]))
        all_diffs.append(np.nanmax(diff))
    max_diff = max(all_diffs)
    if max_diff == 0:
        label_text = "bitwise identical"
    else:
        label_text = "max|$\\Delta$| = %.1e" % max_diff
    ax.text(0.02, 0.02, label_text,
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8))

    if i == 0:
        ax.legend(fontsize=9, loc="upper left")

plt.suptitle("Vertical Profiles: PyTorch Port vs Fortran Reference\n"
             "(4 ERA5 climate regimes × 9 radiation quantities, same columns as Figure 1)",
             fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fig2_vertical_comparison.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig2_vertical_comparison.png", dpi=200, bbox_inches="tight")
print("Saved fig2_vertical_comparison (4 ERA5 regimes × 9 quantities, 3x3)")
