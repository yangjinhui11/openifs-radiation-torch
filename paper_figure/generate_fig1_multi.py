"""Figure 1: Multi-region flux profiles (Arctic, Mid-lat, Equator, Antarctic).
No summary panel — 7 panels (a-g) in a 2x4 layout with last slot empty."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

with open("multi_region_fluxes.json") as f:
    d = json.load(f)

regions = d["regions"]
colors = {"70N": "blue", "45N": "green", "EQ": "red", "70S": "purple"}
labels = {"70N": "70$^\\circ$N (Arctic)", "45N": "45$^\\circ$N (Mid-lat)",
          "EQ": "Equator (0$^\\circ$)", "70S": "70$^\\circ$S (Antarctic)"}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# (a) Temperature
ax = axes[0, 0]
for r in regions:
    T = np.array(r["T"]); p = np.array(r["p_full_hpa"])
    ax.plot(T, p, color=colors[r["name"]], linewidth=2, label=labels[r["name"]])
ax.set_ylim(1020, 0.5); ax.set_xlabel("Temperature (K)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(a) Temperature", fontsize=13)
ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

# (b) SW downward flux
ax = axes[0, 1]
for r in regions:
    f = np.array(r["sw_down"]); p = np.array(r["p_half_hpa"])
    ax.plot(f, p, color=colors[r["name"]], linewidth=2, label=labels[r["name"]])
ax.set_ylim(1020, 0.5); ax.set_xlabel("SW Downward (W m$^{-2}$)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(b) SW Downward Flux", fontsize=13)
ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

# (c) LW upward flux
ax = axes[0, 2]
for r in regions:
    f = np.array(r["lw_up"]); p = np.array(r["p_half_hpa"])
    ax.plot(f, p, color=colors[r["name"]], linewidth=2, label=labels[r["name"]])
ax.set_ylim(1020, 0.5); ax.set_xlabel("LW Upward (W m$^{-2}$)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(c) LW Upward Flux (OLR)", fontsize=13)
ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

# (d) Specific humidity
ax = axes[0, 3]
for r in regions:
    q = np.array(r["Q"]) * 1000; p = np.array(r["p_full_hpa"])
    ax.plot(q, p, color=colors[r["name"]], linewidth=2, label=labels[r["name"]])
ax.set_ylim(1020, 0.5); ax.set_xlabel("q (g kg$^{-1}$)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(d) Specific Humidity", fontsize=13)
ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

# (e) SW heating rate
ax = axes[1, 0]
for r in regions:
    h = np.array(r["sw_heat"]); p = np.array(r["p_full_hpa"])
    ax.plot(h, p, color=colors[r["name"]], linewidth=2, label=labels[r["name"]])
ax.set_ylim(1020, 0.5); ax.set_xlabel("SW Heating (K day$^{-1}$)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(e) SW Heating Rate", fontsize=13)
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

# (f) LW heating rate
ax = axes[1, 1]
for r in regions:
    h = np.array(r["lw_heat"]); p = np.array(r["p_full_hpa"])
    ax.plot(h, p, color=colors[r["name"]], linewidth=2, label=labels[r["name"]])
ax.set_ylim(1020, 0.5); ax.set_xlabel("LW Heating (K day$^{-1}$)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(f) LW Heating Rate", fontsize=13)
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

# (g) Net heating rate
ax = axes[1, 2]
for r in regions:
    sw = np.array(r["sw_heat"]); lw = np.array(r["lw_heat"])
    p = np.array(r["p_full_hpa"])
    ax.plot(sw + lw, p, color=colors[r["name"]], linewidth=2, label=labels[r["name"]])
ax.set_ylim(1020, 0.5); ax.set_xlabel("Net Heating (K day$^{-1}$)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)
ax.set_title("(g) Net Radiative Heating", fontsize=13)
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.3)

# (h) — turn off, no summary
axes[1, 3].axis("off")

plt.suptitle("Radiation Flux Profiles Across Climate Regimes\n"
             "(ERA5 00Z 1 Jan 2024, 137 levels, clear sky, OpenIFS 48r1 PyTorch Port)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fig1_flux_profiles.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig1_flux_profiles.png", dpi=200, bbox_inches="tight")
print("Saved fig1_flux_profiles (4 regions, no summary)")
