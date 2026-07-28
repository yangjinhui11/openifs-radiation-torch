"""Generate LW verification table from existing data."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load 4-region results
with open("sw_lw_joint_4region.json") as f:
    d4 = json.load(f)

# Load 20-level FD verification result (from lw_test.py)
# OLR=234.80, FD check rel_err=3.0e-10
fd_data = {"profile": "MidLat Summer (20-level)", "OLR": 234.80, "FD_rel_err": 3.0e-10}

regions = ["70N", "45N", "EQ", "70S"]
labels = ["70°N (Arctic)", "45°N (MidLat)", "Equator", "70°S (Antarctic)"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) LW OLR across regions + Planck theory line
ax = axes[0]
Tsfc = [d4[r]["T_sfc"] for r in regions]
olr = [d4[r]["lw_olr"] for r in regions]
dTs = [d4[r]["lw_dTs"] for r in regions]
ax.scatter(Tsfc, olr, c=["blue","green","red","purple"], s=200, zorder=5)
for i,r in enumerate(regions):
    ax.annotate(labels[i], (Tsfc[i]+1, olr[i]+5), fontsize=10)
# Planck curve: sigma*T^4 (blackbody at surface)
T_arr = np.linspace(265, 305, 50)
planck = 5.67e-8 * T_arr**4
ax.plot(T_arr, planck*0.5, "k--", alpha=0.4, linewidth=1, label="$\\sigma T^4 \\times 0.5$ (reference)")
ax.set_xlabel("Surface Temperature (K)", fontsize=13)
ax.set_ylabel("OLR (W m$^{-2}$)", fontsize=13)
ax.set_title("(a) LW OLR vs Surface Temperature", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) dOLR/dTs vs T_sfc + SB theory
ax = axes[1]
ax.scatter(Tsfc, dTs, c=["blue","green","red","purple"], s=200, zorder=5)
for i,r in enumerate(regions):
    ax.annotate(labels[i], (Tsfc[i]+1, dTs[i]+0.05), fontsize=10)
sb = 4 * 5.67e-8 * T_arr**3
ax.plot(T_arr, sb, "k--", alpha=0.4, linewidth=1, label="$4\\sigma T^3$ (SB theory)")
# FD verification point
ax.scatter([294], [4.14], marker="*", c="gold", s=400, zorder=6, edgecolors="black",
           label="20-level FD check (rel. err. $3\\times10^{-10}$)")
ax.set_xlabel("Surface Temperature (K)", fontsize=13)
ax.set_ylabel("$\\partial F_{\\mathrm{OLR}}/\\partial T_{\\mathrm{sfc}}$ (W m$^{-2}$ K$^{-1}$)", fontsize=13)
ax.set_title("(b) LW Planck Sensitivity (surface-only)", fontsize=14)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 7)

plt.suptitle("LW RRTM-G Verification: OLR and Surface-Planck Sensitivity\n"
             "4 ERA5 columns (137 levels) + 20-level AFGL FD cross-check",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("lw_verification.pdf", dpi=200, bbox_inches="tight")
plt.savefig("lw_verification.png", dpi=200, bbox_inches="tight")
print("Saved lw_verification")

# Print summary table
print("\nLW Verification Summary:")
print(f"{'Profile':<25} {'T_sfc':>6} {'OLR':>8} {'dOLR/dTs':>10} {'FD check':>12}")
print("-"*65)
for i,r in enumerate(regions):
    print(f"{labels[i]:<25} {d4[r]['T_sfc']:>6.1f} {d4[r]['lw_olr']:>8.1f} {d4[r]['lw_dTs']:>10.4f} {'—':>12}")
print(f"{'MidLat Summer (20-lev)':<25} {'294':>6} {fd_data['OLR']:>8.1f} {'4.14':>10} {fd_data['FD_rel_err']:>12.1e}")
