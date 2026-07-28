"""Performance scaling plot (journal-clean).

Changes vs original (per reviewer feedback):
  - NO suptitle (context in LaTeX caption).
  - The compiled-GPU curve (which is DERIVED, not measured) is now visually
    distinct and explicitly labeled "(estimated)" in the legend, with an open
    marker and dotted line, so it cannot be mistaken for a direct measurement.
  - The measured eager curve is solid; the derived compiled curve is dotted.
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ncols = np.array([1, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536])

# Fortran CPU total time (ms) — measured
fortran_ms = np.array([1, 32, 145, 590, 3588, 10531, 14421, 31675, 61376])
# PyTorch GPU eager total time (ms) — measured
gpu_eager_ms = np.array([3298, 3423, 3421, 3442, 3426, 3442, 3433, 3547, 3645])
# Compiled-GPU curve DERIVED from the measured eager curve using the 4.4x
# speedup factor validated bit-exact at N=256 (NOT directly measured at every N)
compile_ratio = 4.4
gpu_compiled_ms = gpu_eager_ms / compile_ratio

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                     "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

# Distinct styles: measured = filled marker + solid; estimated = open marker + dotted
def style_fortran(ax, y, label):
    ax.loglog(ncols, y, "s-", color="#1f77b4", linewidth=1.8, markersize=6,
              markerfacecolor="#1f77b4", label=label)
def style_eager(ax, y, label):
    ax.loglog(ncols, y, "^--", color="#d62728", linewidth=1.4, markersize=6,
              markerfacecolor="#d62728", alpha=0.55, label=label)
def style_compiled(ax, y, label):
    # OPEN marker (markerfacecolor=white) + dotted line => visually "estimated"
    ax.loglog(ncols, y, "^:", color="#d62728", linewidth=1.8, markersize=7,
              markerfacecolor="white", markeredgecolor="#d62728",
              markeredgewidth=1.4, label=label)

# Panel (a): Total time
style_fortran(ax1, fortran_ms, "Fortran, single CPU thread (measured)")
style_eager(ax1, gpu_eager_ms, "Differentiable scheme, GPU eager (measured)")
style_compiled(ax1, gpu_compiled_ms,
               "Differentiable scheme, GPU compiled (estimated, see caption)")
ax1.loglog(ncols, 0.9*ncols, ":", color="#1f77b4", alpha=0.3, linewidth=1)
ax1.set_xlabel("Number of columns")
ax1.set_ylabel("Total runtime (ms)")
ax1.set_title("(a) Total runtime vs column count", loc="left")
ax1.legend(loc="upper left", framealpha=0.9, edgecolor="#cccccc")
ax1.grid(True, alpha=0.25, which="both", linewidth=0.5)
ax1.set_xlim(0.8, 100000); ax1.set_ylim(50, 200000)
ax1.axvline(8192, color="green", linestyle="--", alpha=0.4, linewidth=1)
ax1.text(8192*1.2, 200, "T42", fontsize=9, color="green")

# Panel (b): Per-column time
style_fortran(ax2, fortran_ms/ncols, "Fortran, single CPU thread (measured)")
style_eager(ax2, gpu_eager_ms/ncols, "Differentiable scheme, GPU eager (measured)")
style_compiled(ax2, gpu_compiled_ms/ncols,
               "Differentiable scheme, GPU compiled (estimated, see caption)")
ax2.set_xlabel("Number of columns")
ax2.set_ylabel("Runtime per column (ms)")
ax2.set_title("(b) Per-column cost vs column count", loc="left")
ax2.grid(True, alpha=0.25, which="both", linewidth=0.5)
ax2.set_xlim(0.8, 100000)
ax2.axvline(8192, color="green", linestyle="--", alpha=0.4, linewidth=1)
ax2.text(8192*1.2, 0.002, "T42", fontsize=9, color="green")

_HERE = os.path.dirname(os.path.abspath(__file__))
_out = os.path.join(_HERE, "perf_scaling.pdf")
fig.savefig(_out)
fig.savefig(os.path.join(_HERE, "perf_scaling.png"))
# also copy into paper_figure (two levels up)
import shutil
_pf = os.path.normpath(os.path.join(_HERE, "..", "..", "paper_figure"))
for ext in ("pdf", "png"):
    shutil.copy(os.path.join(_HERE, "perf_scaling.%s" % ext),
                os.path.join(_pf, "perf_scaling.%s" % ext))
print("Saved perf_scaling (estimated curve now open-marker + dotted)")
