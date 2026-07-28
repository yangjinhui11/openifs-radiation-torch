"""Shared plotting style for all paper figures.

Design rules (per reviewer feedback):
  - NO suptitle / figure-level title. Context belongs in the LaTeX caption.
  - Panel titles only, left-aligned, consistent font size.
  - Global maps use Cartopy PlateCarree with coastlines + gridlines.
  - Consistent fonts, margins, and colorbar style across all figures.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Global rcParams (journal-clean) ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Default geometry string for global maps
GEO_DATA_NOTE = "ERA5, 00Z 1 Jan 2024, 137 levels"


def setup_geo_axes(ax, draw_coastlines=True, gridline_step=60):
    """Turn a standard Axes into a Cartopy PlateCarree axis with coastlines.

    Must be called with a GeoAxes created via projection=ccrs.PlateCarree().
    Returns the axis for chaining.  Falls back gracefully (no coastlines) if
    cartopy is unavailable.
    """
    try:
        import cartopy.feature as cfeature
        import cartopy.crs as ccrs
    except ImportError:
        return ax
    if draw_coastlines:
        ax.coastlines(resolution="110m", linewidth=0.5, color="#444444")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#bbbbbb",
                      alpha=0.5, linestyle="--",
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = matplotlib.ticker.MultipleLocator(gridline_step)
    gl.ylocator = matplotlib.ticker.MultipleLocator(gridline_step)
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    return ax


def make_global_axes(nrows, ncols, figsize, extent=None):
    """Create a grid of Cartopy PlateCarree subaxes for global maps.

    extent: optional (lon_min, lon_max, lat_min, lat_max) to set.
    Returns (fig, axes_flat_list).
    """
    try:
        import cartopy.crs as ccrs
        projection = ccrs.PlateCarree(central_longitude=0)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            subplot_kw={"projection": projection},
            constrained_layout=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()
        for ax in axes_flat:
            setup_geo_axes(ax)
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
        return fig, axes_flat
    except ImportError:
        # Fallback: plain axes (no coastlines)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 constrained_layout=True)
        return fig, np.atleast_1d(axes).ravel()


def colorbar_shrink(im, ax, label, shrink=0.85, pad=0.02):
    """Consistent colorbar styling."""
    cb = plt.colorbar(im, ax=ax, shrink=shrink, pad=pad, extend="both")
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb
