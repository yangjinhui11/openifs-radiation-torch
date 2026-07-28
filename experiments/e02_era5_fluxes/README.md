# Experiment 2: ERA5-Driven Fluxes (Paper §4)

ERA5 reanalysis-driven shortwave fluxes and heating rates, demonstrating
physical consistency across four climate regimes (Arctic, mid-latitude,
equator, Antarctic) and on the global 0.5° grid.

## Files

- `gen_data.py` — preprocessing: ERA5 GRIB → npz (see data/README.md)
- `vertical_profiles_multi.json` — 4-column vertical profiles (T, q, fluxes, heating)
- `generate_fig1_multi.py` — Figure 1: 4-regime flux profiles (7 panels)
- `generate_global_map.py` — Figure (global SW map, 4 panels)
- `plot_figure.py` — Figure 2: vertical per-kernel comparison

## Running

```bash
# Generate ERA5 npz from GRIB (requires ECMWF MIR or eccodes)
python gen_data.py

# Reproduce Figure 1
python generate_fig1_multi.py

# Reproduce Figure 2 (per-kernel vertical comparison)
python plot_figure.py
```

## Inputs

- `/tmp/global_sw.npz` (or `data/global_sw.npz`) — global ERA5 grid
- `/tmp/era5_real.npz` — 256-column subset

## Paper output

Figure 1 (4-regime fluxes), Figure 2 (per-kernel comparison),
global SW map, and Table 3 (global 16-column verification).
