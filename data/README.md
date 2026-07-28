# Data Files

This directory holds large preprocessed ERA5 data files used by the
experiments. The files are excluded from the repository via `.gitignore`
because of their size. To regenerate:

1. Obtain ECMWF ERA5 model-level analysis from the Copernicus Climate
   Data Store (00 UTC 1 January 2024):
   - `/data/era5/2024/202401.grib`

2. Run the preprocessing pipeline:
   ```bash
   cd ../experiments/e02_era5_fluxes
   python gen_data.py    # produces global_sw.npz, era5_real.npz
   ```

3. Move the outputs here, or update the paths referenced in each
   experiment's `gen_*.py`.

## Required files

| File | Size | Used by |
|---|---|---|
| `global_sw.npz` | ~1.7 GB | e02 (global flux maps), e03 (global gradients), e04 (calibration) |
| `era5_real.npz` | ~2 MB | e02 (subset fluxes), e03 (single-column gradient test) |

The `*.json` result files shipped with each experiment are the final
computed outputs, sufficient to reproduce the figures without rerunning
the forward model.
