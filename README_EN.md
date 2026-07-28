# Differentiable OpenIFS Radiation

A bit-exact PyTorch port of the ECMWF OpenIFS 48r1 shortwave (6-band) and longwave (RRTM-G) radiation parameterization, supporting automatic differentiation for gradient-based sensitivity analysis and parameter calibration.

## Repository Structure

```
radi_paper/
├── AGU_JAMES_AGUTeX_Article/     # Paper LaTeX source
│   ├── agujournaltemplate.tex     # Main manuscript
│   ├── agujournal2019.cls         # AGU document class
│   └── agusample.bib             # Bibliography
├── radiation_solver/              # Radiation parameterization code
│   └── openifs_radiation/
│       ├── classic_sw/            # Shortwave 6-band scheme
│       └── rrtm_lw/               # Longwave RRTM-G scheme
├── experiments/                   # Experiment scripts and data
│   ├── e01_bitexact_verification/ # Per-kernel bit-exact verification
│   ├── e02_era5_fluxes/           # ERA5-driven fluxes and heating rates
│   ├── e03_gradient_sensitivity/  # Autograd gradients + SW+LW feedback kernels
│   ├── e04_parameter_calibration/ # Gradient-based parameter calibration
│   ├── fig4_global_gradients/     # Global SW+LW sensitivity maps + LW verification
│   ├── calibration/               # Calibration experiment data and plots
│   └── perf_benchmark/            # CPU/GPU performance benchmarks
├── paper_figure/                  # Paper figures (PDF + PNG)
├── presentation/                  # Undergraduate lecture notes (HTML)
└── data/                          # Large data file notes
```

## Key Features

- **Bit-exact**: All 12 SW kernels and the LW RRTM-G chain achieve ULP-level agreement with the Fortran reference (max |Δ| < 10⁻¹⁴)
- **Fully differentiable**: Both SW and LW chains support autograd backward pass; FD cross-check rel_err < 10⁻¹⁰
- **GPU accelerated**: Same code runs on NVIDIA A100; ~8,000 columns × 137 levels SW+LW in under 2 seconds

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended)
- NumPy, Matplotlib, netCDF4
- gfortran (for building the Fortran reference library)
- LaTeX (MiKTeX recommended, with AGU `agujournal2019.cls`)

## Data Files

ERA5 reanalysis data are publicly available from the Copernicus Climate Data Store. The original spectral GRIB files (~5.5 GB) are transformed to a 0.5° regular grid using CDO. Due to size, these are not included in the repository. Each experiment directory contains pre-computed `*.json` / `*.npz` result files for direct figure reproduction. Instructions for regenerating full-resolution data from ERA5 GRIB are in `e02_era5_fluxes/gen_data.py`.

## License

- OpenIFS source code: ECMWF OpenIFS License
- PyTorch port and experiment scripts: MIT License
- ERA5 data: Copernicus Climate Data Store (Hersbach et al. 2020)

## Contact

Yang Jinhui — National University of Defense Technology, Changsha, China
Corresponding author: Zhao Juan — zhaojuan@nudt.edu.cn
