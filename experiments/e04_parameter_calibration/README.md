# Experiment 4: Parameter Calibration (Paper §7)

Gradient-based calibration of physical parameters via autograd + L-BFGS,
demonstrating that the differentiable scheme can solve inverse problems
the Fortran version cannot (without a hand-coded adjoint).

## Files

- `gen_calibration.py` — 1-D albedo recovery (controlled recovery)
- `gen_calibration2d.py` — 2-D (albedo, cloud-fraction) joint recovery
- `calibration_results.json` — 1-D results (loss history, recovered α)
- `calibration2d_results.json` — 2-D results (parameter trajectory)
- `flux_residual.json` — before/after flux residuals (RMSE collapse)
- `plot_calibration_combined.py` — Figure 5: 1-D + 2-D + cost scaling (3 panels)
- `plot_flux_residual_only.py` — Figure 6: per-column flux residual collapse

## Running

```bash
python gen_calibration.py     # 1-D (~10 min on A100)
python gen_calibration2d.py   # 2-D (~15 min on A100)
python plot_calibration_combined.py
python plot_flux_residual_only.py
```

## Key results

- 1-D albedo: recovered α=0.299999 (truth 0.30), loss drops 10 orders of magnitude
- 2-D (α, c_f): recovered (0.25000, 0.60000) (truth 0.25, 0.60), rel err 2.5e-7
- Flux RMSE: 198.9 → 1.8e-5 W/m² (improvement 1.1e7)

## Paper output

Figure 5 (calibration process + scaling) and Figure 6 (flux-fit quality).
