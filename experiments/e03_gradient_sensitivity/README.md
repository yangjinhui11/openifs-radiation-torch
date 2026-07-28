# Experiment 3: Gradient Sensitivity (Paper §5–6)

Autograd gradient correctness verification (vs central finite differences)
and global 2-D radiation sensitivity fields.

## Files

- `gen_gradient_data.py` — single-column gradient test (∂F/∂T_k, ∂F/∂μ₀) + FD convergence
- `gradient_verification.json` — single-column results (convergence curves, vertical profiles)
- `gen_global_gradients.py` — global 2-D sensitivity fields (∂F/∂T_sfc, ∂F/∂q_sfc, ∂F/∂μ₀)
- `global_gradients_fd.json` — finite-difference cross-check at one column
- `plot_gradient.py` — Figure 3: single-column gradient verification (3 panels)
- `plot_global_gradients.py` — Figure 4: global 2-D sensitivity fields (4 panels)

## Running

```bash
python gen_gradient_data.py     # single-column gradient + FD convergence
python gen_global_gradients.py  # global 2-D fields (~7 min on A100)
python plot_gradient.py
python plot_global_gradients.py
```

NOTE: Both gen scripts use the autograd-compatible pure-Python computation
path (JIT kernels disabled), as documented in paper §5.

## Paper output

Figure 3 (gradient correctness) and Figure 4 (global sensitivity maps).
