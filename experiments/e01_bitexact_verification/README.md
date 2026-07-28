# Experiment 1: Bit-Exact Verification (Paper §3)

Per-kernel numerical verification of the PyTorch port against a Fortran
reference library compiled from the unmodified OpenIFS 48r1 source.

## Important: Fortran reference dependency

These tests compare PyTorch outputs against `libsw_ref.so`, a shared library
compiled from the verbatim OpenIFS Fortran source (phys_radi/*.F90) with
BIND(C) wrappers. **The compiled library is NOT included in this repository**
because it requires the OpenIFS source (ECMWF OpenIFS license).

To build it:
1. Obtain the OpenIFS 48r1 source under the ECMWF OpenIFS license.
2. Compile the radiation kernels (swu.F90, swclr.F90, swr.F90, swni.F90,
   swde.F90, swtt.F90, swuvo3.F90, sw1s.F90) with BIND(C) wrappers.
3. Place the resulting `libsw_ref.so` and a Python `sw_ref.py` wrapper in
   `radiation_solver/offline_ref/` (create the directory).

If the library is absent, the tests **auto-skip** with a clear message
(they do not fail) — the PyTorch forward fluxes themselves do not need it.

## Files

- `test_swu_ref.py`        — SWU absorber amounts (PUD H2O/CO2)
- `test_swclr_ref.py`      — SWCLR clear-sky adding matrix
- `test_swni_ref.py`       — SWNI near-IR solver
- `test_swr_ref.py`        — SWR cloudy adding matrix
- `test_swuvo3_ref.py`     — ozone UV transmission
- `test_sw1s_ref.py`       — SW1S UV/visible band solver
- `test_sw_solver.py`      — end-to-end SW solver (no Fortran ref needed)
- `validate_t95_vs_fortran.py` — T95 configuration end-to-end validation

## Running

```bash
python test_sw_solver.py        # always runs (PyTorch-only)
pytest test_swu_ref.py          # skips if libsw_ref.so absent
```

## Paper output

Table 1 (per-stage agreement) is derived from these test outputs when the
Fortran reference library is available.
