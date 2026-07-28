#!/usr/bin/env python3
"""Extract RRTM reference tables from surrtrf.F90 into .npy files.

The production reference-profile tables live as hard-coded array assignments in
``openifs-48r1/ifs-source/arpifs/phys_radi/surrtrf.F90``::

    PREF( :) = (/ ... /)         ! 59 reference pressures (hPa)
    PREFLOG( :) = (/ ... /)      ! log(PREF)
    TREF( :) = (/ ... /)         ! 59 reference temperatures (K)
    CHI_MLS(i, j:k) = (/ ... /)  ! reference volume mixing ratios

This script parses the source directly and writes the arrays as float64 .npy
files under ``openifs_radiation/rrtm_lw/tables/``. Generating from source is the
only way to guarantee the torch port uses the same constants as the Fortran
reference.

Run from anywhere; paths default to the openifs workspace layout.

    python3 _extract_setcoef_tables.py                 # write .npy, print summary
    python3 _extract_setcoef_tables.py --check         # only verify against current .npy
    SURRTRF=/path/surrtrf.F90 python3 ...              # override source path
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DEFAULT_SURRTRF = Path(
    "/home/qixiang/yangjinhui/openifs/openifs-48r1/ifs-source/arpifs/phys_radi/surrtrf.F90"
)

# A single Fortran real literal, e.g. "1.05363E+03_JPRB" or "0.2090_JPRB".
_VAL_RE = re.compile(r"[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?")

# 1-D full array: PREF( :) = (/ ... /)
_1D_RE = re.compile(
    r"^\s*(?P<name>PREF|PREFLOG|TREF)\s*\(\s*:\s*\)\s*=\s*\(\s*/\s*(?P<vals>.*?)\s*/\s*\)",
    re.MULTILINE | re.DOTALL,
)

# 2-D partial array: CHI_MLS(i, start:end) = (/ ... /)
_2D_RE = re.compile(
    r"^\s*CHI_MLS\s*\(\s*(?P<spec>\d+)\s*,\s*(?P<start>\d+)\s*:\s*(?P<end>\d+)\s*\)"
    r"\s*=\s*\(\s*/\s*(?P<vals>.*?)\s*/\s*\)",
    re.MULTILINE | re.DOTALL,
)


def _parse_vals(vals_text: str) -> list[float]:
    out: list[float] = []
    for tok in vals_text.split(","):
        m = _VAL_RE.search(tok.strip())
        if not m:
            continue
        s = m.group(0).replace("D", "E").replace("d", "e")
        out.append(float(s))
    return out


def extract(surrtrf_path: Path) -> dict[str, np.ndarray]:
    """Parse surrtrf.F90 and return {table_name: ndarray} (float64)."""
    if not surrtrf_path.exists():
        raise FileNotFoundError(f"surrtrf.F90 not found at {surrtrf_path}")
    text = surrtrf_path.read_text(errors="ignore")

    out: dict[str, np.ndarray] = {}

    # 1-D tables
    for m in _1D_RE.finditer(text):
        name = m.group("name")
        vals = _parse_vals(m.group("vals"))
        expected = 59
        if len(vals) != expected:
            raise ValueError(f"{name}: parsed {len(vals)} values, expected {expected}")
        out[name] = np.array(vals, dtype=np.float64)

    # 2-D CHI_MLS
    chi: dict[int, dict[int, float]] = {i: {} for i in range(1, 8)}
    for m in _2D_RE.finditer(text):
        spec = int(m.group("spec"))
        start = int(m.group("start"))
        end = int(m.group("end"))
        vals = _parse_vals(m.group("vals"))
        if len(vals) != end - start + 1:
            raise ValueError(
                f"CHI_MLS({spec},{start}:{end}): parsed {len(vals)} values, "
                f"expected {end - start + 1}"
            )
        for idx, v in enumerate(vals, start=start):
            chi[spec][idx] = v

    # Verify CHI_MLS shape
    chi_arr = np.zeros((7, 59), dtype=np.float64)
    for spec in range(1, 8):
        for idx in range(1, 60):
            if idx not in chi[spec]:
                raise ValueError(f"CHI_MLS({spec},{idx}) missing")
            chi_arr[spec - 1, idx - 1] = chi[spec][idx]
    out["CHI_MLS"] = chi_arr

    return out


def write_npy(tables: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, arr in tables.items():
        p = out_dir / f"{name}.npy"
        np.save(p, arr)
        written.append(p)
    return written


def check_against_existing(tables: dict[str, np.ndarray], out_dir: Path) -> bool:
    """Return True iff every existing .npy matches the freshly extracted values."""
    ok = True
    for name, arr in tables.items():
        p = out_dir / f"{name}.npy"
        if not p.exists():
            print(f"  {name}: no existing .npy (will be written)")
            continue
        old = np.load(p)
        if old.shape != arr.shape:
            print(f"  {name}: SHAPE MISMATCH old={old.shape} new={arr.shape}")
            ok = False
        elif np.array_equal(old, arr):
            print(f"  {name}: OK (matches existing)")
        else:
            max_d = float(np.max(np.abs(old - arr)))
            print(f"  {name}: DIFFERS max|d|={max_d:.3e}")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--surrtrf",
        default=os.environ.get("SURRTRF", str(_DEFAULT_SURRTRF)),
        help="path to surrtrf.F90 (default: %(default)s)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(_HERE / "tables"),
        help="where to write <NAME>.npy (default: %(default)s)",
    )
    ap.add_argument("--check", action="store_true", help="only verify against existing .npy")
    args = ap.parse_args()

    tables = extract(Path(args.surrtrf))
    print(f"parsed {len(tables)} tables from {args.surrtrf}:")
    for name, arr in tables.items():
        print(f"  {name:10s} shape={arr.shape}  range=[{arr.min():.4e}, {arr.max():.4e}]")

    out_dir = Path(args.out_dir)
    if args.check:
        print("\nchecking against existing .npy:")
        return 0 if check_against_existing(tables, out_dir) else 1

    written = write_npy(tables, out_dir)
    print(f"\nwrote {len(written)} files:")
    for p in written:
        print(f"  {p}")
    print("\npost-write verification:")
    return 0 if check_against_existing(tables, out_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
