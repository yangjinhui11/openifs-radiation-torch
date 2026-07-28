"""RRTM reference-profile tables used by SETCOEF.

Tables are extracted from ``openifs-48r1/ifs-source/arpifs/phys_radi/surrtrf.F90``
by ``_extract_setcoef_tables.py`` and stored as ``.npy`` files. They are loaded
once at import time and padded with a dummy index 0 to preserve the Fortran
1-based indexing used by ``rrtm_setcoef_140gp``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent


def _load_table(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    path = _HERE / "tables" / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{name}.npy not found at {path}. Run _extract_setcoef_tables.py first."
        )
    arr = np.load(path)
    if arr.shape != shape:
        raise ValueError(f"{name}.npy has shape {arr.shape}, expected {shape}")
    return torch.from_numpy(arr).to(torch.float64)


# (59,) reference pressures (hPa) and their logs, plus log reference temperatures.
PREF_SRC = _load_table("PREF", (59,))
PREFLOG_SRC = _load_table("PREFLOG", (59,))
TREF_SRC = _load_table("TREF", (59,))
CHI_MLS_SRC = _load_table("CHI_MLS", (7, 59))

# Pad with dummy index 0 so that Fortran indices 1..N map directly to Python.
PREF = torch.zeros(60, dtype=torch.float64)
PREFLOG = torch.zeros(60, dtype=torch.float64)
TREF = torch.zeros(60, dtype=torch.float64)
PREF[1:60] = PREF_SRC
PREFLOG[1:60] = PREFLOG_SRC
TREF[1:60] = TREF_SRC

CHI_MLS = torch.zeros(8, 60, dtype=torch.float64)
CHI_MLS[1:8, 1:60] = CHI_MLS_SRC

__all__ = ["PREF", "PREFLOG", "TREF", "CHI_MLS"]
