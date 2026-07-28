"""Cloud-optics tables for the LW + SW ZTAUCLD / cloud-optical-property computations.

All tables are loaded from ``.npy`` files generated from the verbatim upstream
source by ``_extract_tables.py``. The .npy files store the *source* values as
assigned in ``suclopn.F90``; any scaling the source applies at load time
(``RASWCA *= 1e-2``, ``RASWCF *= 1e-3``) is applied here so the public names
match what ``radlsw.F90`` actually consumes.

Production configuration (48r1):
  * LW cloud optics (CASE 12): NLWLIQOPT=2 (Lindner-Li) + NLWICEOPT=3 (Fu-98)
  * SW cloud optics:            NSWLIQOPT=2 (Slingo-89) + NSWICEOPT=3 (Fu-96)
  * NSW = 6 spectral bands

To regenerate after an upstream change::

    python3 openifs_radiation/_extract_tables.py            # rewrite .npy
    python3 openifs_radiation/_extract_tables.py --check    # verify only
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent


def _load(name: str) -> torch.Tensor:
    """Load one table from ``<name>.npy`` next to this module (float64)."""
    path = _HERE / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{name}.npy not found at {path}; run "
            f"`python3 openifs_radiation/_extract_tables.py` to regenerate "
            f"the cloud-optics tables from suclopn.F90")
    return torch.from_numpy(np.load(path)).clone()


# ---------------------------------------------------------------------------
# LW CASE-12 cloud-optics tables  (16, k) -- RRTM bands
# ---------------------------------------------------------------------------
RLILIA = _load("RLILIA")   # (16, 5) Lindner-Li liquid mass-extinction coeffs
RLILIB = _load("RLILIB")   # (16, 4) Lindner-Li liquid single-scatter albedo
RFUETA = _load("RFUETA")   # (16, 3) Fu-98 ice mass-extinction coeffs
RFUETB = _load("RFUETB")   # (16, 4) Fu-98 ice coalbedo coeffs
RFUETC = _load("RFUETC")   # (16, 4) Fu-98 ice asymmetry coeffs


# ---------------------------------------------------------------------------
# SW cloud-optics tables  (6,) -- NSW=6 spectral bands (production)
# ---------------------------------------------------------------------------
# Slingo (1989) liquid water, NSWLIQOPT=2. suclopn applies:
#   RASWCA = ZASWCA6 * 1e-2 ; RASWCF = ZASWCF6 * 1e-3 ; others unscaled.
# Source: suclopn.F90 lines 979-984 (KSW==6 branch).
RASWCA = _load("ZASWCA6") * 1.0e-2     # mass-extinction intercept
RASWCB = _load("ZASWCB6")              # mass-extinction 1/re slope
RASWCC = _load("ZASWCC6")              # single-scatter intercept (1-omega)
RASWCD = _load("ZASWCD6")              # single-scatter 1/re slope
RASWCE = _load("ZASWCE6")              # asymmetry intercept
RASWCF = _load("ZASWCF6") * 1.0e-3     # asymmetry re slope

# Fu (1996) ice, NSWICEOPT=3. unscaled from source (suclopn lines 1023-1032).
RFUAA0 = _load("ZFUAA06")              # mass-extinction intercept
RFUAA1 = _load("ZFUAA16")              # mass-extinction 1/de slope
RFUBB0 = _load("ZFUBB06")              # single-scatter poly c0
RFUBB1 = _load("ZFUBB16")              # single-scatter poly c1
RFUBB2 = _load("ZFUBB26")              # single-scatter poly c2
RFUBB3 = _load("ZFUBB36")              # single-scatter poly c3
RFUCC0 = _load("ZFUCC06")              # asymmetry poly c0
RFUCC1 = _load("ZFUCC16")              # asymmetry poly c1
RFUCC2 = _load("ZFUCC26")              # asymmetry poly c2
RFUCC3 = _load("ZFUCC36")              # asymmetry poly c3


__all__ = [
    # LW CASE-12
    "RLILIA", "RLILIB", "RFUETA", "RFUETB", "RFUETC",
    # SW Slingo-89 liquid
    "RASWCA", "RASWCB", "RASWCC", "RASWCD", "RASWCE", "RASWCF",
    # SW Fu-96 ice
    "RFUAA0", "RFUAA1",
    "RFUBB0", "RFUBB1", "RFUBB2", "RFUBB3",
    "RFUCC0", "RFUCC1", "RFUCC2", "RFUCC3",
]
