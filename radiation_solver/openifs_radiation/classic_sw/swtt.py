"""Port of swtt.F90 + swtt1.F90 — Pade gas transmission functions.

Fortran references:
  openifs-48r1/ifs-source/arpifs/phys_radi/swtt.F90
  openifs-48r1/ifs-source/arpifs/phys_radi/swtt1.F90

Uses APAD/BPAD/D tables extracted from suswn.F90. The Horner evaluation uses
pure tensor ops (autograd-compatible, no JIT).
"""
import numpy as np
import torch
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TABLE_DIR = _HERE.parent / "rrtm_lw" / "tables"

# Lazy-loaded tables (CPU) + device cache
_tables = {}
_dev_cache = {}   # {(key, device_str): tensor_on_device}


def _load():
    if not _tables:
        _tables["apad"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_apad.npy"))).to(torch.float64)
        _tables["bpad"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_bpad.npy"))).to(torch.float64)
        _tables["d"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_d.npy"))).to(torch.float64)
        _tables["nexpo3"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_nexpo3.npy"))).to(torch.int64)
        _tables["rxpo3"] = torch.from_numpy(
            np.load(str(_TABLE_DIR / "sw_rxpo3.npy"))).to(torch.float64)


def _get(key, dev):
    """Return table ``key`` on device ``dev``, caching the transfer."""
    ck = (key, str(dev))
    t = _dev_cache.get(ck)
    if t is None:
        t = _tables[key].to(dev)
        _dev_cache[ck] = t
    return t


def swtt(knu: int, ka: int, pu: torch.Tensor) -> torch.Tensor:
    """Single-absorber Pade transmission (swtt.F90).

    Args:
        knu: spectral interval index (1..6)
        ka:  absorber index (1..3: 1=H2O, 2=UMG, 3=O3)
        pu:  absorber amount (nlon,)

    Returns:
        ptr: transmission (nlon,), in [0, 1].
    """
    _load()
    dev = pu.device
    apad = _get("apad", dev)
    bpad = _get("bpad", dev)
    d = _get("d", dev)

    i = knu - 1; j = ka - 1
    a = apad[i, j, :]                 # (7,)
    b = bpad[i, j, :]
    zd = d[i, j]

    zr1 = _horner_vec(a, pu)          # (nlon,)
    zr2 = _horner_vec(b, pu)
    ptr = (zr1 / zr2) * (1.0 - zd) + zd
    return ptr


def _horner_vec(coef: torch.Tensor, pu: torch.Tensor) -> torch.Tensor:
    """Vectorised Horner evaluation for a single coefficient set."""
    zr = coef[6].expand_as(pu)
    for k in range(5, -1, -1):
        zr = coef[k] + pu * zr
    return zr


def swtt1(knu: int, kabs: int, kkind: list, pu: torch.Tensor) -> torch.Tensor:
    """Multi-absorber Pade transmission (swtt1.F90)."""
    _load()
    dev = pu.device
    apad = _get("apad", dev)
    bpad = _get("bpad", dev)
    d = _get("d", dev)
    i = knu - 1
    # Gather coefficient rows for all absorbers at once: (kabs, 7).
    idx = torch.tensor([k - 1 for k in kkind], device=dev)
    a = apad[i].index_select(0, idx)          # (kabs, 7)
    b = bpad[i].index_select(0, idx)          # (kabs, 7)
    zd = d[i].index_select(0, idx)            # (kabs,)

    # Horner over the last axis (7), broadcasting against pu (nlon, kabs):
    # coef (kabs,7) broadcasts with pu (nlon,kabs) -> need pu unsqueezed to
    # (nlon, kabs, 1) and coef (1, kabs, 7). Sum reduces to (nlon, kabs).
    zr1 = _horner_batched(a, pu)              # (nlon, kabs)
    zr2 = _horner_batched(b, pu)
    ptr = (zr1 / zr2) * (1.0 - zd) + zd
    return ptr


def _horner_batched(coef: torch.Tensor, pu: torch.Tensor) -> torch.Tensor:
    """Horner for multiple coefficient sets evaluated at corresponding pu."""
    # Broadcast coef (kabs,7) -> (1,kabs,7), pu (nlon,kabs) -> (nlon,kabs,1).
    c = coef.unsqueeze(0)                       # (1, kabs, 7)
    p = pu.unsqueeze(-1)                        # (nlon, kabs, 1)
    zr = c[..., 6]                              # (1, kabs) -> broadcast (nlon,kabs)
    zr = zr.expand_as(pu)
    for k in range(5, -1, -1):
        zr = c[..., k] + p[..., 0] * zr         # (nlon, kabs)
    return zr


def swuvo3(knu: int, pu: torch.Tensor) -> torch.Tensor:
    """O3 transmission in the UV/visible (swuvo3.F90).

    Sum-of-exponentials form::

        PTR = Σ_{jx=1..NEXPO3(KNU)} REXPO3(KNU,1,jx)
                                          * exp(-REXPO3(KNU,2,jx) * PU)

    Only bands 1-3 carry non-zero O3 absorption in the production NSW=6 config
    (NEXPO3 = [7,7,6,4,0,0]); swuvo3 is only called from sw1s (bands 1-3) in
    that path. Bands with NEXPO3=0 return PTR=0, matching the Fortran reference
    verbatim.

    Args:
        knu: spectral interval (1..6).
        pu:  O3 column amount (nlon,) or (nlon, kabs).

    Returns:
        ptr: O3 transmission, same shape as ``pu``.
    """
    _load()
    nexpo3 = _tables["nexpo3"]               # (6,) int, stays on CPU
    rxpo3 = _get("rxpo3", pu.device)         # (6, 2, 7) on device
    i = knu - 1
    n = int(nexpo3[i].item())
    if n == 0:
        return torch.zeros_like(pu)
    coef = rxpo3[i, 0, :n]                   # (n,)
    expo = rxpo3[i, 1, :n]                   # (n,)
    pu_exp = pu.unsqueeze(-1) * (-expo)      # (..., n)
    return (coef * torch.exp(pu_exp)).sum(dim=-1)
