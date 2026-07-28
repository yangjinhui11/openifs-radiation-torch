"""Shared path configuration for experiment scripts.

Centralizes the resolution of (1) the radiation_solver package root and
(2) the ERA5 data directory, so that individual experiment scripts do not
hard-code absolute paths.

Resolution order (first match wins):
  - environment variable RAD_ROOT / ERA5_DATA (if set)
  - the radiation_solver shipped in this repository
  - the legacy server path /home/qixiang/... (for backward compatibility)

Usage in an experiment script:
    from path_config import setup_paths, find_era5
    setup_paths()                       # makes openifs_radiation importable
    era5_path = find_era5("global_sw.npz")
    g = np.load(era5_path)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# This file lives at <repo>/experiments/path_config.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOLVER = _REPO_ROOT / "radiation_solver"
_LEGACY = Path("/home/qixiang/yangjinhui/openifs/physics_callpar/openifs_radiation_pytorch")

# ERA5 data candidates
_DATA_CANDIDATES = [
    _REPO_ROOT / "data",
    Path("/tmp"),
    Path("/data/era5"),
]


def solver_root() -> Path:
    """Return the radiation_solver root, preferring env var then repo then legacy."""
    env = os.environ.get("RAD_ROOT")
    if env and Path(env).exists():
        return Path(env)
    if (_SOLVER / "openifs_radiation" / "classic_sw" / "driver.py").exists():
        return _SOLVER
    if _LEGACY.exists():
        return _LEGACY
    raise FileNotFoundError(
        "radiation_solver not found. Set RAD_ROOT env var or check "
        "radiation_solver/ in the repo root.")


def setup_paths() -> Path:
    """Insert the solver root into sys.path and set DATA env var.

    Returns the solver root for reference.
    """
    root = solver_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ.setdefault("DATA", root_str)
    return root


def find_era5(filename: str) -> Path:
    """Locate an ERA5 data file by name across candidate data directories."""
    env = os.environ.get("ERA5_DATA")
    if env:
        p = Path(env) / filename
        if p.exists():
            return p
    for d in _DATA_CANDIDATES:
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        "%s not found in %s. Set ERA5_DATA env var or regenerate per "
        "data/README.md." % (filename, [str(d) for d in _DATA_CANDIDATES]))
