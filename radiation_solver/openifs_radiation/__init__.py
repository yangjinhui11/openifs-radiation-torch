"""Public API for the OpenIFS radiation torch port.

Currently in M0 (scaffold). The top-level ``OpenIFSRadiation`` nn.Module will
land in M6 once the underlying kernels (RADLSW setup, LW, SW) are ported and
bit-exact-validated against the Fortran reference. For now, this module exposes
the configuration and constants so downstream code can be written against the
target API.
"""
from __future__ import annotations

from .config import RadiationConfig, NSW, NLWEMISS, NGPTLW, NLWBANDS
from . import constants

__all__ = [
    "RadiationConfig",
    "constants",
    "NSW",
    "NLWEMISS",
    "NGPTLW",
    "NLWBANDS",
]
