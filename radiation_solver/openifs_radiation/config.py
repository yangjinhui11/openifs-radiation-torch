"""Configuration for the OpenIFS radiation torch port.

All values are transcribed verbatim from
``openifs-48r1/ifs-source/arpifs/phys_radi/suecrad.F90`` (production defaults
for ``LECMWF=.TRUE.``) and the associated YOMCST / YOETHF constants. Keeping
them in one place lets the torch port and the Fortran reference share an
identical parameterisation -- any mismatch is then purely arithmetic.

Vertical orientation follows OpenIFS: level 1 = top of atmosphere,
level ``nlev`` = surface (pressure increasing with index). The leading axis of
every torch field is the vertical axis ``(nlev, nlon)``.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Spectral discretisation (suecrad.F90 production defaults)
# ---------------------------------------------------------------------------
NSW: int = 6  # number of shortwave albedo bands (YERAD%NSW)
NLWEMISS: int = 6  # number of longwave emissivity bands

# RRTM g-point structure (140 g-points total, 16 LW bands)
# (YRESWRT layout; matches rrtm_rrtm_140gp.F90)
NGPTLW: int = 140
NLWBANDS: int = 16


# ---------------------------------------------------------------------------
# Well-mixed gas volume-mixing ratios (suecrad.F90 defaults)
# H2O and O3 are prognostic and come from the state.
# ---------------------------------------------------------------------------
VMR_CO2: float = 420.0e-6     # carbon dioxide
VMR_CH4: float = 1.8e-6       # methane   (RCH4 in suecrad)
VMR_N2O: float = 3.2e-7       # nitrous oxide (RN2O)
VMR_NO2: float = 0.0          # NO2 (set to 0 in production)
VMR_CFC11: float = 0.0        # RCFC11 — set to 0 unless scenario overrides
VMR_CFC12: float = 0.0        # RCFC12
VMR_CFC22: float = 0.0        # RHCFC22
VMR_CCL4: float = 0.0         # RCCL4


# ---------------------------------------------------------------------------
# Solar constant and astronomical parameters
# ---------------------------------------------------------------------------
SOLAR_IRRADIANCE: float = 1361.0  # W/m2 total solar irradiance at 1 AU (PRII0)


# ---------------------------------------------------------------------------
# Cloud overlap (NRADLP=2 production: exponential-random)
# ---------------------------------------------------------------------------
OVERLAP_DECORR_LEN_DEFAULT: float = 2000.0  # m, cloud overlap decorrelation length


@dataclass
class RadiationConfig:
    """User-tunable configuration. Defaults match production 48r1."""
    nsw: int = NSW
    nlev: int = 137  # standard 137-level vertical grid
    dtype: object = None  # set to torch.float64 by OpenIFSRadiation.__init__
    device: str = "cpu"
    cloud_optics: bool = True          # include cloud radiative effects
    interval_steps: int = 1            # recompute radiation every N steps
    # Allow the torch port to fall back to the bridge for un-implemented kernels
    # during M1-M5 incremental bring-up.
    allow_bridge_fallback: bool = False
    bridge_fallback_name: str = "openifs_ecrad_radiation"
