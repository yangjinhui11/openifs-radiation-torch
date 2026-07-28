# Physical constants for the OpenIFS radiation torch port.
#
# Transcribed from ``openifs-48r1/ifs-source/arpifs/phys_radi/radpar.F90`` and
# ``yomcst.F90`` (YOMCST) for the values the radiation kernels actually consume.
# These are full-precision doubles, identical bit-for-bit to the Fortran.
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# YOMCST — fundamental physical constants
# ---------------------------------------------------------------------------
G: float = 9.80665              # m/s2  gravitational acceleration (RG)
RD: float = 287.059640          # J/K/kg dry-air gas constant (RD)
RV: float = 461.525029          # J/K/kg water-vapour gas constant (RV)
CPD: float = 1004.707795        # J/K/kg dry-air specific heat at constant p (RCPD)
CPV: float = 1846.              # J/K/kg water-vapour specific heat (RCPV)
RETV: float = RV / RD - 1.0     # ~0.6078 (RETV)

# Thermodynamic (YOETHF / saturation)
RTT: float = 273.16             # K triple point of water (RTT)
RVTMP2: float = RV / RD - 1.0   # used in cp_moist = cpd * (1 + RVTMP2*q)

# Latent heats (J/kg)
RLVTT: float = 2500800.0        # vaporisation at RTT (RLVTT)
RLSTT: float = 2834500.0        # sublimation at RTT (RLSTT)
RLMLT: float = RLSTT - RLVTT    # fusion at RTT (334.7 kJ/kg)

# Derived ratios used by cloud optics
RALVDCP: float = RLVTT / CPD    # ~2491.4
RALSDCP: float = RLSTT / CPD    # ~2821.6


# ---------------------------------------------------------------------------
# Radiation-specific (radpar.F90 / yoerad.F90)
# ---------------------------------------------------------------------------
# Stefan-Boltzmann (RSAVO in yoerad; sigma for blackbody)
SIGMA: float = 5.670374419e-8   # W/m2/K4

# Planck constants used in LW (yoerrtf) -- filled in M3 from the yoerrtf module
# but kept here so M1/M2 can reference the symbol without importing tables.


# ---------------------------------------------------------------------------
# RRTM_ECRT_140GP molecular constants (rrtm_ecrt_140gp.F90)
# Atomic/molecular weights (g/mol) and Avogadro's number used to convert
# mass mixing ratios to volume mixing ratios and column amounts.
# ---------------------------------------------------------------------------
MOL_WEIGHT_DRY_AIR: float = 28.970          # ZAMD
MOL_WEIGHT_H2O: float = 18.0154             # ZAMW
MOL_WEIGHT_CO2: float = 44.011               # ZAMCO2
MOL_WEIGHT_O3: float = 47.9982               # ZAMO
MOL_WEIGHT_CH4: float = 16.043               # ZAMCH4
MOL_WEIGHT_N2O: float = 44.013               # ZAMN2O
MOL_WEIGHT_CFC11: float = 137.3686           # ZAMC11
MOL_WEIGHT_CFC12: float = 120.9140           # ZAMC12
MOL_WEIGHT_CFC22: float = 86.4690            # ZAMC22
MOL_WEIGHT_CCL4: float = 153.8230            # ZAMCL4
AVOGADRO: float = 6.02214e23                 # ZAVGDRO (molecules/mole)

# Gravity used by RRTM in cgs (cm/s2). Fortran: ZGRAVIT = (RG/RPLRG)*1.E2;
# for normal Earth RG = 9.80665 m/s2 and RPLRG = 1.
GRAVITY_CGS: float = G * 100.0

# Oxygen volume mixing ratio used in the RRTM broadening gas column.
VMR_O2: float = 0.209488

# Default cloud overlap scheme (NOVLP in YOERAD). 1 = max-random overlap is
# the IFS operational default; 4 = clear-sky fraction = 1 (no cloud scaling).
OVERLAP_DEFAULT: int = 1

