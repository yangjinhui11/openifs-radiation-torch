"""Classic 6-band shortwave solver (Fouquart & Bonnel 1980).

Clear-sky path only. Cloud/aerosol/delta-Eddington deferred.
"""
from .driver import sw

__all__ = ["sw"]
