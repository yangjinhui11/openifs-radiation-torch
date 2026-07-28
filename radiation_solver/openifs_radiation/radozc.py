"""Port of radozc.F90 — latitude-pressure ozone climatology interpolation.

Fortran reference: openifs-48r1/ifs-source/arpifs/phys_radi/radozc.F90

RPROC is TOA→surface (ascending pressure Pa), with zero-padding beyond IMAXC.
PAPRS is also TOA→surface (ascending pressure Pa).
All operations vectorised — no Python loops over columns or levels.
"""
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_HERE = Path(__file__).resolve().parent
_TABLE_DIR = _HERE / "rrtm_lw" / "tables"


def _lt(name):
    p = _TABLE_DIR / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing ozone table: {p}")
    return torch.from_numpy(np.load(str(p))).to(torch.float64)


@dataclass
class OzoneClimatology:
    rsinc: torch.Tensor  # (NLAT,) sine of latitude
    rozt: torch.Tensor   # (NLAT, NP) ozone mass mixing ratio (kg/kg)
    rproc: torch.Tensor  # (NP,)  reference pressure [Pa] TOA→sfc


def _load_climatology() -> OzoneClimatology:
    try:
        rsinc = _lt("ozone_rsinc")
        rozt = _lt("ozone_rozt")
        rproc = _lt("ozone_rproc")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Ozone climatology tables not found.  Run "
            "'python -m openifs_radiation.radozc --build-tables' first."
        )
    return OzoneClimatology(rsinc=rsinc, rozt=rozt, rproc=rproc)


def _build_standard_tables():
    """Build ozone climatology from suecozc.F90 / ICRCCM standard profile."""
    import numpy as np
    nlat = 64
    rsinc = np.sin(np.deg2rad(np.linspace(-90, 90, nlat)))

    # RPROC from suecozc.F90 (Pa, TOA→sfc, indices 1-35 active)
    rproc_pa = np.zeros(61)
    rproc_pa[1:36] = np.array([
        30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0,
        1500.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0,
        30000.0, 50000.0, 70000.0, 100000.0, 110000.0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ])

    # Standard ICRCCM mid-latitude ozone (kg/kg) at RPROC levels
    std_p_hpa = np.array([
        0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0,
        15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0,
        500.0, 700.0, 1000.0, 1100.0,
    ]) * 100.0  # → Pa, TOA→sfc
    std_o3_mmr = np.array([
        1.60e-05, 1.50e-05, 1.40e-05, 1.30e-05, 1.20e-05, 1.10e-05,
        9.00e-06, 7.00e-06, 5.00e-06, 4.00e-06, 3.00e-06, 2.00e-06,
        1.50e-06, 1.00e-06, 7.00e-07, 5.00e-07, 3.50e-07, 2.50e-07,
        2.00e-07, 1.50e-07, 1.20e-07, 1.00e-07, 5.00e-08,
    ])
    rozt_base = np.zeros(61)
    rozt_base[0] = 1.60e-05  # top-of-atmosphere extrapolation
    for i in range(1, 36):
        p = rproc_pa[i]
        rozt_base[i] = np.interp(np.log(p), np.log(std_p_hpa), std_o3_mmr)
    rozt = np.tile(rozt_base, (nlat, 1))

    np.save(str(_TABLE_DIR / "ozone_rsinc.npy"), rsinc)
    np.save(str(_TABLE_DIR / "ozone_rozt.npy"), rozt.astype(np.float64))
    np.save(str(_TABLE_DIR / "ozone_rproc.npy"), rproc_pa.astype(np.float64))
    print(f"[radozc] Built ozone climatology tables → {_TABLE_DIR}/ozone_*")


def radozc(
    paprs: torch.Tensor,
    pgemu: torch.Tensor,
    climatology: Optional[OzoneClimatology] = None,
) -> torch.Tensor:
    """Compute layer-integrated ozone amount from climatology.

    Args:
        paprs:  (nlon, nlev+1) half-level pressure [Pa], TOA→surface.
        pgemu:  (nlon,) sine of latitude per column.
        climatology: pre-loaded tables (auto-loads if None).

    Returns:
        ozon: (nlon, nlev) layer ozone amount [Pa · kg/kg].
    """
    if climatology is None:
        climatology = _load_climatology()

    dev = paprs.device; dt = paprs.dtype
    nlon, nlev1 = paprs.shape; nlev = nlev1 - 1
    rsinc = climatology.rsinc.to(dev)
    rozt = climatology.rozt.to(dev)
    rproc = climatology.rproc.to(dev)
    nlat = rsinc.shape[0]

    # ---- 1. Latitude interpolation ------------------------------------------
    zsin = pgemu                                                     # (nlon,)
    inla = torch.zeros(nlon, dtype=torch.long, device=dev)
    zsilat = torch.zeros(nlon, dtype=dt, device=dev)
    llatint = torch.zeros(nlon, dtype=torch.bool, device=dev)

    lo = zsin <= rsinc[0]; hi = zsin >= rsinc[-1]; md = ~lo & ~hi
    inla[lo] = 0; inla[hi] = nlat - 1
    idx = torch.searchsorted(rsinc, zsin[md]); idx = idx.clamp(1, nlat - 1)
    inla[md] = idx - 1
    zsilat[md] = (zsin[md] - rsinc[inla[md]]) / (rsinc[inla[md] + 1] - rsinc[inla[md]])
    llatint[md] = True

    # ---- 2. Latitude-interpolated profile -----------------------------------
    zozlt = rozt[inla, :]                                            # (nlon, NP)
    zsilat3 = zsilat.unsqueeze(1)
    zozlt_hi = rozt[(inla + 1).clamp(0, nlat - 1), :]
    zozlt = torch.where(llatint.unsqueeze(1),
                        zozlt + zsilat3 * (zozlt_hi - zozlt), zozlt)

    # ---- 3. Vertical interpolation ------------------------------------------
    # Find IMAXC: last non-zero rproc
    imaxc = int((rproc > 0).nonzero(as_tuple=False)[-1].item())      # NP-1
    np0 = imaxc + 1                                                  # active levels

    # ZRRR(JC) for JC=0..IMAXC-1 (Fortran style, 1-based offset included)
    zrrr = torch.zeros(np0 - 1, dtype=dt, device=dev)
    active = rproc[:np0 - 1] > 0
    drp = rproc[:np0 - 1][active] - rproc[1:np0][active]
    dz = zozlt[:, :np0 - 1][:, active] - zozlt[:, 1:np0][:, active]
    zrrr[active] = (dz / drp.unsqueeze(0)).mean(dim=0)  # average over columns

    zozon = torch.zeros(nlon, nlev1, dtype=dt, device=dev)
    for jc in range(np0 - 1):
        p_lo = rproc[jc]        # lower P (TOA-side)
        p_hi = rproc[jc + 1]    # higher P (sfc-side)
        if p_hi <= 0:
            continue
        mask = (paprs >= p_lo) & (paprs < p_hi)                      # (nlon, nlev+1)
        val = zozlt[:, jc + 1].unsqueeze(1) + (paprs - p_hi) * zrrr[jc]
        zozon = torch.where(mask, val, zozon)

    # Levels beyond the highest climatology level
    mask_top = paprs >= rproc[np0 - 1]
    zozon = torch.where(mask_top, zozlt[:, np0 - 1].unsqueeze(1), zozon)

    # ---- 4. Vertical integration (trapezoidal) ------------------------------
    ozon = (paprs[:, 1:] - paprs[:, :-1]) * (zozon[:, :-1] + zozon[:, 1:]) * 0.5
    return ozon


if __name__ == "__main__":
    import sys
    if "--build-tables" in sys.argv:
        _build_standard_tables()
    else:
        print("Usage: python -m openifs_radiation.radozc --build-tables")
