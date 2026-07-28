"""Port of radaca.F90 — Tegen 6-type aerosol optical depth.

Fortran reference: openifs-48r1/ifs-source/arpifs/phys_radi/radaca.F90 (890 lines)

Current status: SKELETON.  Production runs pass aerosols externally;
the full radaca climatology interpolation will be ported when needed
for aerosol-active simulations.

For clear-sky / aerosol-free profiles, PAER = 0 and optical depths = 0.
"""
import torch


def radaca(
    paprs: torch.Tensor,      # (nlon, nlev+1) half-level pressure [Pa], TOA→surface
    pgelam: torch.Tensor,     # (nlon,) longitude [rad]
    psin: torch.Tensor,       # (nlon,) sine of latitude
    pclon: torch.Tensor,      # (nlon,) cos(longitude)
    pslon: torch.Tensor,      # (nlon,) sin(longitude)
    pth: torch.Tensor,        # (nlon, nlev+1) half-level temperature [K]
) -> dict:
    """Compute 6-type Tegen aerosol optical depths (skeleton).

    Returns all outputs matching the Fortran signature, zero-filled.
    """
    nlon, nlev1 = paprs.shape
    nlev = nlev1 - 1
    dt = paprs.dtype
    dev = paprs.device

    return {
        "paer": torch.zeros(nlon, 6, nlev, dtype=dt, device=dev),
        "paero": torch.zeros(nlon, nlev, 1, dtype=dt, device=dev),
        "pozon": torch.zeros(nlon, nlev, dtype=dt, device=dev),
        "paersig": torch.zeros(nlon, dtype=dt, device=dev),
        "podto": torch.zeros(nlon, dtype=dt, device=dev),
        "podss": torch.zeros(nlon, dtype=dt, device=dev),
        "poddu": torch.zeros(nlon, dtype=dt, device=dev),
        "podom": torch.zeros(nlon, dtype=dt, device=dev),
        "podbc": torch.zeros(nlon, dtype=dt, device=dev),
        "podsu": torch.zeros(nlon, dtype=dt, device=dev),
    }
