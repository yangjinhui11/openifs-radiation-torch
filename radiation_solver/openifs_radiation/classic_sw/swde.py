"""Port of swde.F90 — delta-Eddington cloudy-layer reflectivity/transmissivity.

Fortran reference: openifs-48r1/ifs-source/arpifs/phys_radi/swde.F90

Pure PyTorch implementation (autograd-compatible, no JIT).
"""
import torch
import math


def swde(
    gg: torch.Tensor,       # (n,) asymmetry factor
    pref: torch.Tensor,     # (n,) reflectivity of underlying layer
    prmuz: torch.Tensor,    # (n,) cosine of solar zenith angle
    pto1: torch.Tensor,     # (n,) optical thickness
    pw: torch.Tensor,       # (n,) single-scattering albedo
    novlp: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delta-Eddington layer reflectivity / transmissivity.

    Returns:
        pre1: (n,) reflectivity (no reflection from below)
        ptr1: (n,) transmissivity (no reflection from below)
        pre2: (n,) reflectivity (with reflection from below)
        ptr2: (n,) transmissivity (with reflection from below)
    """
    dt = gg.dtype
    dev = gg.device
    replog = 1e-12  # YOERDU REPLOG
    zdt = 2.0 / 3.0
    zdt2 = 2.0 * zdt

    # Delta-Eddington scaling
    if novlp >= 5:
        # MesoNH version
        zgp = gg
        ztop = pto1
        zwcp = pw
    else:
        # ECMWF version (production config)
        zff = gg * gg
        zgp = gg / (1.0 + gg)
        zpwff = 1.0 - pw * zff
        ztop = zpwff * pto1
        zwcp = (1.0 - zff) * pw / zpwff

    zx1 = 1.0 - zwcp * zgp
    zwm = 1.0 - zwcp
    zrm2 = prmuz * prmuz
    # ZRK = sqrt(MAXJ(REPLOG, 3*ZWM*ZX1)) — Fortran clamps the sqrt argument.
    zrk = torch.sqrt(torch.clamp(3.0 * zwm * zx1, min=replog))
    zx2 = (1.0 - zrk * zrk * zrm2) * zdt2
    # ZRR/ZRP are plain divisions in Fortran (no clamp). The previous clamp
    # flipped the sign of zx2 when zrk^2*zrm2 > 1, producing ~1e30 garbage
    # instead of a finite negative — that broke bit-exactness.
    zrr = 1.0 / zx2
    zrp = zrk / zx1
    zalpha = zwcp * zrm2 * (1.0 + zgp * zwm) * zrr
    zbeta = zwcp * prmuz * (1.0 + 3.0 * zgp * zrm2 * zwm) * zrr

    # Exponential arguments — Fortran clamps ZTOP*ZPRMUZ and ZRK*ZTOP to ±200.
    zprmu = 1.0 / prmuz
    zzarg = -torch.clamp(ztop * zprmu, min=-200.0, max=200.0)
    zzarg2 = torch.clamp(zrk * ztop, max=200.0)

    zexmu0 = torch.exp(zzarg)
    zexkp = torch.exp(zzarg2)
    # Fortran: ZEXKM = 1/ZEXKP (no clamp — the exp clamps above prevent overflow).
    zexkm = 1.0 / zexkp

    zxp2p = 1.0 + zdt * zrp
    zxm2p = 1.0 - zdt * zrp
    zap2b = zalpha + zdt * zbeta
    zam2b = zalpha - zdt * zbeta

    # --- 1.2 Without reflection from underlying layer ---
    za22 = zxp2p * zexkp
    za21 = zxm2p * zexkm
    za23 = zam2b * zexmu0
    zdena = zxp2p * za22 - za21 * zxm2p
    zidena = 1.0 / zdena
    zc1a = (za22 * zap2b - zxm2p * za23) * zidena
    zc2a = (zxp2p * za23 - za21 * zap2b) * zidena
    zri0a = zc1a + zc2a - zalpha
    zri1a = zrp * (zc1a - zc2a) - zbeta
    pre1 = (zri0a - zdt * zri1a) * zprmu

    zri0b = zc1a * zexkm + zc2a * zexkp - zalpha * zexmu0
    zri1b = zrp * (zc1a * zexkm - zc2a * zexkp) - zbeta * zexmu0
    ptr1 = zexmu0 + (zri0b + zdt * zri1b) * zprmu

    # --- 1.3 With reflection from underlying layer ---
    zb21 = za21 - pref * zxp2p * zexkm
    zb22 = za22 - pref * zxm2p * zexkp
    zb23 = za23 - pref * zexmu0 * (zap2b - prmuz)
    zdenb = zxp2p * zb22 - zb21 * zxm2p
    zidenb = 1.0 / zdenb
    zc1b = (zb22 * zap2b - zxm2p * zb23) * zidenb
    zc2b = (zxp2p * zb23 - zb21 * zap2b) * zidenb
    zri0c = zc1b + zc2b - zalpha
    zri1c = zrp * (zc1b - zc2b) - zbeta
    pre2 = (zri0c - zdt * zri1c) * zprmu

    zri0d = zc1b * zexkm + zc2b * zexkp - zalpha * zexmu0
    zri1d = zrp * (zc1b * zexkm - zc2b * zexkp) - zbeta * zexmu0
    ptr2 = zexmu0 + (zri0d + zdt * zri1d) * zprmu

    return pre1, ptr1, pre2, ptr2
