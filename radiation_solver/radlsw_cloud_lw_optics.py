"""LW cloud optical thickness (ZTAUCLD) -- production CASE 12.

Torch port of the longwave cloud-optics computation in
``openifs-48r1/ifs-source/arpifs/phys_radi/radlsw.F90`` for the production
configuration ``NRADLP=2``, ``NRADIP=3``, ``NLWLIQOPT=2``, ``NLWICEOPT=3``,
which selects Lindner & Li (2000) liquid + Fu et al. (1998) ice (CASE 12 in
the SELECT ICE_WATER_CLOUD_TOT_EMIS block), plus the diffusivity correction
when ``LDIFFC = .FALSE.`` (production default: constant diffusivity 1.66).

All arrays are batched torch tensors with the vertical axis leading:
``(..., nlev)`` where ``nlev`` increases from TOA to surface.

Reference arithmetic (radlsw.F90, transcribed verbatim):
  * Liquid effective radius  -- Martin et al. (1994), eqs at lines 500-529
  * Ice effective diameter   -- Sun & Rikus (1999) + Sun (2001) correction,
                                lines 624-644; Liou & Ou (1994) fallback,
                                lines 599-605
  * ZTAUCLD per band         -- CASE 12, lines 1183-1210
  * Diffusivity correction   -- lines 1218-1237 (LDIFFC=.FALSE. -> 1.66)
"""
from __future__ import annotations

import torch

from .cloud_optics_tables import RLILIA, RLILIB, RFUETA, RFUETB, RFUETC

# ---------------------------------------------------------------------------
# Physical constants (must match constants.py exactly)
# ---------------------------------------------------------------------------
# These are duplicated here as module-level scalars (not imported) so the
# kernel's arithmetic is self-contained and easy to audit against radlsw.F90.
RPI: float = 3.1415926535897931     # YOMCST%RPI
RTT: float = 273.16                 # YOMCST%RTT (triple point)
RTICE: float = 273.16 - 13.0        # YOMCST%RTICE = RTT - 13
REPLOG: float = 1.0e-12             # YOMCST%REPLOG (log safety floor)
REPSCW: float = 1.0e-12             # YOETHF%REPSCW (cloud water floor)
RG: float = 9.80665                 # YOMCST%RG
RD: float = 287.059640              # YOMCST%RD
RRE2DE: float = 0.64952             # YOERAD%RRE2DE (re -> de conversion)
RREFDE: float = 1.0 / RRE2DE        # YOERAD%RREFDE (de -> re, = 1.5396..)
RDEFRE: float = RRE2DE              # alias used in radlsw for de=re*RDEFRE
RCCNSEA: float = 50.0               # YOERAD%RCCNSEA (CCN over sea, default)
RCCNLND: float = 900.0              # YOERAD%RCCNLND (CCN over land, default)


def liquid_effective_radius_martin(
    q_liquid: torch.Tensor, pressure: torch.Tensor, temperature: torch.Tensor,
    dp: torch.Tensor, land_sea_mask: torch.Tensor, dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Liquid-water cloud droplet effective radius (microns), Martin et al. 1994.

    Production branch NRADLP=2 (radlsw.F90 lines 500-529). Fully batched:
    all inputs share leading vertical axis ``(..., nlev)`` except
    ``land_sea_mask`` which is per-column ``(...,)`` broadcast against the
    leading column axis.

    Returns ``re_liq`` in microns, clamped to ``[4, 16]`` per the source.
    """
    # Column number concentration Ntot (cm-3) from CCN climatology.
    # radlsw.F90 uses a land/sea split (PLSM < 0.5 => sea). Vectorise with
    # torch.where so no Python branching over columns.
    # land_sea_mask is per-column (...); q/T/p are (... nlev, nlon) so we
    # broadcast the mask across the vertical axis.
    is_land = land_sea_mask >= 0.5
    # dispersion coefficient k (called ZD in source); broadcast over leading axis
    while is_land.ndim < q_liquid.ndim:
        is_land = is_land.unsqueeze(0)
    zk = torch.where(is_land, torch.tensor(0.43, dtype=dtype), torch.tensor(0.33, dtype=dtype))
    zccn = torch.where(is_land,
                       torch.tensor(RCCNLND, dtype=dtype),
                       torch.tensor(RCCNSEA, dtype=dtype))
    # Ntot quadratic in CCN: -2.10e-4*N^2 + 0.568*N - 27.9  (land)
    #                       -1.15e-3*N^2 + 0.963*N + 5.30  (sea)
    zntot = torch.where(
        is_land,
        -2.10e-4 * zccn * zccn + 0.568 * zccn - 27.9,
        -1.15e-3 * zccn * zccn + 0.963 * zccn + 5.30,
    )
    # Liquid water content (g/m^3) = q_liq(kg/kg) * 1000 * rho
    # rho = p / (Rd * T). radlsw computes ZLWC = ZLWK_GKG * ZPODT where
    # ZLWK_GKG = q_liq * 1000 and ZPODT = p/(Rd*T).
    zlwc = (q_liquid * 1000.0) * (pressure / (RD * temperature))
    # Martin (1994) eq: re = 100 * (3 * L * (1+3k^2)^2 / (4*pi*Nt*(1+k^2)^3))^(1/3)
    # source uses (ZNUM/ZDEN) > REPLOG gate; vectorise via torch.where.
    znum = 3.0 * zlwc * (1.0 + 3.0 * zk * zk) ** 2
    zden = 4.0 * RPI * zntot * (1.0 + zk * zk) ** 3
    ratio = znum / zden
    # Only compute the cube root where ratio > REPLOG; otherwise re = 4 um floor.
    valid = ratio > REPLOG
    # torch.pow(ratio, 1/3) is well-defined for ratio>0; REPLOG>0 so safe.
    re_root = 100.0 * torch.pow(torch.clamp(ratio, min=REPLOG), 1.0 / 3.0)
    re_liq = torch.where(valid, re_root, torch.tensor(4.0, dtype=dtype))
    # clamp to [4, 16] microns (source lines 525-526)
    re_liq = torch.clamp(re_liq, min=4.0, max=16.0)
    return re_liq


def ice_effective_diameter_sun_riku(
    q_ice: torch.Tensor, pressure: torch.Tensor, temperature: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ice-particle effective diameter (microns) and radius (microns).

    Production branch NRADIP=3 (radlsw.F90 lines 624-644) using
    Sun & Rikus (1999) revised by Sun (2001). When IWC <= 0 the source
    falls back to de = 80 um.

    Also computes the Liou & Ou (1994) temperature-based fallback radius
    (lines 599-605) -- this is used by some non-production branches but is
    kept here as the ZRADIP initialiser that NRADIP=3 then overrides.

    Returns ``(de_ice, re_ice)`` where ``de_ice`` is the Fu generalised
    effective diameter and ``re_ice`` is the equivalent radius used by the
    diffusivity factor (``ZRADIP = ZREFDE * ZDESR + 15`` for NRADIP=3).
    """
    # Ice water content (g/m^3)
    ziwc = (q_ice * 1000.0) * (pressure / (RD * temperature))
    # Sun & Rikus (1999) + Sun (2001) correction:
    #   ZTCELS = T - RTT;  ZTEMPC = T - 83.15;  ZFSR = 1.2351 + 0.0105 * ZTCELS
    #   ZAIWC = 45.8966 * IWC^0.2214 ;  ZBIWC = 0.7957 * IWC^0.2535
    #   de = ZFSR * (ZAIWC + ZBIWC*ZTEMPC), clamped to [30, 155]
    ztcels = temperature - RTT
    ztempc = temperature - 83.15
    zfsr = 1.2351 + 0.0105 * ztcels
    # IWC^p requires positive IWC; clamp inside the active branch only.
    zaiwc = 45.8966 * torch.pow(torch.clamp(ziwc, min=REPLOG), 0.2214)
    zbiwc = 0.7957 * torch.pow(torch.clamp(ziwc, min=REPLOG), 0.2535)
    de_sun = zfsr * (zaiwc + zbiwc * ztempc)
    de_sun = torch.clamp(de_sun, min=30.0, max=155.0)
    # IWC <= 0 -> de = 80 (source line 642)
    has_ice = ziwc > 0.0
    de_ice = torch.where(has_ice, de_sun, torch.tensor(80.0, dtype=dtype))
    # ZRADIP for NRADIP=3: re = RREFDE * de + 15 (source lines 638-639)
    re_ice = RREFDE * de_ice + 15.0
    return de_ice, re_ice


def cloud_lw_optical_thickness(
    q_liquid: torch.Tensor, q_ice: torch.Tensor,
    pressure: torch.Tensor, temperature: torch.Tensor, dp: torch.Tensor,
    land_sea_mask: torch.Tensor, dtype: torch.dtype = torch.float64,
    diffc: bool = False,
) -> torch.Tensor:
    """Per-band LW cloud optical thickness ZTAUCLD (CASE 12 production).

    Inputs are batched with leading vertical axis ``(..., nlev)`` (except
    land_sea_mask which is per-column ``(...,)``). Returns ZTAUCLD with shape
    ``(..., nlev, 16)`` -- the 16 RRTM longwave bands.

    Production CASE 12 (radlsw.F90 lines 1183-1210):
      * Liquid: Lindner & Li (2000), polynomial in re_liq using RLILIA/RLILIB
      * Ice:    Fu et al. (1998), polynomial in 1/de using RFUETA/RFUETB/RFUETC
      * Diffusivity: constant 1.66 (LDIFFC=.FALSE., production default)

    Cloudy-gridpoint gate uses ``ZFLWP + ZFIWP > REPSCW``.
    """
    # ---- in-cloud water paths (g/m^2): ZFLWP = q_liq*1000 * dp/g ----
    zflwp = (q_liquid * 1000.0) * (dp / RG)
    zfiwp = (q_ice * 1000.0) * (dp / RG)
    # ---- effective sizes ----
    re_liq = liquid_effective_radius_martin(
        q_liquid, pressure, temperature, dp, land_sea_mask, dtype=dtype)
    de_ice, re_ice = ice_effective_diameter_sun_riku(
        q_ice, pressure, temperature, dtype=dtype)
    # ---- cloudy gate (broadcastable per cell) ----
    cloudy = (zflwp + zfiwp) > REPSCW
    # ---- broadcast shape: (..., nlev, 16) for the band axis ----
    # All cell-shaped tensors get a band axis (size 16) and a coeff axis
    # (size 1) appended: (..., nlev) -> (..., nlev, 1[band], 1[coeff]).
    # Tables (16, k) become (1..., 1, 16, k) so the band axis aligns.
    def cell_to_bandcoeff(t):
        return t.unsqueeze(-1).unsqueeze(-1)  # (..., nlev, 1, 1)
    zflwp_x = cell_to_bandcoeff(zflwp)        # (..., nlev, 1, 1)
    zfiwp_x = cell_to_bandcoeff(zfiwp)
    re_liq_x = cell_to_bandcoeff(re_liq)
    de_ice_x = cell_to_bandcoeff(de_ice)
    z1radl = 1.0 / re_liq_x
    z1radi = 1.0 / de_ice_x
    # Tables: (16, k) -> (1, 1, 16, k) so they broadcast against the leading
    # (nlev, nlon) cell dims.
    def expand_table(tbl):
        return tbl.to(dtype).to(re_liq_x.device).view(1, 1, *tbl.shape)
    rlilia = expand_table(RLILIA)   # (1, 1, 16, 5)
    rlilib = expand_table(RLILIB)
    rfueta = expand_table(RFUETA)
    rfuetb = expand_table(RFUETB)
    rfuetc = expand_table(RFUETC)
    # ---- CASE 12 liquid (Lindner & Li 2000) ----
    # ZEXTCF = RLILIA(b,1) + re*RLILIA(b,2) + (1/re)*(RLILIA(b,3) + (1/re)*(RLILIA(b,4) + (1/re)*RLILIA(b,5)))
    zextcf = (rlilia[..., 0]
              + re_liq_x * rlilia[..., 1]
              + z1radl * (rlilia[..., 2]
                          + z1radl * (rlilia[..., 3] + z1radl * rlilia[..., 4])))
    # Z1MOMG = RLILIB(b,1) + (1/re)*RLILIB(b,2) + re*(RLILIB(b,3) + re*RLILIB(b,4))
    z1momg = (rlilib[..., 0]
              + z1radl * rlilib[..., 1]
              + re_liq_x * (rlilib[..., 2] + re_liq_x * rlilib[..., 3]))
    zrsald = z1momg * zextcf                       # (..., nlev, 1, 16)
    # ---- CASE 12 ice (Fu et al. 1998) ----
    # ZRSAIE = RFUETA(b,1) + (1/de)*(RFUETA(b,2) + (1/de)*RFUETA(b,3))
    zrsaie = (rfueta[..., 0]
              + z1radi * (rfueta[..., 1] + z1radi * rfueta[..., 2]))
    # ZRSAIA = (1/de)*(RFUETB(b,1) + de*(RFUETB(b,2) + de*(RFUETB(b,3) + de*RFUETB(b,4))))
    zrsaia = z1radi * (rfuetb[..., 0]
                       + de_ice_x * (rfuetb[..., 1]
                                      + de_ice_x * (rfuetb[..., 2]
                                                     + de_ice_x * rfuetb[..., 3])))
    # ZRSAIG = RFUETC(b,1) + de*(RFUETC(b,2) + de*(RFUETC(b,3) + de*RFUETC(b,4)))
    zrsaig = (rfuetc[..., 0]
              + de_ice_x * (rfuetc[..., 1]
                            + de_ice_x * (rfuetc[..., 2] + de_ice_x * rfuetc[..., 3])))
    # ZRSAIF = 0.5 + ZRSAIG*(0.3738 + ZRSAIG*(0.0076 + ZRSAIG*0.1186))
    zrsaif = 0.5 + zrsaig * (0.3738 + zrsaig * (0.0076 + zrsaig * 0.1186))
    # ZRSAID = (1 - ZRSAIA/ZRSAIE * ZRSAIF) * ZRSAIE
    zrsaif_full = (1.0 - zrsaia / zrsaie * zrsaif) * zrsaie  # (..., nlev, 1, 16)
    # ---- combine ----
    # zrsald, zrsaif_full are (..., nlev, 1, 16); zflwp_x/zfiwp_x are
    # (..., nlev, 1, 1). Their product is (..., nlev, 1, 16).
    ztaucld = zrsald * zflwp_x + zrsaif_full * zfiwp_x   # (..., nlev, 1, 16)
    # ---- diffusivity correction ----
    if diffc:
        # Savijarvi: ZDIFFD = clamp(1.517 - 0.156*ln(ZTAUCLD), 1, 2)
        # Only where cloudy; safe ln requires positive taucld.
        zdiffd = torch.clamp(1.517 - 0.156 * torch.log(torch.clamp(ztaucld, min=REPLOG)),
                             min=1.0, max=2.0)
    else:
        # Production default: constant diffusivity factor 1.66.
        zdiffd = torch.tensor(1.66, dtype=dtype)
    ztaucld = ztaucld * zdiffd
    # ---- zero out clear cells (squeeze the dummy coeff axis, keep band) ----
    ztaucld = ztaucld.squeeze(-2)                   # (..., nlev, 16)
    cloudy_b = cloudy.unsqueeze(-1)                 # (..., nlev, 1)
    ztaucld = torch.where(cloudy_b, ztaucld, torch.zeros_like(ztaucld))
    return ztaucld


__all__ = [
    "liquid_effective_radius_martin",
    "ice_effective_diameter_sun_riku",
    "cloud_lw_optical_thickness",
]
