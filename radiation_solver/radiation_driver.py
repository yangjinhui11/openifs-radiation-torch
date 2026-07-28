"""Radiation driver: frequency-controlled full/cheap radiation switching.

Mirrors the gp_model.F90 radiation architecture:
  * Every NRADFR steps: full radiation call (radina → RADLSW)
    → stores transmissivities/emissivities in RadHeatState
  * Other steps: cheap interpolation (radheatn)
    → uses stored RadHeatState + current T/Q/mu0

This is the radiation equivalent of Fortran's:
  §5: RADDRV → RADINTG → RADLSW     (full, NRADFR-frequency)
  §6: RADFLUX_LAYER → RADHEATN       (every step)

Convention: all arrays (nlon, nlev) top-down (TOA=0).
"""
from __future__ import annotations
import torch
from dataclasses import dataclass

from .radheatn import RadHeatState, RadHeatOutput, radheatn

_RII0 = 1361.0
_RRAE = 0.637
_RSIGMA = 5.670374e-8


@dataclass
class RadiationConfig:
    """Radiation frequency control."""
    nradfr: int = 4           # full radiation every N steps (default: hourly @ 900s)
    approxlw_update: bool = True    # LAPPROXLWUPDATE
    manners_sw_update: bool = True  # LMANNERSSWUPDATE


class RadiationDriver:
    """Manages the full/cheap radiation switching like gp_model.F90.

    Call ``step()`` every physics timestep. It internally decides whether
    to run the full radiation scheme (radina) or the cheap interpolation
    (radheatn), and maintains the RadHeatState between calls.
    """

    def __init__(self, config: RadiationConfig | None = None,
                 rii0: float = _RII0):
        self.config = config or RadiationConfig()
        self.rii0 = float(rii0)
        self._step_count = 0
        self._rad_state: RadHeatState | None = None
        self._last_radina_output = None  # cache for diagnostics

    def step(self, radina_fn, state_builder, paphm1, pqm1, ptsm1m, pmu0,
             pte=None):
        """Advance radiation by one physics timestep.

        Args:
            radina_fn: callable that runs full radiation (radina(state) → RadinaOutput).
                       Called only every NRADFR steps.
            state_builder: callable that builds RadinaState from current atmosphere.
                           Called only when radina_fn is called.
            paphm1: (nlon, nlev+1) half-level pressure [Pa], top-down.
            pqm1: (nlon, nlev) specific humidity, top-down.
            ptsm1m: (nlon,) skin temperature [K].
            pmu0: (nlon,) current cos(solar zenith angle).
            pte: (nlon, nlev) temperature tendency to update (K/s), or None.

        Returns:
            RadHeatOutput with fluxes and heating rates for this step.
        """
        self._step_count += 1
        do_full = ((self._step_count - 1) % self.config.nradfr) == 0

        if do_full or self._rad_state is None:
            # ── Full radiation call (like gp_model §5: RADDRV → RADINTG → RADLSW) ──
            radina_state = state_builder()
            radina_out = radina_fn(radina_state)
            self._last_radina_output = radina_out

            # Convert radina output → RadHeatState (transmissivities)
            self._rad_state = self._build_rad_heat_state(radina_out, pmu0, paphm1)

        # ── Cheap radiation interpolation (like gp_model §6: RADHEATN) ──
        out = radheatn(
            state=self._rad_state,
            paphm1=paphm1,
            pqm1=pqm1,
            ptsm1m=ptsm1m,
            pmu0=pmu0,
            pte=pte,
            approxlw_update=self.config.approxlw_update,
            manners_sw_update=self.config.manners_sw_update,
            rii0=self.rii0,
        )
        return out

    def _build_rad_heat_state(self, radina_out, pmu0: torch.Tensor,
                              paphm1: torch.Tensor) -> RadHeatState:
        """Convert RadinaOutput → RadHeatState.

        RadinaOutput has fluxes (W/m2); RadHeatState stores transmissivities
        (dimensionless) and LW net fluxes (W/m2).

        Key conversions:
          * SW transmissivity = SW_flux / (RII0 * mu0_radiation)
          * LW net flux = flux_down - flux_up (but radina only gives flux_down;
            we approximate net = -(flux_down - surface_emission))
        """
        dt = pmu0.dtype
        dev = pmu0.device
        nlon = pmu0.shape[0]
        nlev1 = radina_out.pemtd.shape[1]
        nlev = nlev1 - 1

        # Solar incident at radiation time
        mu0_rad = pmu0.clamp(min=1e-6)
        zi0_rad = self.rii0 * mu0_rad

        # ── SW transmissivities ──
        # ptrsol = SW_down_flux / ZI0  (transmissivity = fraction of TOA incident)
        ptrsol = radina_out.ptrso / zi0_rad.unsqueeze(1)
        ptrsoc = radina_out.ptrsoc / zi0_rad.unsqueeze(1)
        # Surface values
        ptrsod = ptrsol[:, -1].clone()
        ptrsodc = ptrsoc[:, -1].clone()

        # Direct beam transmissivity at surface (approximate: total / diffuse ratio)
        # In full RADLSW, this comes from ZDIRFS/ZDIFFS. Here we approximate
        # PFDIRI as a fraction of total surface transmissivity.
        pfdiri = (ptrsod * 0.7).clamp(min=1e-10)
        pcdiri = (ptrsodc * 0.75).clamp(min=1e-10)

        # ── LW net flux ──
        # RADHEATN expects PEMTD = net LW flux (positive=downward).
        # radina gives pemtd = downward LW flux.
        # Net LW = downward - upward. We approximate:
        #   upward ≈ emissivity * sigma * T_skin^4 + (1-emissivity) * downward
        # So net ≈ downward - upward (negative at most levels = net cooling).
        # However, RADLSW stores the actual net flux; since we only have
        # downward, we use a simplified approach: store downward as proxy.
        # In production, radina should be extended to also output flux_up.
        pemtd = radina_out.pemtd.clone()  # downward LW flux
        pemtec = radina_out.pemtc.clone() if hasattr(radina_out, 'pemtc') else pemtd.clone() * 0.95

        # Surface LW downwelling
        ptrthd = radina_out.pfrted.clone() if hasattr(radina_out, 'pfrted') else pemtd[:, -1].clone()
        ptrthdc = ptrthd * 0.95

        # LW derivative: d(LW_up)/d(T_surf) at each half-level.
        # In RADLSW this is PLWDERIVATIVE. We approximate with a profile
        # that decreases from ~0.8 at surface to ~0 at TOA.
        lwderiv = torch.zeros(nlon, nlev1, dtype=dt, device=dev)
        profile = torch.linspace(0.0, 0.8, nlev1, dtype=dt, device=dev)
        lwderiv[:] = profile.unsqueeze(0)

        # Surface diagnostics
        puvdfi = torch.full((nlon,), 0.05, dtype=dt, device=dev)
        pparfi = ptrsod * 0.4
        pparcfi = ptrsodc * 0.42
        ptincfi = ptrsod * 0.5
        pemis = radina_out.pemit.clone() if hasattr(radina_out, 'pemit') else torch.full((nlon,), 0.98, dtype=dt, device=dev)

        return RadHeatState(
            ptrsol=ptrsol, ptrsoc=ptrsoc, ptrsod=ptrsod, ptrsodc=ptrsodc,
            pfdiri=pfdiri, pcdiri=pcdiri,
            pemtd=pemtd, pemtec=pemtec, ptrthd=ptrthd, ptrthdc=ptrthdc,
            plwderivative=lwderiv, pmu0m=mu0_rad.clone(),
            puvdfi=puvdfi, pparfi=pparfi, pparcfi=pparcfi, ptincfi=ptincfi,
            pemis=pemis,
        )

    @property
    def rad_state(self) -> RadHeatState | None:
        """Access the current stored radiation state (for debugging)."""
        return self._rad_state

    @property
    def last_full_rad_output(self):
        """The last RadinaOutput from a full radiation call."""
        return self._last_radina_output


__all__ = ["RadiationDriver", "RadiationConfig"]
