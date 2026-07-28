"""Loader for the RADLSW dedicated radiation dump.

Reads the binary + JSON dump written by ``RADLSW_DUMP`` in ``radlsw.F90``
and returns structured dicts of LW/SW inputs and outputs ready for
comparison with the torch port.

Binary format:
  - ACCESS='STREAM' (no Fortran record markers)
  - All data is native-endian (little-endian on x86) float64/int32
  - Written sequentially as described in the companion JSON metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .radlsw_inputs import RadLSWInputs, assemble_radlsw_inputs


# ── binary reader ────────────────────────────────────────────────────────────

class _StreamReader:
    """Sequential reader for Fortran STREAM-access binary."""
    def __init__(self, path: Path, big_endian: bool = False):
        self.raw = path.read_bytes()
        self.off = 0
        self.be = big_endian
        dt_i4 = ">i4" if big_endian else "<i4"
        dt_f8 = ">f8" if big_endian else "<f8"
        self._dt_i4 = np.dtype(dt_i4)
        self._dt_f8 = np.dtype(dt_f8)

    def _read(self, dtype, n: int):
        arr = np.frombuffer(self.raw, dtype=dtype, count=n, offset=self.off)
        self.off += n * dtype.itemsize
        return arr.copy()

    def i32(self, n: int = 1) -> np.ndarray:
        return self._read(self._dt_i4, n)

    def f64(self, shape) -> np.ndarray:
        n = int(np.prod(shape))
        arr = self._read(self._dt_f8, n)
        return arr.reshape(shape).copy()

    def f64_scalar(self) -> float:
        return float(self._read(self._dt_f8, 1)[0])


# ── result structs ───────────────────────────────────────────────────────────

@dataclass
class RadLSWDump:
    """All fields from a RADLSW_DUMP snapshot."""
    # metadata
    magic: int
    nstep: int
    klon: int
    klev: int
    nsw: int
    jpband: int
    binary_file: str
    endianness: str
    vertical_convention: str

    # ── LW inputs (from KWHEN=0) ──
    # All arrays on CPU, float64.  Vertical convention: TOA at index 0.
    paph: np.ndarray          # (klon, klev+1)  half-level pressure [Pa]
    pap: np.ndarray           # (klon, klev)    full-level pressure [Pa]
    pdp: np.ndarray           # (klon, klev)    layer thickness [Pa]
    pth: np.ndarray           # (klon, klev+1)  half-level temperature [K]
    pt: np.ndarray            # (klon, klev)    full-level temperature [K]
    pts: np.ndarray           # (klon,)         surface/skin temperature [K]
    pq: np.ndarray            # (klon, klev)    specific humidity [kg/kg]
    co2: np.ndarray           # (klon, klev)    CO₂ mmr
    ch4: np.ndarray           # (klon, klev)    CH₄ mmr
    n2o: np.ndarray           # (klon, klev)    N₂O mmr
    no2: np.ndarray           # (klon, klev)    NO₂ mmr
    c11: np.ndarray           # (klon, klev)    CFC-11 mmr
    c12: np.ndarray           # (klon, klev)    CFC-12 mmr
    c22: np.ndarray           # (klon, klev)    CFC-22 mmr
    cl4: np.ndarray           # (klon, klev)    CCl₄ mmr
    ozn: np.ndarray           # (klon, klev)    O₃ mmr [= POZON/PDP]
    cldf: np.ndarray          # (klon, klev)    cloud fraction [0-1]
    taucld: np.ndarray        # (klon, klev, jpband)  cloud optical depth
    emis: np.ndarray          # (klon,)         surface LW emissivity
    emiw: np.ndarray          # (klon,)         surface LW window emissivity
    aer: np.ndarray           # (klon, 6, klev)  aerosol optical depth
    albd: np.ndarray          # (klon, nsw)     surface SW albedo (diffuse)
    albp: np.ndarray          # (klon, nsw)     surface SW albedo (parallel)
    mu0: np.ndarray           # (klon,)         cos(solar zenith angle)
    ri0: float                # scalar          solar constant [W/m²]
    psol: np.ndarray          # (klon,)         surface pressure [Pa]
    dt0: np.ndarray           # (klon,)         skin-air temperature jump [K]

    # ── LW outputs (from KWHEN=1) ──
    lw_flux: np.ndarray = field(default_factory=lambda: np.zeros((1,2,1)))  # (klon, 2, klev+1)
    lw_fluc: np.ndarray = field(default_factory=lambda: np.zeros((1,2,1)))  # (klon, 2, klev+1)
    lw_emit: np.ndarray = field(default_factory=lambda: np.zeros(1))         # (klon,)
    lw_tclear: np.ndarray = field(default_factory=lambda: np.zeros(1))       # (klon,)

    # ── SW inputs (from KWHEN=2) ──
    sw_tau: np.ndarray = field(default_factory=lambda: np.zeros((1,1,1)))    # (klon, nsw, klev)
    sw_omega: np.ndarray = field(default_factory=lambda: np.zeros((1,1,1)))  # (klon, nsw, klev)
    sw_cg: np.ndarray = field(default_factory=lambda: np.zeros((1,1,1)))     # (klon, nsw, klev)
    sw_pmb: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))      # (klon, klev+1)
    sw_tave: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))     # (klon, klev)
    sw_qsat: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))     # (klon, klev)

    # ── SW outputs (from KWHEN=3) ──
    sw_fsdwn: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))    # (klon, klev+1)
    sw_fsup: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))
    sw_fcdwn: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))
    sw_fcup: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))
    sw_fsdnn: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sw_fsdnv: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sw_fsupn: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sw_fsupv: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sw_diffs: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))
    sw_dirfs: np.ndarray = field(default_factory=lambda: np.zeros((1,1)))
    sw_parf: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sw_uvdf: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sw_parcf: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sw_sudu: np.ndarray = field(default_factory=lambda: np.zeros(1))


# ── main loader ──────────────────────────────────────────────────────────────

def load_radlsw_dump(
    bin_path: str | Path,
    json_path: Optional[str | Path] = None,
) -> RadLSWDump:
    """Load a RADLSW dump from the binary + companion JSON.

    Parameters
    ----------
    bin_path : pathlib.Path or str
        Path to the ``radlsw_dump_step<N>.bin`` file.
    json_path : pathlib.Path or str, optional
        Path to ``radlsw_dump_meta.json``.  Defaults to the same directory
        as the binary.

    Returns
    -------
    RadLSWDump
    """
    bin_path = Path(bin_path)
    if json_path is None:
        json_path = bin_path.parent / "radlsw_dump_meta.json"
    else:
        json_path = Path(json_path)

    meta = json.loads(json_path.read_text())

    # Detect endianness
    big_endian = meta.get("endianness", "little") == "big"
    r = _StreamReader(bin_path, big_endian=big_endian)

    klon = meta["klon"]
    klev = meta["klev"]
    nsw = meta["nsw"]
    jpband = meta["jpband"]

    # ── read LW inputs (KWHEN=0) ──
    magic = int(r.i32(1)[0])
    dims_read = r.i32(4)
    nstep = int(r.i32(1)[0])

    assert magic == meta["magic"], f"Magic mismatch: {magic} != {meta['magic']}"
    assert dims_read[0] == klon and dims_read[1] == klev, \
        f"Dims mismatch: {dims_read} != ({klon},{klev})"
    assert dims_read[2] == nsw and dims_read[3] == jpband, \
        f"NSW/JPBAND mismatch: {dims_read[2:]} != ({nsw},{jpband})"

    d = RadLSWDump(
        magic=magic, nstep=nstep, klon=klon, klev=klev,
        nsw=nsw, jpband=jpband,
        binary_file=str(bin_path),
        endianness=meta["endianness"],
        vertical_convention=meta["vertical_convention"],

        # LW inputs
        paph=r.f64((klon, klev+1)),
        pap=r.f64((klon, klev)),
        pdp=r.f64((klon, klev)),
        pth=r.f64((klon, klev+1)),
        pt=r.f64((klon, klev)),
        pts=r.f64((klon,)),
        pq=r.f64((klon, klev)),
        co2=r.f64((klon, klev)),
        ch4=r.f64((klon, klev)),
        n2o=r.f64((klon, klev)),
        no2=r.f64((klon, klev)),
        c11=r.f64((klon, klev)),
        c12=r.f64((klon, klev)),
        c22=r.f64((klon, klev)),
        cl4=r.f64((klon, klev)),
        ozn=r.f64((klon, klev)),
        cldf=r.f64((klon, klev)),
        taucld=r.f64((klon, klev, jpband)),
        emis=r.f64((klon,)),
        emiw=r.f64((klon,)),
        aer=r.f64((klon, 6, klev)),
        albd=r.f64((klon, nsw)),
        albp=r.f64((klon, nsw)),
        mu0=r.f64((klon,)),
        ri0=r.f64_scalar(),
        psol=r.f64((klon,)),
        dt0=r.f64((klon,)),
    )

    # ── read LW outputs (KWHEN=1) ──
    d.lw_flux = r.f64((klon, 2, klev+1))
    d.lw_fluc = r.f64((klon, 2, klev+1))
    d.lw_emit = r.f64((klon,))
    d.lw_tclear = r.f64((klon,))

    # ── read SW inputs (KWHEN=2) ──
    d.sw_tau = r.f64((klon, nsw, klev))
    d.sw_omega = r.f64((klon, nsw, klev))
    d.sw_cg = r.f64((klon, nsw, klev))
    d.sw_pmb = r.f64((klon, klev+1))
    d.sw_tave = r.f64((klon, klev))
    d.sw_qsat = r.f64((klon, klev))

    # ── read SW outputs (KWHEN=3) ──
    d.sw_fsdwn = r.f64((klon, klev+1))
    d.sw_fsup = r.f64((klon, klev+1))
    d.sw_fcdwn = r.f64((klon, klev+1))
    d.sw_fcup = r.f64((klon, klev+1))
    d.sw_fsdnn = r.f64((klon,))
    d.sw_fsdnv = r.f64((klon,))
    d.sw_fsupn = r.f64((klon,))
    d.sw_fsupv = r.f64((klon,))
    d.sw_diffs = r.f64((klon, nsw))
    d.sw_dirfs = r.f64((klon, nsw))
    d.sw_parf = r.f64((klon,))
    d.sw_uvdf = r.f64((klon,))
    d.sw_parcf = r.f64((klon,))
    d.sw_sudu = r.f64((klon,))

    return d


# ── converter: dump → RadLSWInputs ───────────────────────────────────────────

def dump_to_radlsw_inputs(dump: RadLSWDump) -> RadLSWInputs:
    """Convert a RadLSWDump to a RadLSWInputs for the torch LW chain.

    The dump stores arrays as (klon, klev) with TOA at index 0, matching
    the Fortran convention.  RadLSWInputs expects (nlev, nlon) with TOA
    at index 0 (the ``pavel`` in ECRT will be flipped to bottom-up).
    """
    def t(arr: np.ndarray) -> torch.Tensor:
        """(nlon, nlev) → (nlev, nlon) via contiguous transpose."""
        return torch.from_numpy(arr.copy()).to(torch.float64).permute(1, 0).contiguous()

    def t1(arr: np.ndarray) -> torch.Tensor:
        """1-D array: keep as-is."""
        return torch.from_numpy(arr.copy()).to(torch.float64)

    # Aerosols: dump is (nlon, 6, nlev) → need (nlev, nlon, 6)
    aer_t = torch.from_numpy(dump.aer.copy()).to(torch.float64).permute(2, 0, 1).contiguous()

    # Cloud optics: dump is (nlon, nlev, jpband) → need (nlev, nlon, jpband)
    tau_lw_t = torch.from_numpy(dump.taucld.copy()).to(torch.float64).permute(1, 0, 2).contiguous()

    return RadLSWInputs(
        # Surface
        emis=t1(dump.emis),
        emiw=t1(dump.emiw),
        albd=t1(dump.albd),
        albp=t1(dump.albp),
        pts=t1(dump.pts),

        # Per-layer (nlon, nlev) → (nlev, nlon)
        cloud_fraction=t(dump.cldf),
        pap=t(dump.pap),
        paph=t(dump.paph),
        pdp=t(dump.pdp),
        pt=t(dump.pt),
        pth=t(dump.pth),
        pq=t(dump.pq),

        # Gas VMRs (nlon, nlev) → (nlev, nlon)
        co2=t(dump.co2),
        ch4=t(dump.ch4),
        n2o=t(dump.n2o),
        no2=t(dump.no2),
        cfc11=t(dump.c11),
        cfc12=t(dump.c12),
        cfc22=t(dump.c22),
        ccl4=t(dump.cl4),
        o3_mmr=t(dump.ozn),

        # Aerosols and cloud optics
        aer=aer_t,
        tau_lw=tau_lw_t,
        sw_optics=None,
    )


__all__ = [
    "load_radlsw_dump",
    "dump_to_radlsw_inputs",
    "RadLSWDump",
    "RadLSWInputs",
]
