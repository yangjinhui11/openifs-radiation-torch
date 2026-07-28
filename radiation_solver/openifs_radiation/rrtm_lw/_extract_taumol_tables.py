"""Extract RRTM band tables from librad_ref.so → .npy files.

After ``rad_ref_init()`` populates the YOERRTA1..16 module-level arrays
(ABSA, ABSB, SELFREF, FORREF, FRACREFA/B, minor-gas arrays), this script
reads them via ctypes symbol lookup and saves each as a .npy in the
``tables/`` directory next to this file.

Usage::

    cd openifs_radiation_pytorch
    python openifs_radiation/rrtm_lw/_extract_taumol_tables.py
    python openifs_radiation/rrtm_lw/_extract_taumol_tables.py --check  # verify only
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TABLE_DIR = _HERE / "tables"
_REF_DIR = _HERE.parent.parent / "offline_ref"
_LIB_PATH = Path(os.environ.get("RAD_REF_SO", _REF_DIR / "librad_ref.so"))

# --------------------------------------------------------------------------
# Band definitions: (band_number, ng, nspa, nspb, has_upper_absb, extra_arrays)
# nspa = NSPA(band) from YOERRTWN — number of ref atmospheres for lower atm
# nspb = NSPB(band) — number for upper atm
# --------------------------------------------------------------------------
# These match YOERRTWN and the taumol source files.
BAND_SPECS = {
    1:  {"ng": 10, "nspa": 1, "nspb": 1,
         "ka_shape": (5, 13, 10), "absa_shape": (65, 10),
         "kb_shape": (5, 47, 10), "absb_shape": (235, 10),
         "extra": ["ka_mn2", "kb_mn2"], "extra_shapes": {"ka_mn2": (19, 10), "kb_mn2": (19, 10)},
         "fracrefa_shape": (10,), "fracrefb_shape": (10,)},
    2:  {"ng": 12, "nspa": 1, "nspb": 1,
         "ka_shape": (5, 13, 12), "absa_shape": (65, 12),
         "kb_shape": (5, 47, 12), "absb_shape": (235, 12),
         "fracrefa_shape": (12,), "fracrefb_shape": (12,)},
    3:  {"ng": 16, "nspa": 9, "nspb": 5,
         "ka_shape": (9, 5, 13, 16), "absa_shape": (585, 16),
         "kb_shape": (5, 5, 47, 16), "absb_shape": (1175, 16),
         "extra": ["ka_mn2o", "kb_mn2o"],
         "extra_shapes": {"ka_mn2o": (9, 19, 16), "kb_mn2o": (5, 19, 16)},
         "fracrefa_shape": (16, 9), "fracrefb_shape": (16, 5)},
    4:  {"ng": 14, "nspa": 9, "nspb": 5,
         "ka_shape": (9, 5, 13, 14), "absa_shape": (585, 14),
         "kb_shape": (5, 5, 47, 14), "absb_shape": (1175, 14),
         "fracrefa_shape": (14, 9), "fracrefb_shape": (14, 5)},
    5:  {"ng": 16, "nspa": 9, "nspb": 5,
         "ka_shape": (9, 5, 13, 16), "absa_shape": (585, 16),
         "kb_shape": (5, 5, 47, 16), "absb_shape": (1175, 16),
         "extra": ["ka_mo3", "ccl4"],
         "extra_shapes": {"ka_mo3": (9, 19, 16), "ccl4": (16,)},
         "fracrefa_shape": (16, 9), "fracrefb_shape": (16, 5)},
    6:  {"ng": 8, "nspa": 1, "nspb": 1,
         "ka_shape": (5, 13, 8), "absa_shape": (65, 8),
         "extra": ["ka_mco2", "cfc11adj", "cfc12"],
         "extra_shapes": {"ka_mco2": (19, 8), "cfc11adj": (8,), "cfc12": (8,)},
         "fracrefa_shape": (8,)},
    7:  {"ng": 12, "nspa": 9, "nspb": 1,
         "ka_shape": (9, 5, 13, 12), "absa_shape": (585, 12),
         "kb_shape": (5, 47, 12), "absb_shape": (235, 12),
         "extra": ["ka_mco2", "kb_mco2"],
         "extra_shapes": {"ka_mco2": (9, 19, 12), "kb_mco2": (19, 12)},
         "fracrefa_shape": (12, 9), "fracrefb_shape": (12,)},
    8:  {"ng": 8, "nspa": 1, "nspb": 1,
         "ka_shape": (5, 13, 8), "absa_shape": (65, 8),
         "kb_shape": (5, 47, 8), "absb_shape": (235, 8),
         "extra": ["ka_mco2", "ka_mn2o", "ka_mo3", "kb_mco2", "kb_mn2o",
                    "cfc12", "cfc22adj"],
         "extra_shapes": {"ka_mco2": (19, 8), "ka_mn2o": (19, 8), "ka_mo3": (19, 8),
                          "kb_mco2": (19, 8), "kb_mn2o": (19, 8),
                          "cfc12": (8,), "cfc22adj": (8,)},
         "fracrefa_shape": (8,), "fracrefb_shape": (8,)},
    9:  {"ng": 12, "nspa": 9, "nspb": 1,
         "ka_shape": (9, 5, 13, 12), "absa_shape": (585, 12),
         "kb_shape": (5, 47, 12), "absb_shape": (235, 12),
         "extra": ["ka_mn2o", "kb_mn2o"],
         "extra_shapes": {"ka_mn2o": (9, 19, 12), "kb_mn2o": (19, 12)},
         "fracrefa_shape": (12, 9), "fracrefb_shape": (12,)},
    10: {"ng": 6,  "nspa": 1, "nspb": 1,
         "ka_shape": (5, 13, 6), "absa_shape": (65, 6),
         "kb_shape": (5, 47, 6), "absb_shape": (235, 6),
         "fracrefa_shape": (6,), "fracrefb_shape": (6,)},
    11: {"ng": 8,  "nspa": 1, "nspb": 1,
         "ka_shape": (5, 13, 8), "absa_shape": (65, 8),
         "kb_shape": (5, 47, 8), "absb_shape": (235, 8),
         "extra": ["ka_mo2", "kb_mo2"],
         "extra_shapes": {"ka_mo2": (19, 8), "kb_mo2": (19, 8)},
         "fracrefa_shape": (8,), "fracrefb_shape": (8,)},
    12: {"ng": 8,  "nspa": 9, "nspb": 1,
         "ka_shape": (9, 5, 13, 8), "absa_shape": (585, 8),
         "fracrefa_shape": (8, 9), "strrat": True},
    13: {"ng": 4,  "nspa": 9, "nspb": 1,
         "ka_shape": (9, 5, 13, 4), "absa_shape": (585, 4),
         "extra": ["ka_mco2", "ka_mco", "kb_mo3"],
         "extra_shapes": {"ka_mco2": (9, 19, 4), "ka_mco": (9, 19, 4), "kb_mo3": (19, 4)},
         "fracrefa_shape": (4, 9), "fracrefb_shape": (4,)},
    14: {"ng": 2,  "nspa": 1, "nspb": 1,
         "ka_shape": (5, 13, 2), "absa_shape": (65, 2),
         "kb_shape": (5, 47, 2), "absb_shape": (235, 2),
         "fracrefa_shape": (2,), "fracrefb_shape": (2,)},
    15: {"ng": 2,  "nspa": 9, "nspb": 1,
         "ka_shape": (9, 5, 13, 2), "absa_shape": (585, 2),
         "extra": ["ka_mn2"],
         "extra_shapes": {"ka_mn2": (9, 19, 2)},
         "fracrefa_shape": (2, 9)},
    16: {"ng": 2,  "nspa": 9, "nspb": 1,
         "ka_shape": (9, 5, 13, 2), "absa_shape": (585, 2),
         "kb_shape": (5, 47, 2), "absb_shape": (235, 2),
         "fracrefa_shape": (2, 9), "fracrefb_shape": (2,)},
}

# Common arrays present in every band (shape depends on band).
COMMON_ARRAYS = ["selfref", "forref"]
COMMON_SHAPES = {"selfref": (10,), "forref": (4,)}


def _load_lib():
    """Load the reference .so and call init."""
    if not _LIB_PATH.exists():
        raise FileNotFoundError(
            f"{_LIB_PATH} not found; run offline_ref/build_ref.sh first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.rad_ref_init.argtypes = []
    lib.rad_ref_init.restype = ctypes.c_int
    ret = lib.rad_ref_init()
    if ret != 0:
        raise RuntimeError(f"rad_ref_init returned {ret}")
    return lib


def _read_array(lib: ctypes.CDLL, symbol_name: str, shape: tuple[int, ...],
                dtype: type = np.float64) -> np.ndarray:
    """Read a module-level array from the .so by its exported symbol name."""
    try:
        addr = ctypes.c_void_p.in_dll(lib, symbol_name)
    except (ValueError, AttributeError) as e:
        raise RuntimeError(
            f"Symbol {symbol_name} not found in {_LIB_PATH}: {e}"
        ) from e
    size = int(np.prod(shape))
    arr = np.ctypeslib.as_array(
        (ctypes.c_double * size).from_address(ctypes.addressof(addr))
    ).copy()
    return arr.reshape(shape, order='F')


def _get_eq_symbol_sizes(lib_path: str) -> dict:
    """Parse nm -S output to get sizes of .eq symbols.
    
    Returns dict: {(band, eq_idx): size_in_bytes}
    """
    import subprocess
    result = subprocess.run(
        ["nm", "-S", lib_path],
        capture_output=True, text=True
    )
    sizes = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and '.eq.' in parts[3]:
            # Format: address size type name
            try:
                sym = parts[3]
                size_hex = parts[1]
                size_bytes = int(size_hex, 16)
                # Parse band and eq_idx from symbol name
                # Example: yoerrta1.eq.0_
                sym_clean = sym.rstrip('_')
                parts_sym = sym_clean.split('.')
                if len(parts_sym) >= 3:
                    band_str = parts_sym[0].replace('yoerrta', '').replace('yoerrto', '')
                    eq_str = parts_sym[2]
                    if band_str.isdigit() and eq_str.isdigit():
                        band = int(band_str)
                        eq_idx = int(eq_str)
                        sizes[(band, eq_idx)] = size_bytes
            except (ValueError, IndexError):
                continue
    return sizes


def _read_equivalence_array(lib: ctypes.CDLL, eq_idx: int, band: int,
                            flat_shape: tuple[int, ...]) -> np.ndarray:
    """Read an EQUIVALENCE-d array (ABSA or ABSB) from the .so.
    
    gfortran emits symbols like ``yoerrta1.eq.0_`` for EQUIVALENCEd
    blocks.  The mapping .eq.0→ABSA / .eq.1→ABSB is NOT guaranteed;
    it depends on declaration order inside the Fortran module.
    Callers should use _resolve_eq_mapping() first.
    """
    symbol = f"yoerrta{band}.eq.{eq_idx}_"
    try:
        addr = ctypes.c_void_p.in_dll(lib, symbol)
    except (ValueError, AttributeError):
        return None
    size = int(np.prod(flat_shape))
    arr = np.ctypeslib.as_array(
        (ctypes.c_double * size).from_address(ctypes.addressof(addr))
    ).copy()
    return arr.reshape(flat_shape, order='F')


def _resolve_eq_mapping(lib_path: str, band: int,
                        absa_shape: tuple, absb_shape: tuple | None
                        ) -> tuple[int | None, int | None]:
    """Determine which .eq symbol is ABSA and which is ABSB.
    
    Returns (eq_idx_for_absa, eq_idx_for_absb) based on matching 
    the symbol's allocated BSS size with the expected array size.
    """
    sizes = _get_eq_symbol_sizes(lib_path)
    
    absa_bytes = int(np.prod(absa_shape)) * 8
    absb_bytes = int(np.prod(absb_shape)) * 8 if absb_shape else 0
    
    best_absa = None
    best_absb = None
    
    for eq_idx in [0, 1]:
        key = (band, eq_idx)
        if key not in sizes:
            continue
        sym_bytes = sizes[key]
        
        # Match based on size: the symbol whose size equals absa_bytes is ABSA
        if absa_bytes == sym_bytes:
            best_absa = eq_idx
        elif absb_shape and absb_bytes == sym_bytes:
            best_absb = eq_idx
    
    # If exact match not found (e.g., equivalence block includes both arrays),
    # fall back to size comparison: ABSB is typically larger than ABSA
    # because it covers more reference pressure levels (47 vs 13).
    if best_absa is None:
        # Try to find the smaller one (ABSA) and larger one (ABSB)
        eq0_size = sizes.get((band, 0), 0)
        eq1_size = sizes.get((band, 1), 0)
        if eq0_size > 0 and eq1_size > 0:
            if eq0_size <= eq1_size:
                best_absa, best_absb = 0, 1
            else:
                best_absa, best_absb = 1, 0
        elif eq0_size > 0:
            best_absa = 0
            best_absb = None
        elif eq1_size > 0:
            best_absa = 1
            best_absb = None
    
    return best_absa, best_absb


def _make_symbol(band: int, array_name: str) -> str:
    """gfortran module variable mangling: ``__yoerrtaN_MOD_varname``."""
    return f"__yoerrta{band}_MOD_{array_name}"


def extract_band(lib: ctypes.CDLL, band: int, check: bool = False) -> dict:
    """Extract all tables for a single band. Returns dict of array_name → np.ndarray."""
    spec = BAND_SPECS[band]
    ng = spec["ng"]
    tables = {}

    # FRACREFA, FRACREFB
    fracrefa = _read_array(lib, _make_symbol(band, "fracrefa"),
                           spec["fracrefa_shape"])
    tables["fracrefa"] = fracrefa

    if "fracrefb_shape" in spec:
        fracrefb = _read_array(lib, _make_symbol(band, "fracrefb"),
                               spec["fracrefb_shape"])
        tables["fracrefb"] = fracrefb

    # SELFREF, FORREF (common, always (10, ng) and (4, ng))
    selfref = _read_array(lib, _make_symbol(band, "selfref"), (10, ng))
    forref = _read_array(lib, _make_symbol(band, "forref"), (4, ng))
    tables["selfref"] = selfref
    tables["forref"] = forref

    # Resolve which .eq symbol is ABSA and which is ABSB
    absb_shape_spec = spec.get("absb_shape")
    eq_absa, eq_absb = _resolve_eq_mapping(
        str(_LIB_PATH), band, spec["absa_shape"], absb_shape_spec
    )
    
    # ABSA via its resolved EQUIVALENCE symbol
    if eq_absa is not None:
        absa = _read_equivalence_array(lib, eq_absa, band, spec["absa_shape"])
        if absa is not None:
            tables["absa"] = absa
    
    # ABSB via its resolved EQUIVALENCE symbol (bands with upper atmosphere)
    if absb_shape_spec is not None and eq_absb is not None:
        absb = _read_equivalence_array(lib, eq_absb, band, absb_shape_spec)
        if absb is not None:
            tables["absb"] = absb

    # Extra band-specific arrays (minor gas, cross-section)
    for extra_name in spec.get("extra", []):
        shape = spec["extra_shapes"][extra_name]
        arr = _read_array(lib, _make_symbol(band, extra_name), shape)
        tables[extra_name] = arr

    # STRRAT for band 12
    if spec.get("strrat"):
        strrat = _read_array(lib, _make_symbol(band, "strrat"), ())
        tables["strrat"] = strrat

    if check:
        _check_tables(band, tables)

    return tables


def _check_tables(band: int, tables: dict):
    """Sanity checks on extracted tables."""
    for name, arr in tables.items():
        if name == "strrat":
            assert np.isfinite(arr), f"Band {band} {name}: non-finite"
            continue
        assert arr.size > 0, f"Band {band} {name}: empty"
        assert np.isfinite(arr).all(), f"Band {band} {name}: non-finite values"
        if name in ("absa", "absb", "ka_mn2", "kb_mn2", "ka_mco2", "kb_mco2",
                     "ka_mn2o", "kb_mn2o", "ka_mo2", "kb_mo2", "ka_mo3",
                     "ka_mco", "kb_mo3"):
            # Absorption coefficients should be non-negative
            assert (arr >= 0).all(), f"Band {band} {name}: negative values"
    print(f"  Band {band}: {len(tables)} arrays OK")


def save_band(band: int, tables: dict):
    """Save extracted tables as .npy files."""
    _TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in tables.items():
        fname = _TABLE_DIR / f"band{band}_{name}.npy"
        np.save(fname, arr)
    print(f"  Band {band}: saved {len(tables)} arrays to {_TABLE_DIR}")


def extract_trans_bpade(lib: ctypes.CDLL) -> dict:
    """Extract TRANS and BPADE from YOERRTAB."""
    trans = _read_array(lib, "__yoerrtab_MOD_trans", (5001,))
    bpade = _read_array(lib, "__yoerrtab_MOD_bpade", ())
    return {"trans": trans, "bpade": bpade}


def save_trans_bpade(tables: dict):
    """Save TRANS and BPADE as .npy files."""
    _TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in tables.items():
        fname = _TABLE_DIR / f"{name}.npy"
        np.save(fname, arr)
    print(f"  Saved trans/bpade to {_TABLE_DIR}")


def main():
    check_only = "--check" in sys.argv
    lib = _load_lib()

    # Extract TRANS and BPADE (shared across all bands).
    trans_tables = extract_trans_bpade(lib)
    if not check_only:
        save_trans_bpade(trans_tables)

    # Extract per-band tables.
    all_tables = {}
    for band in range(1, 17):
        tables = extract_band(lib, band, check=check_only)
        all_tables[band] = tables
        if not check_only:
            save_band(band, tables)

    print(f"\n{'Checked' if check_only else 'Extracted'} tables for bands 1-16 + "
          f"TRANS/BPADE from {_LIB_PATH.name}")
    return all_tables


if __name__ == "__main__":
    main()
