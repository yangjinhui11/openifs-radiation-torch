#!/usr/bin/env python3
"""Generate multi-column vertical_profiles_multi.json for Figure 2.

Uses REAL ERA5 atmospheric profiles (T, Q, p) from multi_region_fluxes.json,
which contains 4 distinct regions (70N, 45N, EQ, 70S) — the SAME regions
used in Figure 1. This makes Figure 1 and Figure 2 use identical atmospheric
states, forming a logical chain: physical output (Fig 1) → bit-exact
verification (Fig 2).

For polar-night regions (mu0=0), a small mu0 floor (0.15) is assigned so the
SW kernels are well-defined; this does not affect the bit-exact comparison
since both torch and Fortran receive the identical input.

Output: /tmp/vertical_profiles_multi.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch

# ── Setup paths ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_config import setup_paths, find_era5, solver_root
setup_paths()
_HERE = Path(__file__).resolve().parent
_offline = solver_root() / "offline_ref"
sys.path.insert(0, str(_offline))

try:
    from offline_ref.sw_ref import SwRef
except ImportError as e:
    raise SystemExit("offline_ref/sw_ref.py not found at %s. The bit-exact "
                     "comparison requires the Fortran reference library; see "
                     "experiments/e01_bitexact_verification/README.md." % _offline)
from openifs_radiation.classic_sw.driver import _swu
from openifs_radiation.classic_sw.swclr import swclr as torch_swclr, _prayl

# ── Load real ERA5 multi-region profiles ────────────────────────────────
print("Loading ERA5 multi-region profiles...")
_mrf_path = find_era5("multi_region_fluxes.json")
with open(_mrf_path) as f:
    mrf = json.load(f)

regions = mrf["regions"]
print(f"Regions: {[r['name'] for r in regions]}")

# ── Initialise Fortran reference ─────────────────────────────────────────
print("Initialising Fortran reference library...")
ref = SwRef()
ref.init()

PCARDI = 348.0e-6   # VMR
PSCT = 1361.0

tdir = RAD_ROOT / "openifs_radiation" / "rrtm_lw" / "tables"
rray = torch.from_numpy(np.load(str(tdir / "sw_rray.npy"))).to(torch.float64)
KNU = 1

# ── Per-region physical parameters (consistent with Fig 1) ──────────────
# mu0: use real ERA5 value if > 0, otherwise floor at 0.15 (polar night fix)
# albedo: realistic per-latitude values
# aerosol: exponential decay profile, scaled by latitude
REGION_PARAMS = {
    "70N": {"mu0_floor": 0.15, "albedo": 0.60, "aer_scale": 0.5},
    "45N": {"mu0_floor": 0.337, "albedo": 0.15, "aer_scale": 1.0},
    "EQ":  {"mu0_floor": 0.796, "albedo": 0.08, "aer_scale": 2.0},
    "70S": {"mu0_floor": 0.15, "albedo": 0.60, "aer_scale": 0.5},
}

# ── Helper: process one region ──────────────────────────────────────────
def process_region(region):
    name = region["name"]
    params = REGION_PARAMS[name]
    T   = np.array(region["T"])
    Q   = np.array(region["Q"])
    p_half_hpa = np.array(region["p_half_hpa"])
    p_full_hpa = np.array(region["p_full_hpa"])
    # Convert hPa → Pa for the SW kernels
    p_half = p_half_hpa * 100.0
    p_full = p_full_hpa * 100.0
    mu0 = max(region["mu0"], params["mu0_floor"])
    albedo = params["albedo"]
    aer_scale = params["aer_scale"]
    nlev = len(T)
    T_sfc = float(T[-1])

    print(f"\n=== Processing {name} (T_sfc={T_sfc:.1f}K, mu0={mu0:.3f}, alb={albedo}) ===")

    dt = torch.float64
    T_t   = torch.from_numpy(T).to(dt).reshape(1, -1)
    Q_t   = torch.from_numpy(Q).to(dt).reshape(1, -1)
    O3_t  = torch.zeros((1, nlev), dtype=dt)
    CO2_t = torch.full((1, nlev), PCARDI, dtype=dt)
    p_h_t = torch.from_numpy(p_half).to(dt).reshape(1, -1)
    p_f_t = torch.from_numpy(p_full).to(dt).reshape(1, -1)
    prmu0_t = torch.tensor([mu0], dtype=dt)

    # ── Torch SWU ──
    swu_t = _swu(p_h_t, p_f_t, T_t, Q_t, CO2_t, O3_t, prmu0_t)

    # ── Fortran SWU ──
    ppmb_bu   = np.asfortranarray((p_half / 100.0)[::-1].reshape(1, -1))
    ptave_bu  = np.asfortranarray(T[::-1].reshape(1, -1))
    pwv_td    = np.asfortranarray(Q.reshape(1, -1))
    ppsol     = np.array([p_half[-1]])
    swu_f = ref.swu(PSCT, PCARDI, ppmb_bu, ppsol, np.array([mu0]), ptave_bu, pwv_td)

    pud_t = swu_t["pud"][0].numpy()
    pud_f = np.array(swu_f["pud"])[0][:, ::-1]

    pdsig_t = swu_t["pdsig"]
    psec_t  = swu_t["psec"]
    prmu_t  = swu_t["prmu"]

    # ── SWCLR ──
    rray_knu = rray[KNU - 1]
    prayl_t = _prayl(rray_knu, prmu_t)

    # Aerosol: exponential decay profile × per-region scale
    aer_profile = np.exp(-np.linspace(0, 5, nlev))
    paer = torch.from_numpy(aer_profile * aer_scale * 0.05).to(dt).reshape(1, 1, nlev).expand(1, 6, nlev).contiguous()
    palbp_full = torch.full((1, 6), albedo, dtype=dt)

    out_t = torch_swclr(
        knu=KNU, kaer=1,
        paer_td=paer, palbp=palbp_full,
        pdsig_td=pdsig_t, prayl=prayl_t,
        psec=psec_t, rray=rray, prmu=prmu_t,
    )

    paer_f  = paer.numpy()
    palbp_f = palbp_full.numpy()
    pdsig_f = np.asfortranarray(pdsig_t.numpy()[:, ::-1])
    out_f = ref.swclr(KNU, 1, paer_f, palbp_f, pdsig_f, np.array([prayl_t.item()]), psec_t.numpy())

    pcgaz_t = out_t["pcgaz"].numpy().flatten()
    pcgaz_f = np.array(out_f["pcgaz"]).flatten()
    ppizaz_t = out_t["ppizaz"].numpy().flatten()
    ppizaz_f = np.array(out_f["ppizaz"]).flatten()
    ptauz_t = out_t["ptauz"].numpy().flatten()
    ptauz_f = np.array(out_f["ptauz"]).flatten()
    pray1_t = out_t["pray1"].numpy().flatten()
    pray1_f = np.array(out_f["pray1"]).flatten()
    pray2_t = out_t["pray2"].numpy().flatten()
    pray2_f = np.array(out_f["pray2"]).flatten()
    ptra1_t = out_t["ptra1"].numpy().flatten()
    ptra1_f = np.array(out_f["ptra1"]).flatten()
    ptra2_t = out_t["ptra2"].numpy().flatten()
    ptra2_f = np.array(out_f["ptra2"]).flatten()
    prmu0_out_t = out_t["prmu0"].numpy().flatten()
    prmu0_out_f = np.array(out_f["prmu0"]).flatten()
    ptrclr_t = out_t["ptrclr"].numpy().flatten()
    ptrclr_f = np.array(out_f["ptrclr"]).flatten()

    diffs = {
        "pud_h2o":  float(np.max(np.abs(pud_t[0] - pud_f[0]))),
        "pud_co2":  float(np.max(np.abs(pud_t[1] - pud_f[1]))),
        "pcgaz":    float(np.nanmax(np.abs(pcgaz_t - pcgaz_f))),
        "ppizaz":   float(np.nanmax(np.abs(ppizaz_t - ppizaz_f))),
        "ptauz":    float(np.nanmax(np.abs(ptauz_t - ptauz_f))),
        "pray1":    float(np.nanmax(np.abs(pray1_t - pray1_f))),
        "pray2":    float(np.nanmax(np.abs(pray2_t - pray2_f))),
        "ptra1":    float(np.nanmax(np.abs(ptra1_t - ptra1_f))),
        "ptra2":    float(np.nanmax(np.abs(ptra2_t - ptra2_f))),
        "prmu0":    float(np.nanmax(np.abs(prmu0_out_t - prmu0_out_f))),
        "ptrclr":   float(np.nanmax(np.abs(ptrclr_t - ptrclr_f))),
    }
    print(f"  Max diffs: {diffs}")

    return {
        "name": name, "T_sfc": T_sfc, "mu0": mu0,
        "T": T.tolist(), "Q": Q.tolist(),
        "p_half_hpa": p_half_hpa.tolist(), "p_full_hpa": p_full_hpa.tolist(),
        "pud_h2o_torch": pud_t[0].tolist(), "pud_h2o_fortran": pud_f[0].tolist(),
        "pud_co2_torch": pud_t[1].tolist(), "pud_co2_fortran": pud_f[1].tolist(),
        "pcgaz_torch": pcgaz_t.tolist(), "pcgaz_fortran": pcgaz_f.tolist(),
        "ppizaz_torch": ppizaz_t.tolist(), "ppizaz_fortran": ppizaz_f.tolist(),
        "ptauz_torch": ptauz_t.tolist(), "ptauz_fortran": ptauz_f.tolist(),
        "pray1_torch": pray1_t.tolist(), "pray1_fortran": pray1_f.tolist(),
        "pray2_torch": pray2_t.tolist(), "pray2_fortran": pray2_f.tolist(),
        "ptra1_torch": ptra1_t.tolist(), "ptra1_fortran": ptra1_f.tolist(),
        "ptra2_torch": ptra2_t.tolist(), "ptra2_fortran": ptra2_f.tolist(),
        "prmu0_torch": prmu0_out_t.tolist(), "prmu0_fortran": prmu0_out_f.tolist(),
        "ptrclr_torch": ptrclr_t.tolist(), "ptrclr_fortran": ptrclr_f.tolist(),
        "diffs": diffs,
    }

# ── Run all regions ──────────────────────────────────────────────────────
columns = []
for region in regions:
    columns.append(process_region(region))

# ── Save ────────────────────────────────────────────────────────────────
out_path = "/tmp/vertical_profiles_multi.json"
result = {
    "p_half_hpa": regions[0]["p_half_hpa"],
    "p_full_hpa": regions[0]["p_full_hpa"],
    "columns": columns,
}
with open(out_path, "w") as f:
    json.dump(result, f)
print(f"\nSaved {len(columns)} columns to {out_path}")
for c in columns:
    print(f"  {c['name']}: T_sfc={c['T_sfc']:.1f}K, mu0={c['mu0']:.3f}, max diff={max(c['diffs'].values()):.2e}")
