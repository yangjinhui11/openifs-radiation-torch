"""RRTM radiative transfer — rrtm_rtrn1a_140gp port.

Follows the Fortran reference (rrtm_rtrn1a_140gp.F90) exactly:
- Shared bglev (Z_BGLEV) variable tracked between downward/upward loops.
- Z_BBU1 pre-computed in downward loop and reused in upward loop.
- Full total-sky (cloudy) path with generalized maximum/random overlap:
    * Phase A precompute (upward overlap fractions, ISTCLD)
    * Phase B precompute (downward overlap fractions, ISTCLDD)
    * Dual-stream clear/cloudy RT with overlap switching in both directions.
- Flux indexing: bottom-up (0=sfc, nlev=TOA) during computation,
  TOA-first on output.
"""
import numpy as np
import torch
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TABLE_DIR = _HERE / "tables"


def _lt(n, s):
    return torch.from_numpy(np.load(str(_TABLE_DIR / f"{n}.npy"))).to(torch.float64)

# Lazy-loaded tables --------------------------------------------------
_tables = {}


def _ensure_tables(dev):
    if not _tables:
        _tables["totplnk"] = _lt("totplnk", (181, 16)).to(dev)
        _tables["delwave"] = _lt("delwave", (16,)).to(dev)
        _tables["ngb"] = _lt("ngb", (141,)).to(torch.int32).to(dev)
        _tables["bpade"] = _lt("bpade", ()).to(dev)


def _t2i(t):
    """Fortran INT(t)-159 logic (0-based)."""
    i = torch.where(t >= 339.0, 180,
                    torch.where(t >= 160.0, (t - 160.0).to(torch.int64),
                                torch.zeros_like(t, dtype=torch.int64)))
    f = torch.where(t >= 339.0, t - 339.0,
                    torch.where(t >= 160.0, t - t.to(torch.int64).to(t.dtype), t - 160.0))
    return i, f


# =====================================================================
# Cloud-overlap precomputation (Phases A & B + boundary block)
#
# These mirror rrtm_rtrn1a_140gp.F90 lines 262-470 exactly. They depend
# ONLY on cldfrac/icldlyr (per-column, per-layer), not on g-points, so we
# vectorize across nlon and loop sequentially across layers (the Z_RAT1/
# Z_RAT2 carry-over makes the layer axis sequential).
# =====================================================================
def _compute_overlap(icldlyr, cldfrac):
    """Precompute the maximum/random overlap switching fractions.

    Parameters
    ----------
    icldlyr : (nlon, nlev) int/bool, bottom-up, 1=cloudy layer.
    cldfrac : (nlon, nlev) float, bottom-up cloud fraction.

    Returns
    -------
    dict of (nlon, nlev+1) tensors keyed by the Fortran variable names:
        istcld, facclr1, facclr2, faccld1, faccld2, faccmb1, faccmb2  (upward)
        istcldd, facclr1d, facclr2d, faccld1d, faccld2d,
            faccmb1d, faccmb2d  (downward)
    The trailing nlev+1 axis is indexed 0..nlev and is used as JLEV+1
    (upward) or JLEV-1 (downward) by the RT loops.
    """
    nlon, nlev = icldlyr.shape
    dt = cldfrac.dtype
    dev = cldfrac.device
    z = lambda: torch.zeros(nlon, nlev + 1, dtype=dt, device=dev)

    out = {
        "istcld": z(), "facclr1": z(), "facclr2": z(),
        "faccld1": z(), "faccld2": z(), "faccmb1": z(), "faccmb2": z(),
        "istcldd": z(), "facclr1d": z(), "facclr2d": z(),
        "faccld1d": z(), "faccld2d": z(), "faccmb1d": z(), "faccmb2d": z(),
    }

    cf = cldfrac                              # (nlon, nlev); JLEV ↔ cf[:, JLEV-1]
    icl = (icldlyr == 1)

    # Boundary initializations (Fortran lines 262-263)
    out["istcld"][:, 0] = 1                    # ISTCLD(:,1)=1
    out["istcldd"][:, nlev] = 1                # ISTCLDD(:,KLEV)=1

    # =================================================================
    # Phase A — UPWARD overlap (Fortran lines 284-370): JLEV = 1..KLEV
    # =================================================================
    rat1 = torch.zeros(nlon, dtype=dt, device=dev)
    rat2 = torch.zeros(nlon, dtype=dt, device=dev)
    for jlev in range(1, nlev + 1):            # 1-based
        jl = jlev - 1                           # 0-based layer index
        jl_p1 = jlev                            # JLEV+1 ↔ idx jlev
        jl_m1 = jlev - 2                        # JLEV-1 ↔ idx (may be -1)
        cldy = icl[:, jl]
        cf_jl = cf[:, jl]                       # P_CLDFRAC(JLEV)
        cf_jlp1 = cf[:, min(jl_p1, nlev - 1)]   # P_CLDFRAC(JLEV+1)
        cf_jlm1 = cf[:, max(jl_m1, 0)]          # P_CLDFRAC(JLEV-1)
        istcld_jlev = out["istcld"][:, jl]

        is_top = (jlev == nlev)                     # python bool (scalar per iter)
        ge = (not is_top) & (cf_jlp1 >= cf_jl)      # line 299  (bool & tensor)
        lt = (not is_top) & (cf_jlp1 < cf_jl)       # line 332

        # ---- ge branch (lines 299-331) ----
        ge_first = ge & (istcld_jlev == 1)
        ge_else = ge & (istcld_jlev != 1)
        fmax_ge = torch.maximum(cf_jl, cf_jlm1)  # line 312
        ge_else_a = ge_else & (cf_jlp1 > fmax_ge)
        ge_else_b = ge_else & (cf_jlp1 < fmax_ge)
        ge_else_c = ge_else & ~(ge_else_a | ge_else_b)
        facclr1_ge_else = torch.where(
            ge_else_a, rat2,
            torch.where(ge_else_b,
                        torch.where((cf_jlm1 - cf_jl).abs() > 1e-30,
                                    (cf_jlp1 - cf_jl) / (cf_jlm1 - cf_jl),
                                    torch.zeros_like(cf_jlp1)),
                        torch.where(ge_else_c, rat2, torch.zeros_like(cf_jlp1))))
        facclr2_ge_else = torch.where(
            ge_else_a,
            torch.where(fmax_ge < 1.0,
                        (cf_jlp1 - fmax_ge) / (1.0 - fmax_ge).clamp(min=1e-30),
                        torch.zeros_like(cf_jlp1)),
            torch.zeros_like(cf_jlp1))
        # ge_first: facclr1=0, facclr2=(cf+ - cf)/(1-cf) if cf<1
        facclr2_ge_first = torch.where(
            ge_first & (cf_jl < 1.0),
            (cf_jlp1 - cf_jl) / (1.0 - cf_jl).clamp(min=1e-30),
            torch.zeros_like(cf_jlp1))
        facclr1_ge = torch.where(ge_first, torch.zeros_like(cf_jlp1), facclr1_ge_else)
        facclr2_ge = torch.where(ge_first, facclr2_ge_first, facclr2_ge_else)
        # rat update (lines 328-331) applies to ALL ge clouds (first or else),
        # using the final merged facclr1/facclr2 values.
        ge_clr_pos = (facclr1_ge > 0) | (facclr2_ge > 0)

        # ---- lt branch (lines 332-354) ----
        lt_first = lt & (istcld_jlev == 1)
        lt_else = lt & (istcld_jlev != 1)
        fmin_lt = torch.minimum(cf_jl, cf_jlm1)   # line 340
        lt_else_a = lt_else & (cf_jlp1 <= fmin_lt)
        lt_else_b = lt_else & ~lt_else_a
        faccld1_lt_else = torch.where(
            lt_else_a, rat1,
            torch.where(lt_else_b,
                        torch.where((cf_jl - fmin_lt).abs() > 1e-30,
                                    (cf_jl - cf_jlp1) / (cf_jl - fmin_lt),
                                    torch.zeros_like(cf_jlp1)),
                        torch.zeros_like(cf_jlp1)))
        faccld2_lt_else = torch.where(
            lt_else_a,
            torch.where(fmin_lt > 1e-30,
                        (fmin_lt - cf_jlp1) / fmin_lt.clamp(min=1e-30),
                        torch.zeros_like(cf_jlp1)),
            torch.zeros_like(cf_jlp1))
        faccld2_lt_first = torch.where(
            lt_first & (cf_jl > 1e-30),
            (cf_jl - cf_jlp1) / cf_jl.clamp(min=1e-30),
            torch.zeros_like(cf_jlp1))
        faccld1_lt = torch.where(lt_first, torch.zeros_like(cf_jlp1), faccld1_lt_else)
        faccld2_lt = torch.where(lt_first, faccld2_lt_first, faccld2_lt_else)
        # rat update (lines 350-353) applies to ALL lt clouds (first or else).
        lt_cld_pos = (faccld1_lt > 0) | (faccld2_lt > 0)

        # merge into per-layer fac values (top branch leaves them 0)
        facclr1 = torch.where(ge, facclr1_ge, torch.zeros_like(cf_jlp1))
        facclr2 = torch.where(ge, facclr2_ge, torch.zeros_like(cf_jlp1))
        faccld1 = torch.where(lt, faccld1_lt, torch.zeros_like(cf_jlp1))
        faccld2 = torch.where(lt, faccld2_lt, torch.zeros_like(cf_jlp1))

        # rat updates (lines 328-331, 350-353)
        rat1 = torch.where(ge & ge_clr_pos, torch.ones_like(rat1),
                  torch.where(lt & lt_cld_pos, torch.zeros_like(rat1), rat1))
        rat2 = torch.where(ge & ge_clr_pos, torch.zeros_like(rat2),
                  torch.where(lt & lt_cld_pos, torch.ones_like(rat2), rat2))

        istcld_jlp1 = torch.where(cldy, torch.zeros_like(istcld_jlev),
                                  torch.ones_like(istcld_jlev))

        # ---- FACCMB (lines 356-363) ----
        facclr2_jlev = out["facclr2"][:, jl]    # FACCLR2(JLON,JLEV), idx jl
        faccld2_jlev = out["faccld2"][:, jl]
        if jlev == 1:
            # line 357-358
            faccmb1 = torch.zeros_like(cf_jlp1)
            faccmb2 = faccld1 * facclr2_jlev
        else:
            # lines 360-362
            faccmb1 = facclr1 * faccld2_jlev * cf_jlm1
            faccmb2 = faccld1 * facclr2_jlev * (1.0 - cf_jlm1)

        # Fortran lines 289 / 367: ISTCLD(JLEV+1)=0 where cloudy, =1 where clear.
        # fac arrays stay 0 where clear (Fortran only writes them inside the
        # K_ICLDLYR==1 branch).
        m = cldy
        out["istcld"][:, jl_p1] = torch.where(m, istcld_jlp1,
                                              torch.ones_like(istcld_jlp1))
        out["facclr1"][:, jl_p1] = torch.where(m, facclr1, out["facclr1"][:, jl_p1])
        out["facclr2"][:, jl_p1] = torch.where(m, facclr2, out["facclr2"][:, jl_p1])
        out["faccld1"][:, jl_p1] = torch.where(m, faccld1, out["faccld1"][:, jl_p1])
        out["faccld2"][:, jl_p1] = torch.where(m, faccld2, out["faccld2"][:, jl_p1])
        out["faccmb1"][:, jl_p1] = torch.where(m, faccmb1, out["faccmb1"][:, jl_p1])
        out["faccmb2"][:, jl_p1] = torch.where(m, faccmb2, out["faccmb2"][:, jl_p1])

    # Phase A resets rat1/rat2 (lines 373-374)
    rat1 = torch.zeros(nlon, dtype=dt, device=dev)
    rat2 = torch.zeros(nlon, dtype=dt, device=dev)

    # =================================================================
    # Phase B — DOWNWARD overlap (Fortran lines 380-450): JLEV = KLEV..2
    # =================================================================
    for jlev in range(nlev, 1, -1):            # 1-based, KLEV down to 2
        # ISTCLDD is declared 0-based (0:NFLEVG), so Fortran index JLEV maps
        # directly to the torch array index jlev.
        jl = jlev - 1                          # 0-based layer index for cldfrac
        jl_m1 = jlev - 1                       # ISTCLDD(JLEV-1) write index
        # P_CLDFRAC is 1-based (1:NFLEVG): P_CLDFRAC(JLEV+/-1) -> 0-based JLEV+/-1-1
        cldy = icl[:, jl]
        cf_jl = cf[:, jl]                      # P_CLDFRAC(JLEV)   = cf[:, jlev-1]
        cf_jlm1 = cf[:, jlev - 2]              # P_CLDFRAC(JLEV-1) = cf[:, jlev-2]
        cf_jlp1 = cf[:, min(jlev, nlev - 1)]   # P_CLDFRAC(JLEV+1) = cf[:, jlev]
        istcldd_jlev = out["istcldd"][:, jlev]  # ISTCLDD(JLEV), 0-based idx jlev

        ge = (cf_jlm1 >= cf_jl)                 # line 386
        lt = (cf_jlm1 < cf_jl)

        # ---- ge branch (lines 386-417) ----
        ge_first = ge & (istcldd_jlev == 1)
        ge_else = ge & (istcldd_jlev != 1)
        fmax_ge = torch.maximum(cf_jl, cf_jlp1)
        ge_else_a = ge_else & (cf_jlm1 > fmax_ge)
        ge_else_b = ge_else & (cf_jlm1 < fmax_ge)
        ge_else_c = ge_else & ~(ge_else_a | ge_else_b)
        facclr1d_ge_else = torch.where(
            ge_else_a, rat2,
            torch.where(ge_else_b,
                        torch.where((cf_jlp1 - cf_jl).abs() > 1e-30,
                                    (cf_jlm1 - cf_jl) / (cf_jlp1 - cf_jl),
                                    torch.zeros_like(cf_jlm1)),
                        torch.where(ge_else_c, rat2, torch.zeros_like(cf_jlm1))))
        facclr2d_ge_else = torch.where(
            ge_else_a,
            torch.where(fmax_ge < 1.0,
                        (cf_jlm1 - fmax_ge) / (1.0 - fmax_ge).clamp(min=1e-30),
                        torch.zeros_like(cf_jlm1)),
            torch.zeros_like(cf_jlm1))
        facclr2d_ge_first = torch.where(
            ge_first & (cf_jl < 1.0),
            (cf_jlm1 - cf_jl) / (1.0 - cf_jl).clamp(min=1e-30),
            torch.zeros_like(cf_jlm1))
        facclr1d_ge = torch.where(ge_first, torch.zeros_like(cf_jlm1), facclr1d_ge_else)
        facclr2d_ge = torch.where(ge_first, facclr2d_ge_first, facclr2d_ge_else)
        # rat update (lines 414-417) applies to ALL ge clouds (first or else).
        ge_clr_pos = (facclr1d_ge > 0) | (facclr2d_ge > 0)

        # ---- lt branch (lines 418-439) ----
        lt_first = lt & (istcldd_jlev == 1)
        lt_else = lt & (istcldd_jlev != 1)
        fmin_lt = torch.minimum(cf_jl, cf_jlp1)
        lt_else_a = lt_else & (cf_jlm1 <= fmin_lt)
        lt_else_b = lt_else & ~lt_else_a
        faccld1d_lt_else = torch.where(
            lt_else_a, rat1,
            torch.where(lt_else_b,
                        torch.where((cf_jl - fmin_lt).abs() > 1e-30,
                                    (cf_jl - cf_jlm1) / (cf_jl - fmin_lt),
                                    torch.zeros_like(cf_jlm1)),
                        torch.zeros_like(cf_jlm1)))
        faccld2d_lt_else = torch.where(
            lt_else_a,
            torch.where(fmin_lt > 1e-30,
                        (fmin_lt - cf_jlm1) / fmin_lt.clamp(min=1e-30),
                        torch.zeros_like(cf_jlm1)),
            torch.zeros_like(cf_jlm1))
        faccld2d_lt_first = torch.where(
            lt_first & (cf_jl > 1e-30),
            (cf_jl - cf_jlm1) / cf_jl.clamp(min=1e-30),
            torch.zeros_like(cf_jlm1))
        faccld1d_lt = torch.where(lt_first, torch.zeros_like(cf_jlm1), faccld1d_lt_else)
        faccld2d_lt = torch.where(lt_first, faccld2d_lt_first, faccld2d_lt_else)
        # rat update (lines 436-439) applies to ALL lt clouds (first or else).
        lt_cld_pos = (faccld1d_lt > 0) | (faccld2d_lt > 0)

        facclr1d = torch.where(ge, facclr1d_ge, torch.zeros_like(cf_jlm1))
        facclr2d = torch.where(ge, facclr2d_ge, torch.zeros_like(cf_jlm1))
        faccld1d = torch.where(lt, faccld1d_lt, torch.zeros_like(cf_jlm1))
        faccld2d = torch.where(lt, faccld2d_lt, torch.zeros_like(cf_jlm1))

        rat1 = torch.where(ge & ge_clr_pos, torch.ones_like(rat1),
                  torch.where(lt & lt_cld_pos, torch.zeros_like(rat1), rat1))
        rat2 = torch.where(ge & ge_clr_pos, torch.zeros_like(rat2),
                  torch.where(lt & lt_cld_pos, torch.ones_like(rat2), rat2))

        istcldd_jlm1 = torch.where(cldy, torch.zeros_like(istcldd_jlev),
                                    torch.ones_like(istcldd_jlev))

        # FACCMBD (lines 441-444), only if JLEV /= KLEV.
        # Reads Z_FACCLD2D(JLON,JLEV) / Z_FACCLR2D(JLON,JLEV) — these are the
        # 0-based ISTCLDD-family arrays so index = JLEV = jlev.
        facclr2d_jlev = out["facclr2d"][:, jlev]
        faccld2d_jlev = out["faccld2d"][:, jlev]
        if jlev == nlev:
            faccmb1d = torch.zeros_like(cf_jlm1)
            faccmb2d = torch.zeros_like(cf_jlm1)
        else:
            faccmb1d = facclr1d * faccld2d_jlev * cf_jlp1
            faccmb2d = faccld1d * facclr2d_jlev * (1.0 - cf_jlp1)

        m = cldy
        # ISTCLDD(JLEV-1)=0 where cloudy, =1 where clear (Fortran lines 384/447)
        out["istcldd"][:, jl_m1] = torch.where(m, istcldd_jlm1,
                                               torch.ones_like(istcldd_jlm1))
        out["facclr1d"][:, jl_m1] = torch.where(m, facclr1d, out["facclr1d"][:, jl_m1])
        out["facclr2d"][:, jl_m1] = torch.where(m, facclr2d, out["facclr2d"][:, jl_m1])
        out["faccld1d"][:, jl_m1] = torch.where(m, faccld1d, out["faccld1d"][:, jl_m1])
        out["faccld2d"][:, jl_m1] = torch.where(m, faccld2d, out["faccld2d"][:, jl_m1])
        out["faccmb1d"][:, jl_m1] = torch.where(m, faccmb1d, out["faccmb1d"][:, jl_m1])
        out["faccmb2d"][:, jl_m1] = torch.where(m, faccmb2d, out["faccmb2d"][:, jl_m1])

    # Boundary block (Fortran lines 452-470): ILEV=1 downward surface.
    # When layer 1 is cloudy: ISTCLDD(:,0)=0 and all six D arrays at idx 0
    # remain 0 (their FACCMB recompute also yields 0 because the factors
    # just zeroed are 0). When clear: ISTCLDD(:,0) stays 1.
    cldy0 = icl[:, 0]
    out["istcldd"][:, 0] = torch.where(cldy0, torch.zeros(nlon, dtype=dt, device=dev),
                                       out["istcldd"][:, 0])
    return out


def rrtm_rtrn1a_140gp(
    nlev, istart, iend,
    icldlyr, cldfrac, taucld,
    atr1, od, tf1,
    tavel, tz, tbound,
    pfrac, semiss,
):
    """Clear-sky + total-sky LW radiative transfer.

    All array inputs expected in (nlon, ngpt, nlev) or (nlon, nlev) order,
    bottom-up (index 0 = surface). ``taucld`` is (nlon, nlev, 16) band-space.

    Returns dict with totufluc, totdfluc, totuflux, totdflux, semislw.
    All output arrays are TOA-first (index 0 = TOA).
    """
    dt = tavel.dtype
    dev = tavel.device
    nlon = tavel.shape[0]
    _ensure_tables(dev)

    totplnk = _tables["totplnk"]
    delwave = _tables["delwave"]
    ngb = _tables["ngb"]
    bpade = float(_tables["bpade"].item())

    band_of_g = ngb[1:].long()                      # (140,), 1-based band per gpt
    wtn = 0.5

    # ---- Planck per band per level -------------------------------------------
    il, tlf = _t2i(tz)                               # (nlon, nlev+1)
    ild, tlayf = _t2i(tavel)                         # (nlon, nlev)

    play = torch.zeros(nlon, 16, nlev, dtype=dt, device=dev)
    plvl = torch.zeros(nlon, 16, nlev + 1, dtype=dt, device=dev)
    for jb in range(16):
        dlev = totplnk[(il.clamp(0, 179) + 1).clamp(1, 180), jb] - totplnk[il.clamp(0, 179), jb]
        plvl[:, jb, :] = delwave[jb] * (totplnk[il.clamp(0, 179), jb] + tlf * dlev)
        dlay = totplnk[(ild.clamp(0, 179) + 1).clamp(1, 180), jb] - totplnk[ild.clamp(0, 179), jb]
        play[:, jb, :] = delwave[jb] * (totplnk[ild.clamp(0, 179), jb] + tlayf * dlay)

    # Surface emission ---------------------------------------------------------
    ib, tbf = _t2i(tbound)
    ib = ib.clamp(0, 179)
    plankbnd = delwave * (totplnk[ib, :] + tbf.unsqueeze(-1) * (
        totplnk[(ib + 1).clamp(1, 180), :] - totplnk[ib, :]))
    plankemit = semiss * plankbnd                        # (nlon, 16)
    semislw = (plankemit.sum(dim=-1) / plankbnd.sum(dim=-1).clamp(min=1e-30))

    # ---- Expand Planck to g-points -------------------------------------------
    def _bg(arr, jk):
        """arr: (nlon, 16, nlev_or_nlev+1) -> (nlon, 140) for level/layer jk."""
        return arr[:, band_of_g - 1, jk]

    # Cloud optical depth in g-space: taucld (nlon,nlev,16) -> (nlon,140,nlev)
    taucld_g = taucld[:, :, band_of_g - 1].permute(0, 2, 1).contiguous()
    trncld = torch.exp(-torch.clamp(taucld_g, max=200.0))    # Z_TRNCLD
    abs_cld = 1.0 - trncld                                   # 1 - Z_TRNCLD

    # ---- Cloud-overlap precompute -------------------------------------------
    ov = _compute_overlap(icldlyr.to(dev), cldfrac)
    istcld = ov["istcld"]            # (nlon, nlev+1)
    istcldd = ov["istcldd"]
    icl = (icldlyr == 1)             # (nlon, nlev) bool

    # ---- Downward RT ---------------------------------------------------------
    radclrd = torch.zeros(nlon, 140, dtype=dt, device=dev)   # clear downward
    radld = torch.zeros(nlon, 140, dtype=dt, device=dev)     # total downward
    cldradd = torch.zeros(nlon, 140, dtype=dt, device=dev)
    clrradd = torch.zeros(nlon, 140, dtype=dt, device=dev)
    oldcld = torch.zeros(nlon, 140, dtype=dt, device=dev)
    oldclr = torch.zeros(nlon, 140, dtype=dt, device=dev)
    zrad = torch.zeros(nlon, 140, dtype=dt, device=dev)

    tdfc_list = []
    tdf_list = []
    bbu1_stored = torch.zeros(nlon, 140, nlev, dtype=dt, device=dev)
    bbutot1_stored = torch.zeros(nlon, 140, nlev, dtype=dt, device=dev)
    atot1_stored = torch.zeros(nlon, 140, nlev, dtype=dt, device=dev)

    # flux lists (TOA appended after loop)

    iclddn = torch.zeros(nlon, dtype=torch.bool, device=dev)

    bglev = pfrac[:, :, nlev - 1] * _bg(plvl, nlev)     # init at TOA boundary

    for jk in range(nlev - 1, -1, -1):
        bglay = pfrac[:, :, jk] * _bg(play, jk)
        delbgup = bglev - bglay
        bbu1 = bglay + tf1[:, :, jk] * delbgup
        bbu1_stored[:, :, jk] = bbu1
        bglev = pfrac[:, :, jk] * _bg(plvl, jk)
        delbgdn = bglev - bglay
        bbd = bglay + tf1[:, :, jk] * delbgdn

        cldy_jk = icl[:, jk]
        iclddn = iclddn | cldy_jk

        # overlap at JLEV-1 for downward; JLEV=jk+1 -> idx jk
        # Fortran reads FAC*D(JLEV-1) and ISTCLDD(JLEV) in the downward loop.
        # jk ↔ JLEV-1, so JLEV = jk+1.  Therefore the *D arrays use idx jk
        # (JLEV-1) but ISTCLDD uses idx jk+1 (JLEV).
        fclr1d = ov["facclr1d"][:, jk]
        fcld1d = ov["faccld1d"][:, jk]
        fcmb1d = ov["faccmb1d"][:, jk]
        fcmb2d = ov["faccmb2d"][:, jk]
        fclr2d = ov["facclr2d"][:, jk]
        fcld2d = ov["faccld2d"][:, jk]
        istd = istcldd[:, jk + 1]

        # total-sky quantities (lines 648-656)
        odsm = od[:, :, jk] + taucld_g[:, :, jk]
        factot1 = odsm / (bpade + odsm)
        atot1 = atr1[:, :, jk] + abs_cld[:, :, jk] - atr1[:, :, jk] * abs_cld[:, :, jk]
        bbutot1 = bglay + factot1 * delbgup
        bbdtot = bglay + factot1 * delbgdn
        ttot = 1.0 - atot1
        gassrc = bbd * atr1[:, :, jk]
        cldsrc = bbdtot * atot1
        bbutot1_stored[:, :, jk] = bbutot1
        atot1_stored[:, :, jk] = atot1

        cf_jk = cldfrac[:, jk].unsqueeze(-1)
        atr1_jk = atr1[:, :, jk]

        # ---- initialize cloud-stream split on first cloudy layer of group
        init_split = cldy_jk.unsqueeze(-1) & (istd == 1).unsqueeze(-1)
        cldradd = torch.where(init_split, cf_jk * radld, cldradd)
        clrradd = torch.where(init_split, radld - cf_jk * radld, clrradd)
        oldcld = torch.where(init_split, cldradd, oldcld)
        oldclr = torch.where(init_split, clrradd, oldclr)
        zrad = torch.where(init_split, torch.zeros_like(zrad), zrad)

        # ---- cloudy-layer dual-stream RT (lines 659-688) ----
        cldradd_cld = cldradd * ttot + cf_jk * cldsrc
        clrradd_cld = clrradd * (1.0 - atr1_jk) + (1.0 - cf_jk) * gassrc
        radld_cld = cldradd_cld + clrradd_cld
        radclrd_cld = radclrd + (bbd - radclrd) * atr1_jk

        radmod = (zrad * (fclr1d.unsqueeze(-1) * (1.0 - atr1_jk)
                          + fcld1d.unsqueeze(-1) * ttot)
                  - fcmb1d.unsqueeze(-1) * gassrc
                  + fcmb2d.unsqueeze(-1) * cldsrc)
        oldcld_m = cldradd_cld - radmod
        oldclr_m = clrradd_cld + radmod
        zrad_m = (-radmod + fclr2d.unsqueeze(-1) * oldclr_m
                  - fcld2d.unsqueeze(-1) * oldcld_m)
        cldradd_cld = cldradd_cld + zrad_m
        clrradd_cld = clrradd_cld - zrad_m

        # ---- clear-layer RT (lines 698-716) ----
        radld_clr = radld + (bbd - radld) * atr1_jk
        radclrd_clr_indep = radclrd + (bbd - radclrd) * atr1_jk
        radclrd_clr = torch.where(iclddn.unsqueeze(-1), radclrd_clr_indep, radld_clr)

        # apply cloudy vs clear per column
        cmask = cldy_jk.unsqueeze(-1)
        radld = torch.where(cmask, radld_cld, radld_clr)
        radclrd = torch.where(cmask, radclrd_cld, radclrd_clr)
        cldradd = torch.where(cmask, cldradd_cld, cldradd)
        clrradd = torch.where(cmask, clrradd_cld, clrradd)
        oldcld = torch.where(cmask, oldcld_m, oldcld)
        oldclr = torch.where(cmask, oldclr_m, oldclr)
        zrad = torch.where(cmask, zrad_m, zrad)

        # flux accumulation (lines 723-736)
        drad1 = radld.sum(dim=1)
        dradcl1 = torch.where(cldy_jk | iclddn, radclrd.sum(dim=1), drad1)
        tdf_list.append(drad1 * wtn)
        tdfc_list.append(dradcl1 * wtn)

    # ---- Surface -----------------------------------------------------
    semiss_g = semiss[:, band_of_g - 1]
    plankemit_g = plankemit[:, band_of_g - 1]
    raduemit = pfrac[:, :, 0] * plankemit_g

    radclru = raduemit + (1.0 - semiss_g) * radclrd
    radlu = raduemit + (1.0 - semiss_g) * radld

    tdfc_list.append(torch.zeros(nlon, dtype=dt, device=dev))
    tdf_list.append(torch.zeros(nlon, dtype=dt, device=dev))

    tufc_list = [radclru.sum(dim=1) * wtn]
    tuf_list = [radlu.sum(dim=1) * wtn]

    # ---- Upward RT --------------------------------------------------
    cldradu = torch.zeros(nlon, 140, dtype=dt, device=dev)
    clrradu = torch.zeros(nlon, 140, dtype=dt, device=dev)

    for jk in range(nlev):
        bbu1_jk = bbu1_stored[:, :, jk]
        atot1_jk = atot1_stored[:, :, jk]
        bbutot1_jk = bbutot1_stored[:, :, jk]
        cldy_jk = icl[:, jk]
        cf_jk = cldfrac[:, jk].unsqueeze(-1)
        atr1_jk = atr1[:, :, jk]

        # overlap at JLEV+1 for upward; JLEV=jk+1 -> idx jk+1
        fclr1 = ov["facclr1"][:, jk + 1]
        fcld1 = ov["faccld1"][:, jk + 1]
        fcmb1 = ov["faccmb1"][:, jk + 1]
        fcmb2 = ov["faccmb2"][:, jk + 1]
        fclr2 = ov["facclr2"][:, jk + 1]
        fcld2 = ov["faccld2"][:, jk + 1]
        ist = istcld[:, jk]

        # initialize cloud-stream split (lines 822-828)
        init_split = cldy_jk.unsqueeze(-1) & (ist == 1).unsqueeze(-1)
        cldradu = torch.where(init_split, cf_jk * radlu, cldradu)
        clrradu = torch.where(init_split, radlu - cf_jk * radlu, clrradu)
        oldcld = torch.where(init_split, cldradu, oldcld)
        oldclr = torch.where(init_split, clrradu, oldclr)
        zrad = torch.where(init_split, torch.zeros_like(zrad), zrad)

        # cloudy-layer dual-stream upward RT (lines 832-872)
        gassrc_u = bbu1_jk * atr1_jk
        ttot = 1.0 - atot1_jk
        trns = 1.0 - atr1_jk
        cldsrc_u = bbutot1_jk * atot1_jk
        cldradu_cld = cldradu * ttot + cf_jk * cldsrc_u
        clrradu_cld = clrradu * trns + (1.0 - cf_jk) * gassrc_u
        radlu_cld = cldradu_cld + clrradu_cld
        radclru_cld = radclru + (bbu1_jk - radclru) * atr1_jk

        radmod = (zrad * (fclr1.unsqueeze(-1) * trns + fcld1.unsqueeze(-1) * ttot)
                  - fcmb1.unsqueeze(-1) * gassrc_u
                  + fcmb2.unsqueeze(-1) * cldsrc_u)
        oldcld_m = cldradu_cld - radmod
        oldclr_m = clrradu_cld + radmod
        zrad_m = (-radmod + fclr2.unsqueeze(-1) * oldclr_m
                  - fcld2.unsqueeze(-1) * oldcld_m)
        cldradu_cld = cldradu_cld + zrad_m
        clrradu_cld = clrradu_cld - zrad_m

        # clear-layer upward RT (lines 878-888)
        radlu_clr = radlu + (bbu1_jk - radlu) * atr1_jk
        radclru_clr = radclru + (bbu1_jk - radclru) * atr1_jk

        cmask = cldy_jk.unsqueeze(-1)
        radlu = torch.where(cmask, radlu_cld, radlu_clr)
        radclru = torch.where(cmask, radclru_cld, radclru_clr)
        cldradu = torch.where(cmask, cldradu_cld, cldradu)
        clrradu = torch.where(cmask, clrradu_cld, clrradu)
        oldcld = torch.where(cmask, oldcld_m, oldcld)
        oldclr = torch.where(cmask, oldclr_m, oldclr)
        zrad = torch.where(cmask, zrad_m, zrad)

        tuf_list.append(radlu.sum(dim=1) * wtn)
        tufc_list.append(radclru.sum(dim=1) * wtn)

    # Upward lists are built surface-first (bottom-up); flip to TOA-first.
    stack_flip = lambda lst: torch.flip(torch.stack(lst, dim=1), dims=[1])
    # Downward lists are built TOA-first (down loop runs jk=nlev-1 -> 0);
    # the appended trailing zero is the surface half-level boundary, so the
    # list is already in TOA-first order and must NOT be flipped.
    stack_plain = lambda lst: torch.stack(lst, dim=1)
    totdflux = stack_plain(tdf_list)
    totdfluc = stack_plain(tdfc_list)
    # The Fortran RTRN1A leaves the surface half-level (index nlev) of the
    # downward flux at zero (the down loop fills only 0..nlev-1).  For the
    # heating-rate divergence to be physically correct at the surface layer,
    # mirror the near-surface value (index nlev-1) into the surface half-level,
    # consistent with the upward flux which IS defined at the surface.
    totdflux[:, -1] = totdflux[:, -2]
    totdfluc[:, -1] = totdfluc[:, -2]
    # ---- Output: TOA-first (index 0 = TOA, index nlev = surface) ----
    return {
        "totufluc": stack_flip(tufc_list),
        "totdfluc": totdfluc,
        "totuflux": stack_flip(tuf_list),
        "totdflux": totdflux,
        "semislw": semislw,
    }
