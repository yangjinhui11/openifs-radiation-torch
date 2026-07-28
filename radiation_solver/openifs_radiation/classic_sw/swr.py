"""Port of swr.F90 — cloudy-sky reflectivity/transmissivity adding.

Fortran reference:
  openifs-48r1/ifs-source/arpifs/phys_radi/swr.F90

Computes the cloudy-column adding matrix used by sw1s (bands 1-3) and swni
(bands 4-6). swr mirrors swclr but couples the clear-sky layer optics (from
swclr: PCGAZ/PPIZAZ/PTAUAZ) with the cloud optics (PCG/POMEGA/PTAU/PCLD via a
delta-Eddington blend, then calls swde for the layer R/T.

Production config only: NOVLP=1 (maximum-random), ECMWF branch (NOVLP<5),
NSW=6. Uses the verbatim swde port for the delta-Eddington layer R/T.

Bottom-up internal convention (matches Fortran line-for-line); PREFZ index
1 = surface (Fortran 1-based layout, returned 0-based).

Autograd note: this version avoids every in-place index assignment
(``arr[..., idx] = expr``), cumulative ``+=`` and ``.fill_``/``.zero_`` on
gradient-carrying tensors, so the whole routine is differentiable. Loops that
previously filled a pre-allocated tensor instead collect per-level results
into Python lists and ``torch.stack`` them at the end; carried state
(``zclear``/``zcloud``/``prefz_prev``/``prj_prev``) is rebound with functional
``x = x + delta``-style expressions. The arithmetic is unchanged, so outputs
are bit-exact with the in-place version.
"""
from __future__ import annotations
import torch

from .swde import swde as _swde

# REPSEC cloud-cover security (YOECLD, set in rad_ref wrapper to 1e-12).
_REPSEC = 1.0e-12


def swr(knu: int, palbp: torch.Tensor, pcg: torch.Tensor, pcld_bu: torch.Tensor,
        pomega: torch.Tensor, psec: torch.Tensor, ptau: torch.Tensor,
        pcgaz_bu: torch.Tensor, ppizaz_bu: torch.Tensor, ptauz_bu: torch.Tensor):
    """Cloudy-sky adding matrix — port of swr.F90.

    Args (bottom-up unless noted; cloud fields use Fortran IKL top-down index):
        knu:       spectral interval (1..6).
        palbp:     (nlon, nsw) surface SW albedo.
        pcg:       (nlon, nsw, klev) cloud asymmetry factor.
        pcld_bu:   (nlon, klev) effective cloud fraction, bottom-up (from swu).
        pomega:    (nlon, nsw, klev) cloud SSA.
        psec:      (nlon,) secant = 1/mu0.
        ptau:      (nlon, nsw, klev) cloud optical thickness.
        pcgaz_bu/ppizaz_bu/ptauz_bu: (nlon,klev) clear-sky layer props (from
                      swclr), bottom-up (index 0 = ground).

    Returns dict (Fortran 1-based layout, PREFZ[:,:,0]=surface):
        pray1, pray2, prefz, prj, prk, prmue, ptra1, ptra2, ptrcld.
    """
    nlon, nsw, klev = pcg.shape
    dt = pcg.dtype
    dev = pcg.device
    ib = knu - 1
    palbp_k = palbp[:, ib]                         # (nlon,)

    # Cloud per-band slices (nlon, klev). Fortran indexes these at IKL
    # (top-down) inside the ZC1I loop, and at JKM1 (bottom-up) in the layer
    # loop — both directly into the (nlon,nsw,klev) array. We keep the full
    # array and index per the Fortran calls.
    # pcg_k = pcg[:, ib]  etc. computed inline below.

    # ================================================================
    # 1. Effective cloudiness ZC1I (NOVLP=1 maximum-random).
    #    swr.F90 lines 137-235. JK=1 then JK=2..KLEV, IKL=KLEV+1-JK.
    #    Reads PCG/POMEGA/PTAU/PCLD at IKL (top-down), PCGAZ/PPIZAZ/PTAUAZ at IKL.
    # ================================================================
    # ── Stage 1 precompute (vectorised over all levels) ──
    # Compute zcorae, zcorcd, zr21, zr22, zss1 for ALL levels at once, then
    # do the sequential zclear accumulation in a thin loop.
    cg_az_all = pcgaz_bu                                # (nlon, klev) bottom-up
    pz_az_all = ppizaz_bu
    tz_az_all = ptauz_bu
    cg_c_all = pcg[:, ib, :]                            # (nlon, klev) cloud g
    om_c_all = pomega[:, ib, :]
    tau_c_all = ptau[:, ib, :]
    cld_all = pcld_bu                                   # (nlon, klev) cloud frac
    zcorae_all = (1.0 - pz_az_all * cg_az_all * cg_az_all) * tz_az_all * psec.unsqueeze(-1)
    zcorcd_all = (1.0 - om_c_all * cg_c_all * cg_c_all) * tau_c_all * psec.unsqueeze(-1)
    zr21_all = torch.exp(-torch.clamp(zcorae_all, max=200.0))
    zr22_all = torch.exp(-torch.clamp(zcorcd_all, max=200.0))
    # ZSS1 = PCLD*(1-ZR21*ZR22) + (1-PCLD)*(1-ZR21) for all levels.
    zss1_all = cld_all * (1.0 - zr21_all * zr22_all) + (1.0 - cld_all) * (1.0 - zr21_all)
    # ── Stage 1 sequential accumulation (thin loop, ~5 ops/iter) ──
    # zclear/zcloud are carried across iterations (sequential overlap sweep);
    # rebinding these Python vars is autograd-safe. Only the per-level ZC1I
    # value is collected into a list (avoids in-place index writes on a tensor
    # that carries gradient history through zss1_all -> zcorae_all).
    # zc1i_list collects in Fortran JK order: JK=1 -> IKL=KLEV -> zc1i index
    # klev-1; ...; JK=KLEV -> IKL=1 -> zc1i index 0. So list[k] holds the value
    # for index (klev-1-k); we reverse on assembly.
    zclear = torch.ones(nlon, dtype=dt, device=dev)
    zcloud = torch.zeros(nlon, dtype=dt, device=dev)
    zc1i_list: list = []
    for jk in range(1, klev + 1):                  # Fortran JK=1..KLEV
        ikl = klev + 1 - jk                         # top-down level 1..KLEV
        zss1 = zss1_all[:, ikl - 1]                 # precomputed
        ziclear = 1.0 / (1.0 - torch.clamp(zcloud, max=1.0 - _REPSEC))
        zclear = zclear * (1.0 - torch.maximum(zss1, zcloud)) * ziclear
        zc1i_list.append(1.0 - zclear)              # Fortran ZC1I(:,:,IKL)
        zcloud = zss1
    # Reverse so assembled position i == index i; pad TOA boundary
    # ZC1I(:,:,KLEV+1)=0 (set at init in Fortran) with a zero column.
    zc1i_core = torch.stack(list(reversed(zc1i_list)), dim=1)   # (nlon, klev)
    zc1i = torch.cat(
        [zc1i_core, torch.zeros(nlon, 1, dtype=dt, device=dev)],
        dim=1)                                     # (nlon, klev+1)

    # ================================================================
    # 2. Delta-Eddington layer R/T + adding with cloud coupling
    #    (autograd-compatible).
    # ================================================================
    pcg_b = pcg[:, ib, :]          # (nlon, klev) this band's cloud g
    pomega_b = pomega[:, ib, :]
    ptau_b = ptau[:, ib, :]
    # PREFZ has a sequential dependency: PREFZ(:,:,JK) depends on
    # PREFZ(:,:,JK-1). We carry ``prefz_prev`` ([nlon,2]) across iterations
    # (rebinding, not in-place) and collect per-level R/T/PREFZ/ZTR/PRMUE
    # into Python lists, then torch.stack at the end. This preserves the
    # exact arithmetic of the Fortran adding equations while keeping every
    # intermediate a fresh tensor in the autograd graph.
    prefz_list: list = []                     # per-level [nlon,2], idx 0..klev
    pray1_list: list = []                     # per-level [nlon], idx 0..klev-1
    pray2_list: list = []
    ptra1_list: list = []
    ptra2_list: list = []
    ztr_list: list = []                       # per-level [nlon,2], idx 0..klev-1
    prmue_tail: list = []                     # per-level [nlon], idx 1..klev

    # Level-0 init (Fortran PREFZ(JL,1,1)=PALBP, PREFZ(JL,2,1)=PALBP).
    prefz_prev = torch.stack([palbp_k, palbp_k], dim=1)   # [nlon,2]
    prefz_list.append(prefz_prev)

    for jk_f in range(2, klev + 2):
        jkm1 = jk_f - 1
        zc1i_jk = zc1i[:, jk_f - 1]; jk_bu = jkm1 - 1
        cg_az = pcgaz_bu[:, jk_bu]; pz_az = ppizaz_bu[:, jk_bu]; tz_az = ptauz_bu[:, jk_bu]
        cg_c = pcg_b[:, jk_bu]; om_c = pomega_b[:, jk_bu]; tau_c = ptau_b[:, jk_bu]
        cld = pcld_bu[:, jk_bu]
        zmue = (1.0 - zc1i_jk) * psec + zc1i_jk * 1.66
        prmue_jk = 1.0 / zmue
        zbmu0 = 0.5 - 0.75 * cg_az * prmue_jk
        zden = 1.0 + (1.0 - pz_az + zbmu0 * pz_az) * tz_az * zmue + \
               (1.0 - pz_az) * (1.0 - pz_az + 2.0 * zbmu0 * pz_az) * tz_az * tz_az * zmue * zmue
        ptra1_jk = 1.0 / zden
        pray1_jk = zbmu0 * pz_az * tz_az * zmue * ptra1_jk
        zbmu1 = 0.5 - 0.75 * cg_az * 0.5
        zden1 = 1.0 + (1.0 - pz_az + zbmu1 * pz_az) * tz_az * 2.0 + \
                (1.0 - pz_az) * (1.0 - pz_az + 2.0 * zbmu1 * pz_az) * tz_az * tz_az * 4.0
        ptra2_jk = 1.0 / zden1
        pray2_jk = zbmu1 * pz_az * tz_az * 2.0 * ptra2_jk
        zdiv1 = 1.0 / om_c; zdiv2 = 1.0 / pz_az
        zto1 = tau_c * zdiv1 + tz_az * zdiv2; zr21 = tau_c + tz_az; zr22 = tau_c / zr21
        zgg = zr22 * cg_c + (1.0 - zr22) * cg_az
        zw = torch.where((om_c == 1.0) & (pz_az == 1.0), torch.ones_like(om_c), zr21 / zto1)
        prefz_prev0 = prefz_prev[:, 0]
        prefz_prev1 = prefz_prev[:, 1]
        zre1, ztr1, zre2, ztr2 = _swde(zgg, prefz_prev0, prmue_jk, zto1, zw, novlp=1)
        zrr = 1.0 / (1.0 - pray2_jk * prefz_prev0).clamp(min=1e-300)
        prefz0_new = (1.0 - cld) * (pray1_jk + prefz_prev0 * ptra1_jk * ptra2_jk * zrr) + cld * zre2
        ztr0_jk = cld * ztr2 + ptra1_jk * zrr * (1.0 - cld)
        prefz1_new = (1.0 - cld) * (pray1_jk + prefz_prev1 * ptra1_jk * ptra2_jk) + cld * zre1
        ztr1_jk = cld * ztr1 + ptra1_jk * (1.0 - cld)
        # collect this level's results.
        pray1_list.append(pray1_jk)
        pray2_list.append(pray2_jk)
        ptra1_list.append(ptra1_jk)
        ptra2_list.append(ptra2_jk)
        ztr_list.append(torch.stack([ztr0_jk, ztr1_jk], dim=1))
        prmue_tail.append(prmue_jk)
        # carry updated PREFZ to next level (new tensor, not in-place).
        prefz_prev = torch.stack([prefz0_new, prefz1_new], dim=1)
        prefz_list.append(prefz_prev)
    # After-loop: PRMUE(:,1) (Fortran JK=1, our index 0) and PTRCLD.
    zmue1 = (1.0 - zc1i[:, 0]) * psec + zc1i[:, 0] * 1.66
    prmue_head = 1.0 / zmue1
    ptrcld = 1.0 - zc1i[:, 0]

    # Assemble level-indexed tensors from the collected lists.
    # PREFZ: stack [nlon,2] per level on a new level axis -> permute to
    # (nlon,2,klev+1). Every level 0..klev was set explicitly above.
    _prefz = torch.stack(prefz_list, dim=1)              # (nlon, klev+1, 2)
    prefz_f = _prefz.permute(0, 2, 1).contiguous()       # (nlon, 2, klev+1)
    # PRAY1/2: indices 0..klev-1 from loop, index klev stays 0 (Fortran init).
    _zero_col = torch.zeros(nlon, 1, dtype=dt, device=dev)
    pray1_f = torch.cat([torch.stack(pray1_list, dim=1), _zero_col], dim=1)
    pray2_f = torch.cat([torch.stack(pray2_list, dim=1), _zero_col], dim=1)
    # PTRA1/2: indices 0..klev-1 from loop, index klev stays 1 (Fortran init).
    _one_col = torch.ones(nlon, 1, dtype=dt, device=dev)
    ptra1_f = torch.cat([torch.stack(ptra1_list, dim=1), _one_col], dim=1)
    ptra2_f = torch.cat([torch.stack(ptra2_list, dim=1), _one_col], dim=1)
    # ZTR: (nlon,2,klev), all levels 0..klev-1 set in loop.
    _ztr = torch.stack(ztr_list, dim=1)                  # (nlon, klev, 2)
    ztr_f = _ztr.permute(0, 2, 1).contiguous()           # (nlon, 2, klev)
    # PRMUE: index 0 from after-loop head, indices 1..klev from loop tail.
    prmue_f = torch.stack([prmue_head] + prmue_tail, dim=1)   # (nlon, klev+1)

    # ================================================================
    # 3. PRJ/PRK adding matrix (same structure as swclr).
    #    INU1=3 for NSW=6. KNU<=INU1 (bands 1-3): JAJ=2 only.
    #    KNU>INU1 (bands 4-6): JAJ=1,2.
    #
    #    PRJ has a sequential dependency: PRJ(:,JAJ,ikl-1) depends on
    #    PRJ(:,JAJ,iklp1-1) (the level above). We carry ``prj_prev`` across
    #    iterations and build each used JAJ row as a [nlon,klev+1] tensor,
    #    then assemble the full (nlon,6,klev+1) matrix with torch.stack over
    #    rows. Unused rows are constant zeros (no in-place writes anywhere).
    # ================================================================
    inu1 = 3                                        # NSW==6
    ones_n = torch.ones(nlon, dtype=dt, device=dev)
    prj_rows = [torch.zeros(nlon, klev + 1, dtype=dt, device=dev) for _ in range(6)]
    prk_rows = [torch.zeros(nlon, klev + 1, dtype=dt, device=dev) for _ in range(6)]
    if knu <= inu1:
        # KNU<=INU1 (bands 1-3): JAJ=2 only, col=0.
        jaj = 2
        col = 0
        ztr_col = ztr_f[:, col, :]                  # (nlon, klev) — slicing read
        prefz_col = prefz_f[:, col, :]              # (nlon, klev+1) — slicing read
        # PRJ(:,JAJ,KLEV)=1, downward scan PRJ(ikl-1)=PRJ_prev*ZTR(ikl-1).
        prj_scan: list = [None] * (klev + 1)        # index = ikl-1 (0..klev)
        prk_scan: list = [None] * (klev + 1)
        prj_scan[klev] = ones_n
        prk_scan[klev] = prefz_col[:, klev]
        prj_prev = ones_n
        for jk in range(1, klev + 1):               # Fortran JK=1..KLEV
            ikl = klev + 1 - jk                     # top-down level
            iklp1 = ikl + 1
            zre11 = prj_prev * ztr_col[:, ikl - 1]
            prj_scan[ikl - 1] = zre11
            prk_scan[ikl - 1] = zre11 * prefz_col[:, ikl - 1]
            prj_prev = zre11
        prj_rows[jaj - 1] = torch.stack(prj_scan, dim=1)
        prk_rows[jaj - 1] = torch.stack(prk_scan, dim=1)
    else:
        # KNU>INU1 (bands 4-6): JAJ=1,2, each using its own ZTR/PREFZ column.
        for jaj in (1, 2):
            col = jaj - 1
            ztr_col = ztr_f[:, col, :]              # (nlon, klev) — slicing read
            prefz_col = prefz_f[:, col, :]          # (nlon, klev+1) — slicing read
            prj_scan: list = [None] * (klev + 1)
            prk_scan: list = [None] * (klev + 1)
            prj_scan[klev] = ones_n
            prk_scan[klev] = prefz_col[:, klev]
            prj_prev = ones_n
            for jk in range(1, klev + 1):           # Fortran JK=1..KLEV
                ikl = klev + 1 - jk
                iklp1 = ikl + 1
                zre11 = prj_prev * ztr_col[:, ikl - 1]
                prj_scan[ikl - 1] = zre11
                prk_scan[ikl - 1] = zre11 * prefz_col[:, ikl - 1]
                prj_prev = zre11
            prj_rows[jaj - 1] = torch.stack(prj_scan, dim=1)
            prk_rows[jaj - 1] = torch.stack(prk_scan, dim=1)
    prj_f = torch.stack(prj_rows, dim=1)            # (nlon, 6, klev+1)
    prk_f = torch.stack(prk_rows, dim=1)            # (nlon, 6, klev+1)

    return {
        "pray1": pray1_f, "pray2": pray2_f, "prefz": prefz_f,
        "prj": prj_f, "prk": prk_f, "prmue": prmue_f,
        "ptra1": ptra1_f, "ptra2": ptra2_f, "ptrcld": ptrcld,
    }


__all__ = ["swr"]
