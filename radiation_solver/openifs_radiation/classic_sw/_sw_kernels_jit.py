"""JIT-compiled hot-path kernels for the SW radiation scheme.

These are leaf-level functions called thousands of times inside the level
loops of swclr/swr/sw1s/swni. JIT compilation removes the per-call Python
dispatch overhead (~10 us each). All are bit-exact with their pure-Python
counterparts (verified in the test suite).
"""
import torch


@torch.jit.script
def horner_batched(coef: torch.Tensor, pu: torch.Tensor) -> torch.Tensor:
    """Horner for (kabs,7) coefficients at (nlon,kabs) points -> (nlon,kabs)."""
    c = coef.unsqueeze(0)          # (1, kabs, 7)
    p = pu.unsqueeze(-1)           # (nlon, kabs, 1)
    zr = c[..., 6]
    zr = zr.expand_as(pu)
    for k in range(5, -1, -1):
        zr = c[..., k] + p[..., 0] * zr
    return zr


@torch.jit.script
def horner_vec(coef: torch.Tensor, pu: torch.Tensor) -> torch.Tensor:
    """Horner for (7,) coefficients at arbitrary-shape pu."""
    zr = coef[6].expand_as(pu)
    for k in range(5, -1, -1):
        zr = coef[k] + pu * zr
    return zr


@torch.jit.script
def swde_jit(gg: torch.Tensor, pref: torch.Tensor, prmuz: torch.Tensor,
             pto1: torch.Tensor, pw: torch.Tensor):
    """Delta-Eddington layer R/T — JIT (ECMWF novlp<5 path). Returns (pre1,ptr1,pre2,ptr2)."""
    replog = 1e-12
    zdt = 2.0 / 3.0
    zdt2 = 2.0 * zdt
    zff = gg * gg
    zgp = gg / (1.0 + gg)
    zpwff = 1.0 - pw * zff
    ztop = zpwff * pto1
    zwcp = (1.0 - zff) * pw / zpwff
    zx1 = 1.0 - zwcp * zgp
    zwm = 1.0 - zwcp
    zrm2 = prmuz * prmuz
    zrk = torch.sqrt(torch.clamp(3.0 * zwm * zx1, min=replog))
    zx2 = (1.0 - zrk * zrk * zrm2) * zdt2
    zrr = 1.0 / zx2
    zrp = zrk / zx1
    zalpha = zwcp * zrm2 * (1.0 + zgp * zwm) * zrr
    zbeta = zwcp * prmuz * (1.0 + 3.0 * zgp * zrm2 * zwm) * zrr
    zprmu = 1.0 / prmuz
    zzarg = -torch.clamp(ztop * zprmu, min=-200.0, max=200.0)
    zzarg2 = torch.clamp(zrk * ztop, max=200.0)
    zexmu0 = torch.exp(zzarg)
    zexkp = torch.exp(zzarg2)
    zexkm = 1.0 / zexkp
    zxp2p = 1.0 + zdt * zrp
    zxm2p = 1.0 - zdt * zrp
    zap2b = zalpha + zdt * zbeta
    zam2b = zalpha - zdt * zbeta
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


@torch.jit.script
def swr_level_sweep(zc1i: torch.Tensor,
                     pcgaz_bu: torch.Tensor, ppizaz_bu: torch.Tensor, ptauz_bu: torch.Tensor,
                     pcg_b: torch.Tensor, pomega_b: torch.Tensor, ptau_b: torch.Tensor,
                     pcld_bu: torch.Tensor, psec: torch.Tensor,
                     palbp_k: torch.Tensor, klev: int):
    """JIT-compiled swr stage-2 adding sweep (the sequential level loop).

    All inputs are (nlon, klev) bottom-up except psec/palbp_k (nlon,).
    Returns: prefz_f (nlon,2,klev+1), pray1_f, pray2_f, ptra1_f, ptra2_f
    (nlon,klev+1), ztr_f (nlon,2,klev), prmue_f (nlon,klev+1), ptrcld (nlon,).
    """
    nlon = psec.shape[0]
    prefz_f = torch.zeros(nlon, 2, klev + 1, dtype=psec.dtype, device=psec.device)
    pray1_f = torch.zeros(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    pray2_f = torch.zeros(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    ptra1_f = torch.ones(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    ptra2_f = torch.ones(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    ztr_f = torch.zeros(nlon, 2, klev, dtype=psec.dtype, device=psec.device)
    prmue_f = torch.zeros(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    # Boundary
    prefz_f[:, 0, 0] = palbp_k
    prefz_f[:, 1, 0] = palbp_k
    for jk_f in range(2, klev + 2):
        jkm1 = jk_f - 1
        zc1i_jk = zc1i[:, jk_f - 1]
        jk_bu = jkm1 - 1
        cg_az = pcgaz_bu[:, jk_bu]
        pz_az = ppizaz_bu[:, jk_bu]
        tz_az = ptauz_bu[:, jk_bu]
        cg_c = pcg_b[:, jk_bu]
        om_c = pomega_b[:, jk_bu]
        tau_c = ptau_b[:, jk_bu]
        cld = pcld_bu[:, jk_bu]
        zmue = (1.0 - zc1i_jk) * psec + zc1i_jk * 1.66
        prmue_val = 1.0 / zmue
        prmue_f[:, jk_f - 1] = prmue_val
        # PRAY1/PTRA1
        zbmu0 = 0.5 - 0.75 * cg_az * prmue_val
        zden = 1.0 + (1.0 - pz_az + zbmu0 * pz_az) * tz_az * zmue + \
               (1.0 - pz_az) * (1.0 - pz_az + 2.0 * zbmu0 * pz_az) * tz_az * tz_az * zmue * zmue
        ptra1_val = 1.0 / zden
        ptra1_f[:, jkm1 - 1] = ptra1_val
        pray1_f[:, jkm1 - 1] = zbmu0 * pz_az * tz_az * zmue * ptra1_val
        # PRAY2/PTRA2
        zbmu1 = 0.5 - 0.75 * cg_az * 0.5
        zden1 = 1.0 + (1.0 - pz_az + zbmu1 * pz_az) * tz_az * 2.0 + \
                (1.0 - pz_az) * (1.0 - pz_az + 2.0 * zbmu1 * pz_az) * tz_az * tz_az * 4.0
        ptra2_val = 1.0 / zden1
        ptra2_f[:, jkm1 - 1] = ptra2_val
        pray2_f[:, jkm1 - 1] = zbmu1 * pz_az * tz_az * 2.0 * ptra2_val
        # Cloud blend
        zdiv1 = 1.0 / om_c
        zdiv2 = 1.0 / pz_az
        zto1 = tau_c * zdiv1 + tz_az * zdiv2
        zr21 = tau_c + tz_az
        zr22 = tau_c / zr21
        zgg = zr22 * cg_c + (1.0 - zr22) * cg_az
        zw = torch.where((om_c == 1.0) & (pz_az == 1.0),
                         torch.ones_like(om_c), zr21 / zto1)
        # Inline SWDE
        zre1, ztr1, zre2, ztr2 = swde_jit(zgg, prefz_f[:, 0, jkm1 - 1], prmue_val, zto1, zw)
        # Adding
        zrr = 1.0 / (1.0 - pray2_f[:, jkm1 - 1] * prefz_f[:, 0, jkm1 - 1]).clamp(min=1e-300)
        prefz_f[:, 0, jk_f - 1] = (1.0 - cld) * (
            pray1_f[:, jkm1 - 1] + prefz_f[:, 0, jkm1 - 1] * ptra1_val *
            ptra2_val * zrr) + cld * zre2
        ztr_f[:, 0, jkm1 - 1] = cld * ztr2 + ptra1_val * zrr * (1.0 - cld)
        prefz_f[:, 1, jk_f - 1] = (1.0 - cld) * (
            pray1_f[:, jkm1 - 1] + prefz_f[:, 1, jkm1 - 1] * ptra1_val *
            ptra2_val) + cld * zre1
        ztr_f[:, 1, jkm1 - 1] = cld * ztr1 + ptra1_val * (1.0 - cld)
    # PRMUE(:,:,1), PTRCLD
    zmue1 = (1.0 - zc1i[:, 0]) * psec + zc1i[:, 0] * 1.66
    prmue_f[:, 0] = 1.0 / zmue1
    ptrcld = 1.0 - zc1i[:, 0]
    return prefz_f, pray1_f, pray2_f, ptra1_f, ptra2_f, ztr_f, prmue_f, ptrcld


@torch.jit.script
def swclr_level_sweep(zc0i: torch.Tensor,
                       pcgaz: torch.Tensor, ppizaz: torch.Tensor, ptauz: torch.Tensor,
                       psec: torch.Tensor, palbp_k: torch.Tensor, klev: int):
    """JIT-compiled swclr stage-3 clear-sky adding sweep (sequential level loop).

    All inputs are (nlon, klev) bottom-up except psec/palbp_k (nlon,) and
    zc0i (nlon, klev+1) top-down.
    Returns: prefz_f (nlon,2,klev+1), pray1_f, pray2_f, ptra1_f, ptra2_f
    (nlon,klev+1), ztr_f (nlon,2,klev), prmu0_f (nlon,klev+1), ptrclr (nlon,).
    """
    nlon = psec.shape[0]
    prefz_f = torch.zeros(nlon, 2, klev + 1, dtype=psec.dtype, device=psec.device)
    pray1_f = torch.zeros(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    pray2_f = torch.zeros(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    ptra1_f = torch.ones(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    ptra2_f = torch.ones(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    ztr_f = torch.zeros(nlon, 2, klev, dtype=psec.dtype, device=psec.device)
    prmu0_f = torch.zeros(nlon, klev + 1, dtype=psec.dtype, device=psec.device)
    prefz_f[:, 0, 0] = palbp_k
    prefz_f[:, 1, 0] = palbp_k
    for jk_f in range(2, klev + 2):
        jkm1 = jk_f - 1
        zc0i_jk = zc0i[:, jk_f - 1]
        jk_bu = jkm1 - 1
        cg = pcgaz[:, jk_bu]; pz = ppizaz[:, jk_bu]; tz = ptauz[:, jk_bu]
        zmue = (1.0 - zc0i_jk) * psec + zc0i_jk * 1.66
        prmu0_val = 1.0 / zmue
        prmu0_f[:, jk_f - 1] = prmu0_val
        zbmu0 = 0.5 - 0.75 * cg * prmu0_val
        zden = 1.0 + (1.0 - pz + zbmu0 * pz) * tz * zmue + \
               (1.0 - pz) * (1.0 - pz + 2.0 * zbmu0 * pz) * tz * tz * zmue * zmue
        ptra1_val = 1.0 / zden
        ptra1_f[:, jkm1 - 1] = ptra1_val
        pray1_f[:, jkm1 - 1] = zbmu0 * pz * tz * zmue * ptra1_val
        zbmu1 = 0.5 - 0.75 * cg * 0.5
        zden1 = 1.0 + (1.0 - pz + zbmu1 * pz) * tz * 2.0 + \
                (1.0 - pz) * (1.0 - pz + 2.0 * zbmu1 * pz) * tz * tz * 4.0
        ptra2_val = 1.0 / zden1
        ptra2_f[:, jkm1 - 1] = ptra2_val
        pray2_f[:, jkm1 - 1] = zbmu1 * pz * tz * 2.0 * ptra2_val
        zrr = 1.0 / (1.0 - pray2_f[:, jkm1 - 1] * prefz_f[:, 0, jkm1 - 1]).clamp(min=1e-300)
        prefz_f[:, 0, jk_f - 1] = pray1_f[:, jkm1 - 1] + \
            prefz_f[:, 0, jkm1 - 1] * ptra1_val * ptra2_val * zrr
        ztr_f[:, 0, jkm1 - 1] = ptra1_val * zrr
        prefz_f[:, 1, jk_f - 1] = pray1_f[:, jkm1 - 1] + \
            prefz_f[:, 1, jkm1 - 1] * ptra1_val * ptra2_val
        ztr_f[:, 1, jkm1 - 1] = ptra1_val
    zmue1 = (1.0 - zc0i[:, 0]) * psec + zc0i[:, 0] * 1.66
    prmu0_f[:, 0] = 1.0 / zmue1
    ptrclr = 1.0 - zc0i[:, 0]
    return prefz_f, pray1_f, pray2_f, ptra1_f, ptra2_f, ztr_f, prmu0_f, ptrclr


@torch.jit.script
def swni_jabs_sweep(
    pcg_b: torch.Tensor, pomega_b: torch.Tensor, ptau_b: torch.Tensor,
    ztauz: torch.Tensor, zpzaz: torch.Tensor, zcgaz: torch.Tensor,
    zray1: torch.Tensor, zray2: torch.Tensor, ztra1: torch.Tensor, ztra2: torch.Tensor,
    zrmue: torch.Tensor, pcld_for_swr: torch.Tensor,
    pwv_td: torch.Tensor, pqs_td: torch.Tensor,
    pud: torch.Tensor, paki: torch.Tensor,
    palbd_k: torch.Tensor, klev: int, ib: int):
    """JIT-compiled swni phase-2 JABS double loop + phase-3 ZRJ/ZRK build.

    Returns zrj (nlon,6,klev+1), zrk (nlon,6,klev+1).
    Columns 0,1 must be pre-filled with clear pseudo-fluxes by the caller.
    """
    nlon = palbd_k.shape[0]
    dt = palbd_k.dtype
    dev = palbd_k.device
    nlev1 = klev + 1
    zrj = torch.zeros(nlon, 6, nlev1, dtype=dt, device=dev)
    zrk = torch.zeros(nlon, 6, nlev1, dtype=dt, device=dev)
    ztr_local = torch.zeros(nlon, 2, klev, dtype=dt, device=dev)
    zrefz_local = torch.zeros(nlon, 2, nlev1, dtype=dt, device=dev)
    paki_k_store = torch.zeros(nlon, dtype=dt, device=dev)
    jn = 2
    for jabs in range(1, 3):
        zrefz_local[:, 0, 0] = palbd_k
        zrefz_local[:, 1, 0] = palbd_k
        ztr_local.zero_()
        paki_k_store = paki[:, jabs - 1, ib]
        for jk_f in range(2, klev + 2):
            jkm1 = jk_f - 1
            ikl = klev + 1 - jkm1
            pud_idx = ikl
            pwv_l = pwv_td[:, ikl - 1].clamp(min=1e-12)
            pqs_l = pqs_td[:, ikl - 1]
            cld_l = pcld_for_swr[:, jkm1 - 1]
            pud_jabs = pud[:, jabs - 1, pud_idx]
            if jabs == 1:
                zbb = pud_jabs * pqs_l / pwv_l
                zcneb = torch.clamp(cld_l, min=1e-12, max=1.0 - 1e-12)
                zaa = torch.clamp((pud_jabs - zcneb * zbb) / (1.0 - zcneb), min=1e-12)
            else:
                zaa = pud_jabs
                zbb = zaa
            om_c = pomega_b[:, jkm1 - 1]
            tau_c = ptau_b[:, jkm1 - 1]
            zw = om_c
            zto1 = tau_c / zw + ztauz[:, jkm1 - 1] / zpzaz[:, jkm1 - 1] + zbb * paki_k_store
            zr21 = tau_c + ztauz[:, jkm1 - 1]
            zr22 = tau_c / zr21
            zgg = zr22 * pcg_b[:, jkm1 - 1] + (1.0 - zr22) * zcgaz[:, jkm1 - 1]
            zw = zr21 / zto1
            # Inline SWDE
            zre1, ztr1, zre2, ztr2 = swde_jit(zgg, zrefz_local[:, 0, jkm1 - 1],
                                               zrmue[:, jk_f - 1], zto1, zw)
            # ZS, ZG
            zchks = torch.clamp(paki_k_store * zaa * 1.66, max=200.0)
            zchkg = torch.clamp(paki_k_store * zaa / zrmue[:, jk_f - 1].clamp(min=1e-10), max=200.0)
            zs = torch.exp(-zchks)
            zg = torch.exp(-zchkg)
            zsg = zs * zg
            zrr = 1.0 / (1.0 - zray2[:, jkm1 - 1] * zrefz_local[:, 0, jkm1 - 1]).clamp(min=1e-300)
            zrefz_local[:, 1, jk_f - 1] = (1.0 - cld_l) * (
                zray1[:, jkm1 - 1] + zrefz_local[:, 1, jkm1 - 1] * ztra1[:, jkm1 - 1] *
                ztra2[:, jkm1 - 1]) * zsg + cld_l * zre1
            ztr_local[:, 1, jkm1 - 1] = cld_l * ztr1 + ztra1[:, jkm1 - 1] * zg * (1.0 - cld_l)
            zrefz_local[:, 0, jk_f - 1] = (1.0 - cld_l) * (
                zray1[:, jkm1 - 1] + zrefz_local[:, 0, jkm1 - 1] * ztra1[:, jkm1 - 1] *
                ztra2[:, jkm1 - 1] * zrr) * zsg + cld_l * zre2
            ztr_local[:, 0, jkm1 - 1] = cld_l * ztr2 + ztra1[:, jkm1 - 1] * zrr * zg * (1.0 - cld_l)
        # ZRJ/ZRK for JREF=1,2
        for jref in range(1, 3):
            jn = jn + 1
            zrj[:, jn - 1, klev] = 1.0
            zrk[:, jn - 1, klev] = zrefz_local[:, jref - 1, klev]
            for jk in range(1, klev + 1):
                jkl = klev + 1 - jk
                jklp1 = jkl + 1
                zre11 = zrj[:, jn - 1, jklp1 - 1] * ztr_local[:, jref - 1, jkl - 1]
                zrj[:, jn - 1, jkl - 1] = zre11
                zrk[:, jn - 1, jkl - 1] = zre11 * zrefz_local[:, jref - 1, jkl - 1]
    return zrj, zrk


@torch.jit.script
def sw1s_flux_sweep(
    apad: torch.Tensor, bpad: torch.Tensor, d_tab: torch.Tensor,
    rxpo3: torch.Tensor, nexpo3_val: int,
    zrj: torch.Tensor, zrk: torch.Tensor,
    zrj0: torch.Tensor, zrk0: torch.Tensor,
    zrmue: torch.Tensor, zrmu0: torch.Tensor,
    pud: torch.Tensor, poz_td: torch.Tensor,
    pclear: torch.Tensor, rsun_k: torch.Tensor,
    palbp_k: torch.Tensor, ptrcld: torch.Tensor, ptrclr: torch.Tensor,
    knu: int, klev: int):
    """JIT-compiled sw1s downward+upward flux sweep with inline Horner + swuvo3.

    Returns pfd, pfu, pcd, pcu (nlon, klev+1), psudu1 (nlon,).
    """
    nlon = pclear.shape[0]
    dt = pclear.dtype
    dev = pclear.device
    nlev1 = klev + 1
    jaj = 2
    kkind = [1, 2, 1, 2]
    i = knu - 1

    pfd = torch.zeros(nlon, nlev1, dtype=dt, device=dev)
    pfu = torch.zeros(nlon, nlev1, dtype=dt, device=dev)
    pcd = torch.zeros(nlon, nlev1, dtype=dt, device=dev)
    pcu = torch.zeros(nlon, nlev1, dtype=dt, device=dev)

    zw = torch.zeros(nlon, 4, dtype=dt, device=dev)
    zo = torch.zeros(nlon, 2, dtype=dt, device=dev)
    pfd[:, klev] = ((1.0 - pclear) * zrj[:, jaj - 1, klev]
                    + pclear * zrj0[:, jaj - 1, klev]) * rsun_k
    pcd[:, klev] = zrj0[:, jaj - 1, klev] * rsun_k

    # O3 coefficients for swuvo3 inline — always define (use dummy if no O3)
    if nexpo3_val > 0:
        coef_o3 = rxpo3[i, 0, :nexpo3_val]
        expo_o3 = rxpo3[i, 1, :nexpo3_val]
    else:
        coef_o3 = torch.zeros(1, dtype=dt, device=dev)
        expo_o3 = torch.zeros(1, dtype=dt, device=dev)
    n_o3 = nexpo3_val  # capture for loop bound

    # Precompute apad/bpad/d rows for the 2 absorbers (1=H2O, 2=UMG)
    a1 = apad[i, 0, :]; b1 = bpad[i, 0, :]; d1 = d_tab[i, 0]
    a2 = apad[i, 1, :]; b2 = bpad[i, 1, :]; d2 = d_tab[i, 1]

    # Downward sweep
    zrtmp = torch.zeros(nlon, dtype=dt, device=dev)
    zclrtmp = torch.zeros(nlon, dtype=dt, device=dev)
    zr_last = torch.zeros(nlon, 4, dtype=dt, device=dev)
    zt_last = torch.zeros(nlon, 2, dtype=dt, device=dev)
    for jk in range(1, klev + 1):
        ikl = klev + 1 - jk
        zre = 1.0 / zrmue[:, ikl - 1].clamp(min=1e-10)
        zr0 = 1.0 / zrmu0[:, ikl - 1].clamp(min=1e-10)
        pud_idx = klev + 1 - ikl
        poz_idx = klev - ikl
        pud_h2o = pud[:, 0, pud_idx]; pud_co2 = pud[:, 1, pud_idx]
        poz = poz_td[:, poz_idx]
        zw[:, 0] = zw[:, 0] + pud_h2o * zre
        zw[:, 1] = zw[:, 1] + pud_co2 * zre
        zw[:, 2] = zw[:, 2] + pud_h2o * zr0
        zw[:, 3] = zw[:, 3] + pud_co2 * zr0
        zo[:, 0] = zo[:, 0] + poz * zre
        zo[:, 1] = zo[:, 1] + poz * zr0
        # Inline SWTT1 for 4 absorbers [1,2,1,2]
        for ja_idx in range(4):
            ia = kkind[ja_idx] - 1
            a_c = apad[i, ia, :]; b_c = bpad[i, ia, :]; zd_c = d_tab[i, ia]
            zu = zw[:, ja_idx]
            zr1 = a_c[6].expand_as(zu)
            for kk in range(5, -1, -1): zr1 = a_c[kk] + zu * zr1
            zr2 = b_c[6].expand_as(zu)
            for kk in range(5, -1, -1): zr2 = b_c[kk] + zu * zr2
            zr_last[:, ja_idx] = (zr1 / zr2) * (1.0 - zd_c) + zd_c
        # Inline SWUVO3 for 2 paths
        for jz in range(2):
            if n_o3 > 0:
                pu_o3 = zo[:, jz]
                pu_exp = pu_o3.unsqueeze(-1) * (expo_o3 * -1.0)
                zt_last[:, jz] = (coef_o3 * torch.exp(pu_exp)).sum(dim=-1)
            else:
                zt_last[:, jz] = 0.0
        zrtmp = zr_last[:, 0] * zr_last[:, 1] * zt_last[:, 0] * zrj[:, jaj - 1, ikl - 1]
        zclrtmp = zr_last[:, 2] * zr_last[:, 3] * zt_last[:, 1] * zrj0[:, jaj - 1, ikl - 1]
        pfd[:, ikl - 1] = ((1.0 - pclear) * zrtmp + pclear * zclrtmp) * rsun_k
        pcd[:, ikl - 1] = zclrtmp * rsun_k

    psudu1 = ((1.0 - pclear) * (zr_last[:, 0] * zr_last[:, 1] * zt_last[:, 0] * ptrcld)
              + pclear * (zr_last[:, 2] * zr_last[:, 3] * zt_last[:, 1] * ptrclr)) * rsun_k
    pfu[:, 0] = ((1.0 - pclear) * zrtmp * palbp_k + pclear * zclrtmp * palbp_k) * rsun_k
    pcu[:, 0] = zclrtmp * palbp_k * rsun_k

    # Upward sweep
    for jk in range(2, klev + 2):
        ikm1 = jk - 1
        pud_idx = klev + 1 - ikm1
        poz_idx = klev - ikm1
        pud_h2o = pud[:, 0, pud_idx]; pud_co2 = pud[:, 1, pud_idx]
        poz = poz_td[:, poz_idx] if poz_idx >= 0 else torch.zeros(nlon, dtype=dt, device=dev)
        zw[:, 0] = zw[:, 0] + pud_h2o * 1.66
        zw[:, 1] = zw[:, 1] + pud_co2 * 1.66
        zw[:, 2] = zw[:, 2] + pud_h2o * 1.66
        zw[:, 3] = zw[:, 3] + pud_co2 * 1.66
        zo[:, 0] = zo[:, 0] + poz * 1.66
        zo[:, 1] = zo[:, 1] + poz * 1.66
        for ja_idx in range(4):
            ia = kkind[ja_idx] - 1
            a_c = apad[i, ia, :]; b_c = bpad[i, ia, :]; zd_c = d_tab[i, ia]
            zu = zw[:, ja_idx]
            zr1 = a_c[6].expand_as(zu)
            for kk in range(5, -1, -1): zr1 = a_c[kk] + zu * zr1
            zr2 = b_c[6].expand_as(zu)
            for kk in range(5, -1, -1): zr2 = b_c[kk] + zu * zr2
            zr_last[:, ja_idx] = (zr1 / zr2) * (1.0 - zd_c) + zd_c
        for jz in range(2):
            if n_o3 > 0:
                pu_o3 = zo[:, jz]
                pu_exp = pu_o3.unsqueeze(-1) * (expo_o3 * -1.0)
                zt_last[:, jz] = (coef_o3 * torch.exp(pu_exp)).sum(dim=-1)
            else:
                zt_last[:, jz] = 0.0
        zrtmp = zr_last[:, 0] * zr_last[:, 1] * zt_last[:, 0] * zrk[:, jaj - 1, jk - 1]
        zclrtmp = zr_last[:, 2] * zr_last[:, 3] * zt_last[:, 1] * zrk0[:, jaj - 1, jk - 1]
        pfu[:, jk - 1] = ((1.0 - pclear) * zrtmp + pclear * zclrtmp) * rsun_k
        pcu[:, jk - 1] = zclrtmp * rsun_k

    return pfd, pfu, pcd, pcu, psudu1
