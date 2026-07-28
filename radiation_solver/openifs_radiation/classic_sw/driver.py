"""Complete 6-band SW solver — faithful port of sw.F90 + swclr + sw1s + swni.

Fortran call chain:  SW -> SWU -> { SW1S (bands 1-3) | SWNI (bands 4-6) }
                               -> SWCLR (clear-sky R/T adding)
                               -> SWTT (Pade gas transmission)
"""
from __future__ import annotations
import numpy as np, torch
from pathlib import Path
from .swde import swde as _swde
from .swtt import swtt as _swtt_import

_TABLE_DIR = Path(__file__).resolve().parent.parent / "rrtm_lw" / "tables"
SOLAR_CONSTANT = 1361.0
_tables = {}
_dev_cache = {}

def _get(key, dev):
    ck = (key, str(dev))
    t = _dev_cache.get(ck)
    if t is None:
        t = _tables[key].to(dev)
        _dev_cache[ck] = t
    return t

def _load():
    if not _tables:
        for n in ["apad","bpad","d","rsun","rray"]:
            _tables[n] = torch.from_numpy(np.load(str(_TABLE_DIR/f"sw_{n}.npy"))).to(torch.float64)

def _swtt_pade(knu, ka, pu):
    _load()
    a,b,d = _tables["apad"],_tables["bpad"],_tables["d"]
    i,j = knu-1, ka-1
    zr1 = a[i,j,6]+torch.zeros_like(pu)
    for k in range(5,-1,-1): zr1 = a[i,j,k]+pu*zr1
    zr2 = b[i,j,6]+torch.zeros_like(pu)
    for k in range(5,-1,-1): zr2 = b[i,j,k]+pu*zr2
    zd = d[i,j]
    return (zr1/zr2)*(1.0-zd)+zd

# suswn.F90 table constants (verified bit-identical to rad_ref_sw_swu_params).
_SWU_RPDH1 = 1.9                # H2O pressure power
_SWU_RPDU1 = 1.75               # CO2 pressure power
_SWU_RPNH = 1.6775567570514103e-07
_SWU_RPNU = 1.0262051544635577e-06
_SWU_RTH2O = 273.0
_SWU_RTDH2O = 0.45
_SWU_RTUMG = 273.0
_SWU_RTDUMG = 0.375
# swu uses REPSCQ as the H2O floor (YOECLD actually, but YOERDU%REPSCQ in swu).
_SWU_REPSCQ = 1.0e-12


def _swu(p_half, p_full, temp, q_h2o, co2_vmr, o3, mu0_raw, pcldsw=None):
    """Absorber amounts (PUD) + geometry + cloud-overlap factors — port of swu.F90.

    Returns per-layer absorber amounts (NOT cumulative) plus geometry, matching
    the Fortran ``SWU`` output contract exactly:

    * ``pud``  — (nlon, 5, nlev+1), top-down, float64.
                 ``pud[:, :, 0]`` is the TOA boundary = 0; ``pud[:, :, 1..nlev]``
                 are per-layer column amounts. Indices: 1=H2O, 2=CO2 (UMG),
                 3=O3 (left 0 here — sw1s/swni use POZ via swuvo3), 4/5 are the
                 H2O split into the 1.6078-scaled vapour / dry components that
                 swu derives via ZFPPW.
    * ``pdsig`` — (nlon, nlev), top-down σ-layer thickness (dp/p_surf).
    * ``prmu``  — (nlon,), cos(sza) passed through (Fortran PRMU=PRMU0).
    * ``psec``  — (nlon,), secant = 1/mu0 (Fortran PSEC = 1/PRMU, no curvature).
    * ``pclear`` — (nlon,), clear-sky column fraction (only if pcldsw given).
    * ``pcld``   — (nlon, nlev), effective cloud fraction normalised to the
                  cloudy column (only if pcldsw given). top-down.
    * ``paki``   — (nlon, 2, 6), grey gas absorption coefficients for swni
                  (only if pcldsw given); bands 1-3 are 0 (sw1s does not use it).

    Three semantic fixes vs the previous implementation:
      1. NO RRAE curvature correction here — that lives in radina (the layer
         above). Fortran swu sets PRMU=PRMU0, PSEC=1/PRMU0 verbatim.
      2. co2 is expected as a volume mixing ratio (VMR), matching Fortran
         PCARDI; no MMR→VMR conversion.
      3. PUD is per-layer (matching Fortran PUD output); sw1s/swni perform the
         cumulative sum themselves (ZW += PUD*ZRE). Previously this returned a
         cumsum, which produced a "cumulative-of-cumulative" downstream.
    """
    nlon, nlev1 = p_half.shape
    nlev = nlev1 - 1
    dt = p_half.dtype
    dev = p_half.device

    # Geometry — Fortran swu lines 137-145: PRMU=PRMU0, PSEC=1/PRMU0.
    prmu = mu0_raw.clamp(min=1.0e-10)
    psec = 1.0 / prmu

    # PPSOL = surface pressure (half-level, bottom). PDSIG = dp/p_surf.
    ppsol = p_half[:, -1:]
    pdsig = (p_half[:, 1:] - p_half[:, :-1]) / ppsol.clamp(min=1.0)

    # PCARDI is the CO2 VMR; ZWH2O = MAX(PWV, REPSCQ).
    pcardi = co2_vmr[:, :nlev]
    ptave = temp.clamp(min=100.0)
    zwh2o = q_h2o[:, :nlev].clamp(min=_SWU_REPSCQ)

    # Per-layer pressure-power differences.
    p_top = p_half[:, :-1]
    p_bot = p_half[:, 1:]
    zdsh2o = p_bot ** _SWU_RPDH1 - p_top ** _SWU_RPDH1
    zdsco2 = p_bot ** _SWU_RPDU1 - p_top ** _SWU_RPDU1
    zrth = (_SWU_RTH2O / ptave) ** _SWU_RTDH2O
    zrtu = (_SWU_RTUMG / ptave) ** _SWU_RTDUMG

    pud_h2o = _SWU_RPNH * zdsh2o * zwh2o * zrth
    pud_co2 = _SWU_RPNU * zdsco2 * pcardi * zrtu
    ziwh2o = 1.0 / (1.0 + 0.608 * zwh2o)
    zfppw = 1.6078 * zwh2o * ziwh2o
    pud_4 = pud_h2o * zfppw
    pud_5 = pud_h2o * (1.0 - zfppw)

    pud = torch.zeros(nlon, 5, nlev + 1, dtype=dt, device=dev)
    pud[:, 0, 1:] = pud_h2o
    pud[:, 1, 1:] = pud_co2
    pud[:, 2, 1:] = 0.0
    pud[:, 3, 1:] = pud_4
    pud[:, 4, 1:] = pud_5

    out = dict(pud=pud, pdsig=pdsig, prmu=prmu, psec=psec)

    if pcldsw is not None:
        # --- Cloud overlap (NOVLP=1 maximum-random) → PCLEAR, PCLD ---
        # swu.F90 lines 150-260. ZC1J built bottom-up from PCLDSW; PCLEAR =
        # 1 - ZC1J(:,:,1); PCLD(JK) = PCLDSW(JK)/(1-PCLEAR) clamped [0,1].
        # PCLDSW in Fortran is top-down (swu reads PCLDSW(JL,JKL) with
        # JKL=KLEV+1-JK). pcldsw here is top-down (index 0=TOA).
        _REPSEC = 1.0e-12
        zclear = torch.ones(nlon, dtype=dt, device=dev)
        zcloud = torch.zeros(nlon, dtype=dt, device=dev)
        zc1j = torch.zeros(nlon, nlev + 1, dtype=dt, device=dev)
        # Fortran JK=1..KLEV, JKL=KLEV+1-JK. ZCLOUD starts 0, ZCLEAR starts 1.
        # ZC1J(JL,JKL) = 1 - ZCLEAR; NOVLP=1: ZCLEAR *= (1-max(PCLDSW(JKL),ZCLOUD))/(1-min(ZCLOUD,1-REPSEC))
        for jk in range(1, nlev + 1):
            jkl = nlev + 1 - jk                  # top-down 1-based
            pc = pcldsw[:, jkl - 1]              # PCLDSW(JL,JKL), top-down
            ziclear = 1.0 / (1.0 - torch.clamp(zcloud, max=1.0 - _REPSEC))
            zclear = zclear * (1.0 - torch.maximum(pc, zcloud)) * ziclear
            zc1j[:, jkl - 1] = 1.0 - zclear
            zcloud = pc
        pclear = 1.0 - zc1j[:, 0]                # ZC1J(:,:,1) -> our index 0
        # PCLD(JK) = PCLDSW(JK)/(1-PCLEAR) clamped [0,1], for JK=1..KLEV.
        # Fortran PCLDSW(JL,JK) here uses JK directly (NOT JKL) — see swu
        # lines 252-258: PCLD(JL,JK)=PCLDSW(JL,JK)*ZICLOUD. So PCLD is indexed
        # the same way as PCLDSW (top-down, matching our pcldsw).
        zicloud = 1.0 / (1.0 - pclear).clamp(min=1e-300)
        pcld = (pcldsw * zicloud.unsqueeze(1)).clamp(min=0.0, max=1.0)
        out["pclear"] = pclear
        out["pcld"] = pcld

        # --- PAKI: grey gas absorption coefficients (swni JABS loop) ---
        # swu lines 262-275: ZUD(JA) = cumsum(PUD(:,:,JA)) * PSEC; then for
        # JNU=INUIR..NSW (4..6 for NSW=6): PAKI(JA,JNU) = -log(SWTT1(ZUD))/ZUD.
        # PAKI for bands 1-3 stays 0 (sw1s doesn't use it).
        from .swtt import swtt1 as _swtt1
        # ZUD cumulative from TOA: ZUD(JA) = sum of PUD(JA,1..JK) * PSEC.
        # Fortran accumulates ZUD bottom-up but the cumulative sum is the same
        # total regardless of direction; ZUD here is the column-total * PSEC.
        zud_total = torch.zeros(nlon, 2, dtype=dt, device=dev)
        zud_total[:, 0] = pud_h2o.sum(dim=1) * psec     # H2O column total
        zud_total[:, 1] = pud_co2.sum(dim=1) * psec     # CO2 column total
        paki = torch.zeros(nlon, 2, 6, dtype=dt, device=dev)
        inuir = 4                                        # NSW==6
        for jnu in range(inuir, 7):                     # JNU=4,5,6
            # SWTT1(KNU=JNU, KABS=2, KKIND=[1,2], PU=ZUD) -> ZR (nlon,2)
            zr = _swtt1(jnu, 2, [1, 2], zud_total)      # (nlon, 2)
            for ja in range(2):
                # PAKI(JA,JNU) = -log(ZR(JA)) / ZUD(JA); guard ZUD>0.
                zud_safe = zud_total[:, ja].clamp(min=1e-300)
                paki[:, ja, jnu - 1] = -torch.log(zr[:, ja].clamp(min=1e-300)) / zud_safe
        out["paki"] = paki

    return out


def _swu_cumsum_topdown(pud):
    """Cumulative column amount from TOA downward, matching how sw1s/swni build
    ZW = ZW + PUD(:,:,IKL)*ZRE during their downward sweep.

    Given per-layer ``pud`` (nlon, 5, nlev+1) with pud[:,:,0]=0 (TOA boundary),
    returns ``zud`` (nlon, 5, nlev+1) where zud[:,:,k] = sum of pud[:,:,1..k]
    (i.e. the cumulative amount encountered from TOA down to level k). This is
    the quantity the downstream gas-transmission calls need; sw1s/swni in Fortran
    fold it into their own JK loop, here we expose it once.
    """
    zud = torch.zeros_like(pud)
    zud[:, :, 1:] = torch.cumsum(pud[:, :, 1:], dim=2)
    return zud

def _swclr(knu, palbp, zdsig, zsec):
    _load()
    nlon,nlev=zdsig.shape; dt=zdsig.dtype; dev=zdsig.device; repsct=1e-8
    rray=_get("rray", dev); ib=knu-1
    cd=torch.cumsum(zdsig,dim=1)
    rcl=torch.zeros(nlon,nlev,dtype=dt,device=dev)
    for k in range(6): rcl+=rray[ib,k]*cd**(k+1)
    rc=torch.cat([torch.zeros(nlon,1,dtype=dt,device=dev),rcl],dim=1)
    prayl=(rc[:,1:]-rc[:,:-1]).clamp(min=0)
    ptaua_z=prayl; ppiza_z=torch.full((nlon,nlev),1.0-repsct,dtype=dt,device=dev)
    pcga_z=torch.zeros(nlon,nlev,dtype=dt,device=dev)
    prefz=torch.zeros(nlon,nlev+1,dtype=dt,device=dev)
    ptra1=torch.ones(nlon,nlev+1,dtype=dt,device=dev)
    ptra2=torch.ones(nlon,nlev+1,dtype=dt,device=dev)
    pray1=torch.zeros(nlon,nlev+1,dtype=dt,device=dev)
    pray2=torch.zeros(nlon,nlev+1,dtype=dt,device=dev)
    prefz[:,nlev]=palbp
    for jk in range(nlev-1,-1,-1):
        zb=0.5-0.75*pcga_z[:,jk]*zsec
        zd=1.0+(1.0-ppiza_z[:,jk]+zb*ppiza_z[:,jk])*ptaua_z[:,jk]*zsec+\
             (1.0-ppiza_z[:,jk])*(1.0-ppiza_z[:,jk]+2.0*zb*ppiza_z[:,jk])*ptaua_z[:,jk]**2*zsec**2
        ptra1[:,jk]=1.0/zd.clamp(min=1e-30)
        pray1[:,jk]=zb*ppiza_z[:,jk]*ptaua_z[:,jk]*zsec*ptra1[:,jk]
        zbd=0.5-0.75*pcga_z[:,jk]*0.5
        zd1=1.0+(1.0-ppiza_z[:,jk]+zbd*ppiza_z[:,jk])*ptaua_z[:,jk]*2.0+\
              (1.0-ppiza_z[:,jk])*(1.0-ppiza_z[:,jk]+2.0*zbd*ppiza_z[:,jk])*ptaua_z[:,jk]**2*4.0
        ptra2[:,jk]=1.0/zd1.clamp(min=1e-30)
        pray2[:,jk]=zbd*ppiza_z[:,jk]*ptaua_z[:,jk]*2.0*ptra2[:,jk]
        zrr=1.0/(1.0-pray2[:,jk]*prefz[:,jk+1]).clamp(min=1e-30)
        prefz[:,jk]=pray1[:,jk]+prefz[:,jk+1]*ptra1[:,jk]*ptra2[:,jk]*zrr
    return dict(ptaua_z=ptaua_z,ppiza_z=ppiza_z,pcga_z=pcga_z,
                prefz=prefz,ptra1=ptra1,ptra2=ptra2,pray1=pray1,pray2=pray2)

def _sw1s(knu,clr,zud,zsec,palbp,sflux):
    nlon=zsec.shape[0]; dt=zsec.dtype; dev=zsec.device
    nlev1=zud.shape[2]; nlev=nlev1-1
    ptr=torch.ones(nlon,nlev1,dtype=dt,device=dev)
    for jk in range(nlev1): ptr[:,jk]=_swtt_pade(knu,3,zud[:,0,jk])
    fdir=sflux.unsqueeze(1)*ptr; fd=fdir.clone()
    fu=torch.zeros(nlon,nlev1,dtype=dt,device=dev)
    fu[:,nlev]=palbp*fdir[:,nlev]
    for jk in range(nlev-1,-1,-1):
        fu[:,jk]=fu[:,jk+1]*clr["ptra2"][:,jk]+fdir[:,jk]*clr["pray1"][:,jk]
    return fd,fu

def _swni(knu,clr,zud,zsec,palbp,sflux):
    nlon=zsec.shape[0]; dt=zsec.dtype; dev=zsec.device
    nlev1=zud.shape[2]; nlev=nlev1-1
    ph=torch.ones(nlon,nlev1,dtype=dt,device=dev)
    pu=torch.ones(nlon,nlev1,dtype=dt,device=dev)
    for jk in range(nlev1):
        ph[:,jk]=_swtt_pade(knu,1,zud[:,1,jk])
        pu[:,jk]=_swtt_pade(knu,2,zud[:,2,jk])
    fdir=sflux.unsqueeze(1)*(ph*pu); fd=fdir.clone()
    fu=torch.zeros(nlon,nlev1,dtype=dt,device=dev)
    fu[:,nlev]=palbp*fdir[:,nlev]
    for jk in range(nlev-1,-1,-1):
        fu[:,jk]=fu[:,jk+1]*clr["ptra2"][:,jk]+fdir[:,jk]*clr["pray1"][:,jk]
    return fd,fu

def sw(nlev, pressure_half=None, temperature=None, q_h2o=None, q_co2=None, q_o3=None, albedo_diffuse=None, albedo_direct=None, mu0=None, p_half=None, temp=None, albd=None, albp=None):
    _load()
    p_half = p_half if p_half is not None else pressure_half
    temp = temp if temp is not None else temperature
    q_h2o = q_h2o
    # q_co2 is a VMR (matches Fortran PCARDI); no MMR→VMR conversion.
    q_co2 = q_co2
    q_o3 = q_o3
    albd = albedo_diffuse if albedo_diffuse is not None else (albd if albd is not None else albedo_diffuse)
    albp = albedo_direct if albedo_direct is not None else (albp if albp is not None else albedo_direct)
    nlon=p_half.shape[0]; dt=p_half.dtype; dev=p_half.device; nsw=6; nl=p_half.shape[1]-1
    rsun=_get("rsun", dev)
    pf=0.5*(p_half[:,:-1]+p_half[:,1:])
    swu=_swu(p_half,pf,temp,q_h2o,q_co2,q_o3,mu0)
    zdsig,psec,prmu,pud=swu["pdsig"],swu["psec"],swu["prmu"],swu["pud"]
    # Solar flux at TOA per band: RSUN*mu0*SOLAR_CONSTANT (no RRAE — swu layer).
    # NOTE: the previous code multiplied by the RRAE-corrected zamu0; the
    # curvature correction belongs in radina, not here, so we use the raw mu0.
    zamu0 = prmu
    zud = _swu_cumsum_topdown(pud)
    fdb=torch.zeros(nsw,nlon,nl+1,dtype=dt,device=dev)
    fub=torch.zeros_like(fdb)
    for jb in range(nsw):
        knu=jb+1
        clr=_swclr(knu,albp[:,jb],zdsig,psec)
        sf=SOLAR_CONSTANT*rsun[jb]*zamu0
        if jb<3: fd,fu=_sw1s(knu,clr,zud,psec,albp[:,jb],sf)
        else: fd,fu=_swni(knu,clr,zud,psec,albp[:,jb],sf)
        fdb[jb]=fd; fub[jb]=fu
    fdt=fdb.sum(dim=0); fut=fub.sum(dim=0)
    return dict(fd_band=fdb,fu_band=fub,fd_total=fdt,fu_total=fut,
                fd_surf=fdt[:,nl],fu_toa=fut[:,0])

def sw_solver(p_half, temp, q_h2o, q_co2_vmr, mu0, albd, albp,
              pcldsw, aer, poz, pcg, pomega, ptau, pqs, rray=None, rsun=None):
    """End-to-end SW solver — the callpar-path entry point.

    Mirrors the RADLSW dump contract: it takes the per-band optical quantities
    that radlsw assembles (cloud fraction PCLDSW, aerosol PAER, ozone POZ, cloud
    optics ZTAU/ZOMEGA/ZCG, saturation humidity PQS) plus the physical state
    (pressure, temperature, humidity, CO2 VMR, solar geometry, albedo) and runs
    the full SW chain: SWU -> {SW1S (bands 1-3) | SWNI (bands 4-6)}.

    Args (top-down torch tensors, index 0 = TOA unless noted):
        p_half:    (nlon, klev+1) half-level pressure (Pa).
        temp:      (nlon, klev) full-level temperature (K).
        q_h2o:     (nlon, klev) specific humidity (kg/kg) — PWV (top-down).
        q_co2_vmr: (nlon, klev) CO2 volume mixing ratio (PCARDI).
        mu0:       (nlon,) cos(solar zenith angle) — NOTE: this is the
                   RRAE-corrected ZAMU0 from radina in production; sw_solver
                   uses it as-is (swu does not re-apply curvature).
        albd/albp: (nlon, nsw) surface SW albedo (diffuse/direct).
        pcldsw:    (nlon, klev) SW cloud fraction (top-down, from radlsw).
        aer:       (nlon, 6, klev) Tegen aerosol optical thickness (top-down).
        poz:       (nlon, klev) O3 column amount in cm-atm (top-down, from radozc).
        pcg/pomega/ptau: (nlon, nsw, klev) cloud asymmetry/SSA/optical-thickness
                   per band (top-down, from radlsw SW cloud optics).
        pqs:       (nlon, klev) saturation specific humidity (top-down).
        rray/rsun: optional pre-loaded tables (for testing).

    Returns dict:
        fd_total/fu_total: (nlon, klev+1) net SW down/up flux (top-down).
        fd_band/fu_band:   (nsw, nlon, klev+1) per-band fluxes.
        fd_surf, fu_toa:   (nlon,) surface down / TOA up.
    """
    _load()
    nlon, nlev1 = p_half.shape
    klev = nlev1 - 1
    dt = p_half.dtype
    dev = p_half.device
    nsw = 6
    if rray is None:
        rray = _get("rray", dev)
    if rsun is None:
        rsun = _get("rsun", dev)

    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    # SWU: absorber amounts + geometry + cloud overlap (pclear/pcld) + paki.
    swu = _swu(p_half, p_full, temp, q_h2o, q_co2_vmr, None, mu0, pcldsw=pcldsw)
    pud = swu["pud"]          # (nlon, 5, klev+1) per-layer top-down
    pclear = swu["pclear"]    # (nlon,)
    pcld = swu["pcld"]        # (nlon, klev) effective cloud fraction (top-down)
    paki = swu["paki"]        # (nlon, 2, 6) grey absorption coefficients

    from .sw1s import sw1s as _sw1s_full
    from .swni import swni as _swni_full

    fdb = torch.zeros(nsw, nlon, nlev1, dtype=dt, device=dev)
    fub = torch.zeros_like(fdb)
    fcdb = torch.zeros_like(fdb)
    fcub = torch.zeros_like(fdb)
    for jb in range(nsw):
        knu = jb + 1
        if knu <= 3:
            out = _sw1s_full(
                knu, aer, albp, swu["pdsig"], swu["psec"], swu["prmu"],
                pud, poz, pcg, pomega, ptau, pcld, pclear, None, rray, rsun)
            fdb[jb] = out["pfd"]; fub[jb] = out["pfu"]
            fcdb[jb] = out["pcd"]; fcub[jb] = out["pcu"]
        else:
            out = _swni_full(
                knu, aer, paki, albp, pcg, pcld, pclear, swu["pdsig"],
                pomega, poz, swu["prmu"], swu["psec"], ptau, pud,
                q_h2o, pqs, rray, rsun)
            fdb[jb] = out["pfdown"]; fub[jb] = out["pfup"]
            fcdb[jb] = out["pcdown"]; fcub[jb] = out["pcup"]

    # The band solvers (sw1s/swni) output fluxes already weighted by RSUN(KNU)
    # (the spectral fraction), but NOT by the incident solar irradiance
    # PSCT*mu0 (= PRII0 in radlsw). Scale to W/m² here.
    sf = (SOLAR_CONSTANT * swu["prmu"]).unsqueeze(0).unsqueeze(-1)  # (1,nlon,1)
    fdb = fdb * sf; fub = fub * sf
    fcdb = fcdb * sf; fcub = fcub * sf
    # sw1s/swni emit fluxes in Fortran's bottom-up order (index 0 = surface,
    # klev = TOA). Flip to top-down (index 0 = TOA) to match the package
    # convention used by radina and the heating-rate formula.
    fdb = fdb.flip(dims=[2]); fub = fub.flip(dims=[2])
    fcdb = fcdb.flip(dims=[2]); fcub = fcub.flip(dims=[2])
    fdt = fdb.sum(dim=0); fut = fub.sum(dim=0)
    return dict(
        fd_band=fdb, fu_band=fub, fd_total=fdt, fu_total=fut,
        fc_band=fcdb, fcu_band=fcub,
        fd_surf=fdt[:, klev], fu_toa=fut[:, 0],
        pclear=pclear, psec=swu["psec"], prmu=swu["prmu"],
    )


__all__ = ["sw", "sw_solver", "swde"]
