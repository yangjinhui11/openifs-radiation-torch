"""Batched torch port of RRTM TAUMOL gas-optics kernels (bands 1-16).

All 16 bands with full batched tensor ops. Transmission lookup via TRANS/BPADE.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TABLE_DIR = _HERE / "tables"

def _load_band(band: int, name: str, shape: tuple) -> torch.Tensor:
    arr = np.load(str(_TABLE_DIR / f"band{band}_{name}.npy"))
    if arr.shape != shape: raise ValueError(f"band{band}_{name}: {arr.shape} != {shape}")
    return torch.from_numpy(arr).to(torch.float64)

def _load_common(name: str, shape: tuple) -> torch.Tensor:
    return torch.from_numpy(np.load(str(_TABLE_DIR / f"{name}.npy"))).to(torch.float64)

def _trunc_int(x): return x.to(torch.int64)

def _lin(tbl, idx, frac):
    idx = idx.clamp(1, tbl.shape[0] - 1).long()
    return tbl[idx - 1] + frac.unsqueeze(-1) * (tbl[idx] - tbl[idx - 1])

def _planck_frac(fracrefa, cA, cB, refrat, oneminus, sm=8.0):
    """Planck fraction interpolation in species dimension (JPL, Z_FPL).
    
    Fortran pattern:
      Z_SPECCOMB_PLANCK = cA + refrat * cB
      Z_SPECPARM_PLANCK = cA / Z_SPECCOMB_PLANCK
      Z_SPECMULT_PLANCK = sm * Z_SPECPARM_PLANCK
      JPL = 1 + INT(Z_SPECMULT_PLANCK)
      Z_FPL = MOD(Z_SPECMULT_PLANCK, 1.0)
      PFRAC = FRACREFA(IG, JPL) + Z_FPL * (FRACREFA(IG, JPL+1) - FRACREFA(IG, JPL))
    
    fracrefa: (ng, nspa) — Planck fraction table
    Returns: (nlev, nlon, ng) — interpolated Planck fractions
    """
    sc_pl = cA + refrat * cB
    sp_pl = (cA / sc_pl.clamp(min=1e-30)).clamp(max=float(oneminus) if isinstance(oneminus, float) else oneminus.max())
    sm_pl = sm * sp_pl
    jpl = (1 + _trunc_int(sm_pl)).clamp(1, fracrefa.shape[1] - 1).long()  # 1-based, ∈ [1, nspa-1]
    fpl = sm_pl - (jpl - 1).to(sm_pl.dtype)  # MOD, ∈ [0, 1)
    
    ng = fracrefa.shape[0]
    n, nl = cA.shape
    dv = cA.device
    
    # fracrefa is (ng, nspa), index with jpl-1 and jpl
    idx0 = jpl.unsqueeze(-1).expand(n, nl, ng) - 1  # 0-based
    idx1 = idx0 + 1  # next column, already within bounds
    
    # Gather using advanced indexing
    fa0 = fracrefa[torch.arange(ng, device=dv).view(1, 1, ng), idx0.clamp(0, fracrefa.shape[1] - 1)]
    fa1 = fracrefa[torch.arange(ng, device=dv).view(1, 1, ng), idx1.clamp(0, fracrefa.shape[1] - 1)]
    
    return fa0 + fpl.unsqueeze(-1) * (fa1 - fa0)

def _g(tbl, idx0, offset, nmax=None):
    if nmax is None: nmax = tbl.shape[0]
    return tbl[(idx0 + offset).clamp(0, nmax - 1)]

def _a4(absa, i0, i1, f00, f10, f01, f11):
    nmax = absa.shape[0]
    i0 = i0.clamp(1, nmax - 2).long()  # safe for i0 and i0+1
    i1 = i1.clamp(1, nmax - 2).long()
    return (f00.unsqueeze(-1)*absa[i0-1] + f10.unsqueeze(-1)*absa[i0]
            + f01.unsqueeze(-1)*absa[i1-1] + f11.unsqueeze(-1)*absa[i1])

def _bsw(sp, fs, f00, f10, f01, f11):
    lo=sp<0.125; hi=sp>0.875; md=~lo&~hi
    zl=fs-1.; zl4=zl**4; zh=-fs; zh4=zh**4
    k0l=zl4; k1l=1.-zl-2.*zl4; k2l=zl+zl4
    k0h=zh4; k1h=1.-zh-2.*zh4; k2h=zh+zh4
    def m3(a,b,c): return torch.where(lo,a,torch.where(md,b,c))
    return {
        "f000":m3(k0l*f00,(1.-fs)*f00,k0h*f00),
        "f100":m3(k1l*f00,fs*f00,k1h*f00),
        "f200":m3(k2l*f00,torch.zeros_like(f00),k2h*f00),
        "f010":m3(k0l*f10,(1.-fs)*f10,k0h*f10),
        "f110":m3(k1l*f10,fs*f10,k1h*f10),
        "f210":m3(k2l*f10,torch.zeros_like(f10),k2h*f10),
        "u3":lo|hi, "lo":lo, "hi":hi}

def _ba(absa, idx0, nspa, w):
    nsa=absa.shape[0]
    def g(o): return _g(absa,idx0,o,nsa)
    # lo branch (3 ref atms, 3-point): indices 0,1,2, nspa,nspa+1,nspa+2
    t3_lo=(w["f000"].unsqueeze(-1)*g(0)+w["f100"].unsqueeze(-1)*g(1)+w["f200"].unsqueeze(-1)*g(2)
           +w["f010"].unsqueeze(-1)*g(nspa)+w["f110"].unsqueeze(-1)*g(nspa+1)+w["f210"].unsqueeze(-1)*g(nspa+2))
    # hi branch: SHIFTED indices (-1,0,1, nspa-1,nspa,nspa+1) matches Fortran
    t3_hi=(w["f000"].unsqueeze(-1)*g(1)+w["f100"].unsqueeze(-1)*g(0)+w["f200"].unsqueeze(-1)*g(-1)
           +w["f010"].unsqueeze(-1)*g(nspa+1)+w["f110"].unsqueeze(-1)*g(nspa)+w["f210"].unsqueeze(-1)*g(nspa-1))
    # md: 2 ref atms, 2-point
    t2=(w["f000"].unsqueeze(-1)*g(0)+w["f100"].unsqueeze(-1)*g(1)
        +w["f010"].unsqueeze(-1)*g(nspa)+w["f110"].unsqueeze(-1)*g(nspa+1))
    return torch.where(w["lo"].unsqueeze(-1), t3_lo,
           torch.where(w["hi"].unsqueeze(-1), t3_hi, t2))

def _bl(band,ng,nspa,sm,cA,cB,rAB,rAB1,f00,f10,f01,f11,
        jp,jt,jt1,sf,srf,ins,ff,frf,inf,sr,fr,absa,ta):
    """Generic multi-reference atmosphere band (nspa>1).
    
    Fortran always computes: Z_SPECCOMB * (FAC00/FAC10 term) + Z_SPECCOMB1 * (FAC01/FAC11 term)
    The two terms use DIFFERENT spec_comb values (sc vs sc1), so they must be separated.
    """
    sc=cA+rAB*cB; sp=(cA/sc).clamp(max=0.999999); sm_=sm*sp
    js=1+_trunc_int(sm_); fs=sm_-(js-1).to(sc.dtype)  # MOD(sm_,1) ∈ [0,1)
    sc1=cA+rAB1*cB; sp1=(cA/sc1).clamp(max=0.999999); sm1=sm*sp1
    js1=1+_trunc_int(sm1); fs1=sm1-(js1-1).to(sc.dtype)  # MOD(sm1,1) ∈ [0,1)
    i0=((jp-1)*5+(jt-1))*nspa+js-1; i1=(jp*5+(jt1-1))*nspa+js1-1
    w0=_bsw(sp,fs,f00,f10,f00,f10); w1=_bsw(sp1,fs1,f01,f11,f01,f11)
    b0=_ba(absa,i0,nspa,w0); b1=_ba(absa,i1,nspa,w1)
    ts=sf.unsqueeze(-1)*_lin(sr,ins,srf); tf=ff.unsqueeze(-1)*_lin(fr,inf,frf)
    return sc.unsqueeze(-1)*b0+sc1.unsqueeze(-1)*b1+ts+tf+ta

# ===== BAND 1 =====
def rrtm_taumol1(pavel, tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1,
                 colh2o, laytrop, selffac, selffrac, indself,
                 minorfrac, indminor, scaleminorn2, colbrd):
    ng=10; n,nl=pavel.shape; dt=pavel.dtype; dv=pavel.device
    aa=_load_band(1,"absa",(65,ng)).to(dv); ab=_load_band(1,"absb",(235,ng)).to(dv)
    sr=_load_band(1,"selfref",(10,ng)).to(dv); fr=_load_band(1,"forref",(4,ng)).to(dv)
    fa=_load_band(1,"fracrefa",(ng,)).to(dv); fb=_load_band(1,"fracrefb",(ng,)).to(dv)
    km=_load_band(1,"ka_mn2",(19,ng)).to(dv); kbm=_load_band(1,"kb_mn2",(19,ng)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    i0=((jp-1)*5+(jt-1))+1; i1=(jp*5+(jt1-1))+1
    a4=_a4(aa,i0,i1,fac00,fac10,fac01,fac11)
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    cr=torch.where(pavel<250.,1.-.15*(250.-pavel)/154.4,torch.ones_like(pavel))
    sc=colbrd*scaleminorn2; tn=sc.unsqueeze(-1)*_lin(km,indminor,minorfrac)
    Tl=cr.unsqueeze(-1)*(colh2o.unsqueeze(-1)*a4+ts+tf+tn)+tauaer[...,0:1]
    T=torch.where(lo.unsqueeze(-1),Tl,T); P=torch.where(lo.unsqueeze(-1),fa.view(1,1,ng),P)
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u=_a4(ab,i0u,i1u,fac00,fac10,fac01,fac11)
    tfu=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac); cru=1.-.15*pavel/95.6
    tnu=sc.unsqueeze(-1)*_lin(kbm,indminor,minorfrac)
    Tu=cru.unsqueeze(-1)*(colh2o.unsqueeze(-1)*a4u+tfu+tnu)+tauaer[...,0:1]
    T=torch.where((~lo).unsqueeze(-1),Tu,T); P=torch.where((~lo).unsqueeze(-1),fb.view(1,1,ng).expand(n,nl,ng),P)
    return T,P

# ===== BAND 2 =====
def rrtm_taumol2(pavel, coldry, tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1,
                 colh2o, laytrop, selffac, selffrac, indself):
    ng=12; n,nl=pavel.shape; dt=pavel.dtype; dv=pavel.device
    aa=_load_band(2,"absa",(65,ng)).to(dv); ab=_load_band(2,"absb",(235,ng)).to(dv)
    sr=_load_band(2,"selfref",(10,ng)).to(dv); fr=_load_band(2,"forref",(4,ng)).to(dv)
    fa=_load_band(2,"fracrefa",(ng,)).to(dv); fb=_load_band(2,"fracrefb",(ng,)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    i0=((jp-1)*5+(jt-1))+1; i1=(jp*5+(jt1-1))+1
    a4=_a4(aa,i0,i1,fac00,fac10,fac01,fac11)
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac); cr=1.-.05*(pavel-100.)/900.
    Tl=cr.unsqueeze(-1)*(colh2o.unsqueeze(-1)*a4+ts+tf)+tauaer[...,1:2]
    T=torch.where(lo.unsqueeze(-1),Tl,T)
    # Upper atmosphere (uses ABSB, no self continuum)
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u=_a4(ab,i0u,i1u,fac00,fac10,fac01,fac11)
    tfu=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    Tu=colh2o.unsqueeze(-1)*a4u+tfu+tauaer[...,1:2]
    T=torch.where((~lo).unsqueeze(-1),Tu,T)
    P=torch.where(lo.unsqueeze(-1),fa.view(1,1,ng).expand(n,nl,ng),
                                 fb.view(1,1,ng).expand(n,nl,ng))
    return T,P

# ===== BANDS 3-9 =====
def rrtm_taumol3(tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                 colh2o, colco2, coln2o, coldry, laytrop,
                 selffac, selffrac, indself,
                 rat_h2oco2, rat_h2oco2_1, minorfrac, indminor):
    """Band 3: 500-630 cm-1 (low-H2O,CO2+N2O; high-H2O,CO2+N2O).
    
    Direct Fortran port with N2O minor gas (KA_MN2O/KB_MN2O) and N2O column adjustment.
    """
    ng=16; nspa=9; nspb=5; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(3,"absa",(585,ng)).to(dv); ab=_load_band(3,"absb",(1175,ng)).to(dv)
    sr=_load_band(3,"selfref",(10,ng)).to(dv); fr=_load_band(3,"forref",(4,ng)).to(dv)
    fa=_load_band(3,"fracrefa",(ng,9)).to(dv); fb=_load_band(3,"fracrefb",(ng,5)).to(dv)
    ka=_load_band(3,"ka_mn2o",(9,19,ng)).to(dv); kb=_load_band(3,"kb_mn2o",(5,19,ng)).to(dv)
    
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    # Ref ratios (CHI_MLS is 1-based padded: chi[S, L] == Fortran CHI_MLS(S, L))
    refrat_planck_a = chi[1, 9] / chi[2, 9]   # CHI_MLS(1,9)/CHI_MLS(2,9)
    refrat_planck_b = chi[1, 13] / chi[2, 13]  # CHI_MLS(1,13)/CHI_MLS(2,13)
    refrat_m_a = chi[1, 3] / chi[2, 3]          # CHI_MLS(1,3)/CHI_MLS(2,3)
    refrat_m_b = chi[1, 13] / chi[2, 13]        # CHI_MLS(1,13)/CHI_MLS(2,13)
    
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    # ── Lower atmosphere ──
    # Major gas: use proven _bl for H2O+CO2
    Tl_bl = _bl(3,ng,nspa,8.,colh2o,colco2,rat_h2oco2,rat_h2oco2_1,
                fac00,fac10,fac01,fac11,jp,jt,jt1,
                selffac,selffrac,indself,forfac,forfrac,indfor,
                sr,fr,aa,torch.zeros_like(tauaer[...,2:3]))
    
    # N2O adjustment: Fortran CHI_MLS(4, K_JP+1) → 1-based chi[4, jp+1]
    jp_n2o = (jp + 1).clamp(1, 59).long()
    chi_n2o = coln2o / coldry.clamp(min=1e-30)
    zratn2o = 1.0e20 * chi_n2o / chi[4, jp_n2o]
    adjfac = torch.where(zratn2o > 1.5, 0.5 + (zratn2o - 0.5)**0.65, torch.ones_like(zratn2o))
    adjcoln2o = torch.where(zratn2o > 1.5,
        adjfac * chi[4, jp_n2o] * coldry * 1.0e-20, coln2o)
    
    # N2O minor gas: KA_MN2O(JMN2O, INDM, IG) — (9, 19, 16)
    sc_mn2o = colh2o + refrat_m_a * colco2
    sp_mn2o = (colh2o / sc_mn2o.clamp(min=1e-30)).clamp(max=0.999999)
    sm_mn2o = 8.0 * sp_mn2o
    jmn2o = (1 + _trunc_int(sm_mn2o)).clamp(1, 9).long()
    f_mn2o = sm_mn2o - (jmn2o - 1).to(dt)
    im_flat = (indminor.clamp(1, 19).long() - 1).reshape(n*nl)
    jm0_flat = (jmn2o - 1).clamp(0, 8).reshape(n*nl)
    jm1_flat = (jmn2o).clamp(0, 8).reshape(n*nl)  # 0-based JMN2O+1 (= Fortran 1-based jmn2o)
    im1_flat = (im_flat + 1).clamp(0, 18)
    zn2om1 = (ka[jm0_flat, im_flat, :] + f_mn2o.reshape(n*nl, 1) * (ka[jm1_flat, im_flat, :] - ka[jm0_flat, im_flat, :])).reshape(n, nl, ng)
    zn2om2 = (ka[jm0_flat, im1_flat, :] + f_mn2o.reshape(n*nl, 1) * (ka[jm1_flat, im1_flat, :] - ka[jm0_flat, im1_flat, :])).reshape(n, nl, ng)
    zabsn2o = zn2om1 + minorfrac.unsqueeze(-1) * (zn2om2 - zn2om1)
    
    Tl = Tl_bl + adjcoln2o.unsqueeze(-1) * zabsn2o + tauaer[...,2:3]
    T=torch.where(lo.unsqueeze(-1), Tl, T)
    
    # Planck fraction lower: FRACREFA(IG, JPL)
    P_lo = _planck_frac(fa, colh2o, colco2, refrat_planck_a, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    
    # ── Upper atmosphere ──
    # N2O adjustment for upper (same formula as lower)
    adjcoln2o_u = adjcoln2o  # same computation
    
    # Major gas: H2O+CO2, sm=4, NSPB=5, MD branch only (no BSW)
    sc=colh2o+rat_h2oco2*colco2; sp=(colh2o/sc.clamp(min=1e-30)).clamp(max=0.999999)
    sm=4.*sp; js=1+_trunc_int(sm); fs=sm-(js-1).to(dt)
    sc1=colh2o+rat_h2oco2_1*colco2; sp1=(colh2o/sc1.clamp(min=1e-30)).clamp(max=0.999999)
    sm1=4.*sp1; js1=1+_trunc_int(sm1); fs1=sm1-(js1-1).to(dt)
    i0=((jp-13)*5+(jt-1))*nspb+js-1; i1=((jp-12)*5+(jt1-1))*nspb+js1-1
    f00=(1.-fs)*fac00; f10=(1.-fs)*fac10; f01=fs*fac00; f11=fs*fac10
    f001=(1.-fs1)*fac01; f101=(1.-fs1)*fac11; f011=fs1*fac01; f111=fs1*fac11
    tfu=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac); nsa=ab.shape[0]
    
    # N2O minor gas for upper: KB_MN2O(JMN2O, INDM, IG) — (5, 19, 16) with species interpolation
    sc_mn2o_u = colh2o + refrat_m_b * colco2
    sp_mn2o_u = (colh2o / sc_mn2o_u.clamp(min=1e-30)).clamp(max=0.999999)
    sm_mn2o_u = 4.0 * sp_mn2o_u
    jmn2o_u = (1 + _trunc_int(sm_mn2o_u)).clamp(1, 5).long()
    f_mn2o_u = sm_mn2o_u - (jmn2o_u - 1).to(dt)
    # KB_MN2O is (5, 19, ng) — species * temp * g-point
    jm0_u = (jmn2o_u - 1).clamp(0, 4).reshape(n*nl)
    jm1_u = (jmn2o_u).clamp(0, 4).reshape(n*nl)  # 0-based JMN2O+1
    zn2om1_u = (kb[jm0_u, im_flat, :] + f_mn2o_u.reshape(n*nl, 1) * (kb[jm1_u, im_flat, :] - kb[jm0_u, im_flat, :])).reshape(n, nl, ng)
    zn2om2_u = (kb[jm0_u, im1_flat, :] + f_mn2o_u.reshape(n*nl, 1) * (kb[jm1_u, im1_flat, :] - kb[jm0_u, im1_flat, :])).reshape(n, nl, ng)
    zabsn2o_u = zn2om1_u + minorfrac.unsqueeze(-1) * (zn2om2_u - zn2om1_u)
    
    Tu=(sc.unsqueeze(-1)*(f00.unsqueeze(-1)*_g(ab,i0,0,nsa)+f01.unsqueeze(-1)*_g(ab,i0,1,nsa)
                   +f10.unsqueeze(-1)*_g(ab,i0,nspb,nsa)+f11.unsqueeze(-1)*_g(ab,i0,nspb+1,nsa))
        +sc1.unsqueeze(-1)*(f001.unsqueeze(-1)*_g(ab,i1,0,nsa)+f011.unsqueeze(-1)*_g(ab,i1,1,nsa)
                    +f101.unsqueeze(-1)*_g(ab,i1,nspb,nsa)+f111.unsqueeze(-1)*_g(ab,i1,nspb+1,nsa))
        +tfu + adjcoln2o_u.unsqueeze(-1)*zabsn2o_u + tauaer[...,2:3])
    T=torch.where((~lo).unsqueeze(-1), Tu, T)
    
    # Planck fraction upper: FRACREFB(IG, JPL)
    P_up = _planck_frac(fb, colh2o, colco2, refrat_planck_b, oneminus, sm=4.0)
    P=torch.where((~lo).unsqueeze(-1), P_up, P)
    
    return T,P

def rrtm_taumol4(tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                 colh2o, colco2, colo3, laytrop,
                 selffac, selffrac, indself,
                 rat_h2oco2, rat_h2oco2_1, rat_o3co2, rat_o3co2_1):
    ng=14; nspa=9; nspb=5; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(4,"absa",(585,ng)).to(dv); ab=_load_band(4,"absb",(1175,ng)).to(dv)
    sr=_load_band(4,"selfref",(10,ng)).to(dv); fr=_load_band(4,"forref",(4,ng)).to(dv)
    fa=_load_band(4,"fracrefa",(ng,9)).to(dv); fb=_load_band(4,"fracrefb",(ng,5)).to(dv)
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    # Lower: H2O+CO2, Planck fraction interpolated in species dimension
    Tl=_bl(4,ng,nspa,8.,colh2o,colco2,rat_h2oco2,rat_h2oco2_1,
           fac00,fac10,fac01,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           sr,fr,aa,tauaer[...,3:4])  # separate_sc always True
    T=torch.where(lo.unsqueeze(-1),Tl,T)
    
    # Planck fraction for lower: use reference H2O/CO2 ratio = CHI_MLS(1,11)/CHI_MLS(2,11)
    refrat_planck_lo = chi[1, 11] / chi[2, 11]  # ~0.0188/0.000355
    sc_pl_lo = colh2o + refrat_planck_lo * colco2
    sp_pl_lo = (colh2o / sc_pl_lo.clamp(min=1e-30)).clamp(max=oneminus)
    sm_pl_lo = 8.0 * sp_pl_lo
    jpl_lo = (1 + _trunc_int(sm_pl_lo)).clamp(1, 9).long()
    fpl_lo = sm_pl_lo - jpl_lo.to(dt) + 1.0
    # fa is (ng, 9), index columns (jpl_lo - 1) and jpl_lo
    fa_lo = fa[torch.arange(ng, device=dv).unsqueeze(0).unsqueeze(0).expand(n,nl,ng),
               (jpl_lo-1).unsqueeze(-1).expand(-1,-1,ng)]
    fa_lo1 = fa[torch.arange(ng, device=dv).unsqueeze(0).unsqueeze(0).expand(n,nl,ng),
                jpl_lo.clamp(1,8).unsqueeze(-1).expand(-1,-1,ng)]
    P_lo = fa_lo + fpl_lo.unsqueeze(-1) * (fa_lo1 - fa_lo)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    
    # upper O3+CO2 + emp
    sc=colo3+rat_o3co2*colco2; sp=(colo3/sc.clamp(min=1e-30)).clamp(max=0.999999); sm=4.*sp
    js=1+_trunc_int(sm); fs=sm-js.to(dt)+1.
    sc1=colo3+rat_o3co2_1*colco2; sp1=(colo3/sc1.clamp(min=1e-30)).clamp(max=0.999999); sm1=4.*sp1
    js1=1+_trunc_int(sm1); fs1=sm1-js1.to(dt)+1.
    i0=((jp-13)*5+(jt-1))*nspb+js-1; i1=((jp-12)*5+(jt1-1))*nspb+js1-1
    f00=(1.-fs)*fac00; f10=(1.-fs)*fac10; f01=fs*fac00; f11=fs*fac10
    f001=(1.-fs1)*fac01; f101=(1.-fs1)*fac11; f011=fs1*fac01; f111=fs1*fac11
    nsa=ab.shape[0]
    Tu=(sc.unsqueeze(-1)*(f00.unsqueeze(-1)*_g(ab,i0,0,nsa)+f01.unsqueeze(-1)*_g(ab,i0,1,nsa)
                   +f10.unsqueeze(-1)*_g(ab,i0,nspb,nsa)+f11.unsqueeze(-1)*_g(ab,i0,nspb+1,nsa))
        +sc1.unsqueeze(-1)*(f001.unsqueeze(-1)*_g(ab,i1,0,nsa)+f011.unsqueeze(-1)*_g(ab,i1,1,nsa)
                    +f101.unsqueeze(-1)*_g(ab,i1,nspb,nsa)+f111.unsqueeze(-1)*_g(ab,i1,nspb+1,nsa))
        +tauaer[...,3:4])
    Tu[...,7]*=0.92; Tu[...,8]*=0.88; Tu[...,9]*=1.07
    Tu[...,10]*=1.1; Tu[...,11]*=0.99; Tu[...,12]*=0.88; Tu[...,13]*=0.943
    T=torch.where((~lo).unsqueeze(-1),Tu,T)
    
    # Planck fraction for upper: reference O3/CO2 ratio = CHI_MLS(3,13)/CHI_MLS(2,13)
    refrat_planck_up = chi[3, 13] / chi[2, 13]
    sc_pl_up = colo3 + refrat_planck_up * colco2
    sp_pl_up = (colo3 / sc_pl_up.clamp(min=1e-30)).clamp(max=oneminus)
    sm_pl_up = 4.0 * sp_pl_up
    jpl_up = (1 + _trunc_int(sm_pl_up)).clamp(1, 5).long()
    fpl_up = sm_pl_up - jpl_up.to(dt) + 1.0
    # fb is (ng, 5)
    fb_up = fb[torch.arange(ng, device=dv).unsqueeze(0).unsqueeze(0).expand(n,nl,ng),
               (jpl_up-1).unsqueeze(-1).expand(-1,-1,ng)]
    fb_up1 = fb[torch.arange(ng, device=dv).unsqueeze(0).unsqueeze(0).expand(n,nl,ng),
                jpl_up.clamp(1,4).unsqueeze(-1).expand(-1,-1,ng)]
    P_up = fb_up + fpl_up.unsqueeze(-1) * (fb_up1 - fb_up)
    P=torch.where((~lo).unsqueeze(-1), P_up, P)
    
    return T,P

def rrtm_taumol5(wx, tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                 colh2o, colco2, colo3, laytrop,
                 selffac, selffrac, indself,
                 rat_h2oco2, rat_h2oco2_1, rat_o3co2, rat_o3co2_1,
                 minorfrac, indminor):
    ng=16; nspa=9; nspb=5; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(5,"absa",(585,ng)).to(dv); ab=_load_band(5,"absb",(1175,ng)).to(dv)
    sr=_load_band(5,"selfref",(10,ng)).to(dv); fr=_load_band(5,"forref",(4,ng)).to(dv)
    fa=_load_band(5,"fracrefa",(ng,9)).to(dv); fb=_load_band(5,"fracrefb",(ng,5)).to(dv)
    ccl4=_load_band(5,"ccl4",(ng,)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)

    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    refrat_planck = chi[1, 5] / chi[2, 5]  # CHI_MLS(1,5)/CHI_MLS(2,5)

    Tl=_bl(5,ng,nspa,8.,colh2o,colco2,rat_h2oco2,rat_h2oco2_1,
           fac00,fac10,fac01,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           sr,fr,aa,tauaer[...,4:5])  # separate_sc always True
    Tl=Tl+wx[...,0:1]*ccl4.view(1,1,ng)
    T=torch.where(lo.unsqueeze(-1),Tl,T)

    # Planck fraction lower: FRACREFA(IG, JPL) — was MISSING (stayed zero!)
    P_lo = _planck_frac(fa, colh2o, colco2, refrat_planck, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    # upper
    sc=colo3+rat_o3co2*colco2; sp=(colo3/sc).clamp(max=0.999999); sm=4.*sp
    js=1+_trunc_int(sm); fs=sm-js.to(dt)+1.
    sc1=colo3+rat_o3co2_1*colco2; sp1=(colo3/sc1).clamp(max=0.999999); sm1=4.*sp1
    js1=1+_trunc_int(sm1); fs1=sm1-js1.to(dt)+1.
    i0=((jp-13)*5+(jt-1))*nspb+js-1; i1=((jp-12)*5+(jt1-1))*nspb+js1-1
    f00=(1.-fs)*fac00; f10=(1.-fs)*fac10; f01=fs*fac00; f11=fs*fac10
    f001=(1.-fs1)*fac01; f101=(1.-fs1)*fac11; f011=fs1*fac01; f111=fs1*fac11
    nsa=ab.shape[0]
    Tu=(sc.unsqueeze(-1)*(f00.unsqueeze(-1)*_g(ab,i0,0,nsa)+f01.unsqueeze(-1)*_g(ab,i0,1,nsa)
                   +f10.unsqueeze(-1)*_g(ab,i0,nspb,nsa)+f11.unsqueeze(-1)*_g(ab,i0,nspb+1,nsa))
        +sc1.unsqueeze(-1)*(f001.unsqueeze(-1)*_g(ab,i1,0,nsa)+f011.unsqueeze(-1)*_g(ab,i1,1,nsa)
                    +f101.unsqueeze(-1)*_g(ab,i1,nspb,nsa)+f111.unsqueeze(-1)*_g(ab,i1,nspb+1,nsa))
        +tauaer[...,4:5])
    T=torch.where((~lo).unsqueeze(-1),Tu,T)

    # Planck fraction upper: FRACREFB(IG, JPL) — species-interpolated (O3+CO2)
    # Fortran: ZREFRAT_PLANCK_B = CHI_MLS(3,43)/CHI_MLS(2,43)
    refrat_planck_b = chi[3, 43] / chi[2, 43]
    P_up = _planck_frac(fb, colo3, colco2, refrat_planck_b, oneminus, sm=4.0)
    P=torch.where((~lo).unsqueeze(-1), P_up, P)
    return T,P

def rrtm_taumol6(wx, tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1,
                 colh2o, colco2, coldry, laytrop,
                 selffac, selffrac, indself, minorfrac, indminor):
    ng=8; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(6,"absa",(65,ng)).to(dv); sr=_load_band(6,"selfref",(10,ng)).to(dv)
    fr=_load_band(6,"forref",(4,ng)).to(dv); fa=_load_band(6,"fracrefa",(ng,)).to(dv)
    km=_load_band(6,"ka_mco2",(19,ng)).to(dv)
    c11=_load_band(6,"cfc11adj",(ng,)).to(dv); c12=_load_band(6,"cfc12",(ng,)).to(dv)
    
    # Reference CO2 VMR from CHI_MLS (1-based padded: species 2 = CO2)
    from .setcoef import CHI_MLS
    chi_co2_ref = CHI_MLS[2, :].to(dv)  # (60,), 1-based: index 2 = CO2
    
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    # ── Lower atmosphere (≤ laytrop): H2O + foreign + self + CO2 continuum ──
    i0l=((jp-1)*5+(jt-1))+1; i1l=(jp*5+(jt1-1))+1
    a4=_a4(aa,i0l,i1l,fac00,fac10,fac01,fac11)
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    ac=_lin(km,indminor,minorfrac)
    
    # Fortran CO2 adjustment: CHI_MLS(2, K_JP) ≈ CHI_MLS(2, K_JP+1) for well-mixed CO2
    chi_co2 = colco2 / coldry.clamp(min=1e-30)              # CO2 mass fraction
    ratco2 = 1.0e20 * chi_co2 / chi_co2_ref[jp].unsqueeze(0).clamp(min=1e-30)
    adjfac = torch.where(ratco2 > 3.0, 2.0 + (ratco2 - 2.0)**0.77, torch.ones_like(ratco2))
    colco2_adj = torch.where(ratco2 > 3.0,
        adjfac * chi_co2_ref[jp].unsqueeze(0) * coldry * 1.0e-20,
        colco2)
    
    Tl = colh2o.unsqueeze(-1)*a4 + ts + tf \
         + colco2_adj.unsqueeze(-1) * ac \
         + wx[...,0:1]*c11.view(1,1,ng) + wx[...,1:2]*c12.view(1,1,ng) \
         + tauaer[...,5:6]
    T=torch.where(lo.unsqueeze(-1), Tl, T)
    
    # ── Upper atmosphere (> laytrop): nothing important (only CFC + aerosol) ──
    Tu = wx[...,0:1]*c11.view(1,1,ng) + wx[...,1:2]*c12.view(1,1,ng) + tauaer[...,5:6]
    T=torch.where((~lo).unsqueeze(-1), Tu, T)
    
    P=fa.view(1,1,ng).expand(n,nl,ng)
    return T,P

def rrtm_taumol7(tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                 colh2o, colo3, colco2, coldry, laytrop,
                 selffac, selffrac, indself,
                 rat_h2oo3, rat_h2oo3_1, minorfrac, indminor):
    """Band 7: 980-1080 cm-1 (low - H2O,O3 + CO2 minor; high - O3 + CO2 minor)."""
    ng=12; nspa=9; nspb_value=1; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(7,"absa",(585,ng)).to(dv); ab=_load_band(7,"absb",(235,ng)).to(dv)
    sr=_load_band(7,"selfref",(10,ng)).to(dv); fr=_load_band(7,"forref",(4,ng)).to(dv)
    fa=_load_band(7,"fracrefa",(ng,9)).to(dv); fb=_load_band(7,"fracrefb",(ng,)).to(dv)
    ka=_load_band(7,"ka_mco2",(9,19,ng)).to(dv); kb=_load_band(7,"kb_mco2",(19,ng)).to(dv)
    
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    refrat_planck = chi[1, 3] / chi[3, 3]   # CHI_MLS(1,3)/CHI_MLS(3,3)
    refrat_m = chi[1, 3] / chi[3, 3]          # CHI_MLS(1,3)/CHI_MLS(3,3)
    
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    # ── Lower: use proven _bl for H2O+O3 major gas, then add CO2 minor ──
    Tl_bl = _bl(7,ng,nspa,8.,colh2o,colo3,rat_h2oo3,rat_h2oo3_1,
                fac00,fac10,fac01,fac11,jp,jt,jt1,
                selffac,selffrac,indself,forfac,forfrac,indfor,
                sr,fr,aa,torch.zeros_like(tauaer[...,6:7]))
    
    # CO2 adjustment for lower: Fortran CHI_MLS(2, K_JP+1) → 1-based chi[2, jp+1]
    jp_co2 = (jp + 1).clamp(1, 59).long()
    chi_co2 = colco2 / coldry.clamp(min=1e-30)
    zratco2 = 1.0e20 * chi_co2 / chi[2, jp_co2]
    adjfac_lo = torch.where(zratco2 > 3.0, 3.0 + (zratco2 - 3.0)**0.79, torch.ones_like(zratco2))
    adjcolco2_lo = torch.where(zratco2 > 3.0,
        adjfac_lo * chi[2, jp_co2] * coldry * 1.0e-20, colco2)
    
    # CO2 minor gas via KA_MCO2 (species * temp * g-point)
    sc_mco2 = colh2o + refrat_m * colo3
    sp_mco2 = (colh2o / sc_mco2.clamp(min=1e-30)).clamp(max=0.999999)
    sm_mco2 = 8.0 * sp_mco2
    jmco2 = (1 + _trunc_int(sm_mco2)).clamp(1, 9).long()
    f_mco2 = sm_mco2 - (jmco2 - 1).to(dt)
    im_flat = (indminor.clamp(1, 19).long() - 1).reshape(n*nl)
    jm0_flat = (jmco2 - 1).clamp(0, 8).reshape(n*nl)
    jm1_flat = (jmco2).clamp(0, 8).reshape(n*nl)  # 0-based JMCO2+1
    im1_flat = (im_flat + 1).clamp(0, 18)
    zco2m1 = (ka[jm0_flat, im_flat, :] + f_mco2.reshape(n*nl, 1) * (ka[jm1_flat, im_flat, :] - ka[jm0_flat, im_flat, :])).reshape(n, nl, ng)
    zco2m2 = (ka[jm0_flat, im1_flat, :] + f_mco2.reshape(n*nl, 1) * (ka[jm1_flat, im1_flat, :] - ka[jm0_flat, im1_flat, :])).reshape(n, nl, ng)
    zabsco2 = zco2m1 + minorfrac.unsqueeze(-1) * (zco2m2 - zco2m1)
    
    Tl = Tl_bl + adjcolco2_lo.unsqueeze(-1)*zabsco2 + tauaer[...,6:7]
    T=torch.where(lo.unsqueeze(-1), Tl, T)
    
    # Planck fraction
    P_lo = _planck_frac(fa, colh2o, colo3, refrat_planck, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    
    # ── Upper atmosphere ──
    adjfac_up = torch.where(zratco2 > 3.0, 2.0 + (zratco2 - 2.0)**0.79, torch.ones_like(zratco2))
    adjcolco2_up = torch.where(zratco2 > 3.0,
        adjfac_up * chi[2, jp_co2] * coldry * 1.0e-20, colco2)
    
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u = _a4(ab, i0u, i1u, fac00, fac10, fac01, fac11)
    zabsco2_u = _lin(kb, indminor, minorfrac)  # KB_MCO2: simple 1D interpolation in T
    Tu = colo3.unsqueeze(-1)*a4u + adjcolco2_up.unsqueeze(-1)*zabsco2_u + tauaer[...,6:7]
    
    # Empirical scaling (Fortran lines 340-345)
    emps = torch.ones(ng, device=dv)
    emps[5]=0.92; emps[6]=0.88; emps[7]=1.07; emps[8]=1.1; emps[9]=0.99; emps[10]=0.855
    Tu = Tu * emps.view(1, 1, ng)
    
    T=torch.where((~lo).unsqueeze(-1), Tu, T)
    P=torch.where((~lo).unsqueeze(-1), fb.view(1,1,ng).expand(n,nl,ng), P)
    
    return T,P

def rrtm_taumol8(wx, tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1,
                 colh2o, colo3, coln2o, colco2, coldry, laytrop,
                 selffac, selffrac, indself, minorfrac, indminor):
    ng=8; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(8,"absa",(65,ng)).to(dv); ab=_load_band(8,"absb",(235,ng)).to(dv)
    sr=_load_band(8,"selfref",(10,ng)).to(dv); fr=_load_band(8,"forref",(4,ng)).to(dv)
    fa=_load_band(8,"fracrefa",(ng,)).to(dv); fb=_load_band(8,"fracrefb",(ng,)).to(dv)
    kmc=_load_band(8,"ka_mco2",(19,ng)).to(dv); kmn=_load_band(8,"ka_mn2o",(19,ng)).to(dv)
    kmo=_load_band(8,"ka_mo3",(19,ng)).to(dv)
    kbc=_load_band(8,"kb_mco2",(19,ng)).to(dv); kbn=_load_band(8,"kb_mn2o",(19,ng)).to(dv)
    c12=_load_band(8,"cfc12",(ng,)).to(dv); c22=_load_band(8,"cfc22adj",(ng,)).to(dv)
    from .tables import CHI_MLS; chi=CHI_MLS.to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    i0=((jp-1)*5+(jt-1))+1; i1=(jp*5+(jt1-1))+1
    a4=_a4(aa,i0,i1,fac00,fac10,fac01,fac11)
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    ac=_lin(kmc,indminor,minorfrac); ao=_lin(kmo,indminor,minorfrac)
    an=_lin(kmn,indminor,minorfrac)
    zc=colco2/coldry.clamp(min=1e-30); zr=1e20*zc/chi[2,jp.clamp(0,59).long()]
    af=2.+(zr-2.).clamp(min=0)**0.65; adj=af*chi[2,jp.clamp(0,59).long()]*coldry*1e-20
    adj=torch.where(zr>3.,adj,colco2)
    Tl=colh2o.unsqueeze(-1)*a4+ts+tf+adj.unsqueeze(-1)*ac+colo3.unsqueeze(-1)*ao+coln2o.unsqueeze(-1)*an \
       +wx[...,2:3]*c12.view(1,1,ng)+wx[...,3:4]*c22.view(1,1,ng)+tauaer[...,7:8]
    T=torch.where(lo.unsqueeze(-1),Tl,T); P=torch.where(lo.unsqueeze(-1),fa.view(1,1,ng),P)
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u=_a4(ab,i0u,i1u,fac00,fac10,fac01,fac11)
    acu=_lin(kbc,indminor,minorfrac); anu=_lin(kbn,indminor,minorfrac)
    Tu=colo3.unsqueeze(-1)*a4u+adj.unsqueeze(-1)*acu+coln2o.unsqueeze(-1)*anu \
       +wx[...,2:3]*c12.view(1,1,ng)+wx[...,3:4]*c22.view(1,1,ng)+tauaer[...,7:8]
    T=torch.where((~lo).unsqueeze(-1),Tu,T); P=torch.where((~lo).unsqueeze(-1),fb.view(1,1,ng).expand(n,nl,ng),P)
    return T,P

def rrtm_taumol9(tauaer, fac00, fac01, fac10, fac11,
                 forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                 colh2o, coln2o, colch4, coldry, laytrop, layswtch, laylow,
                 selffac, selffrac, indself,
                 rat_h2och4, rat_h2och4_1, minorfrac, indminor):
    """Band 9: 1180-1390 cm-1 (low-H2O,CH4+N2O; high-CH4+N2O).
    
    Direct Fortran port with N2O minor gas (KA_MN2O/KB_MN2O) and N2O adjustment.
    """
    ng=12; nspa=9; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(9,"absa",(585,ng)).to(dv); ab=_load_band(9,"absb",(235,ng)).to(dv)
    sr=_load_band(9,"selfref",(10,ng)).to(dv); fr=_load_band(9,"forref",(4,ng)).to(dv)
    fa=_load_band(9,"fracrefa",(ng,9)).to(dv); fb=_load_band(9,"fracrefb",(ng,)).to(dv)
    ka=_load_band(9,"ka_mn2o",(9,19,ng)).to(dv); kb=_load_band(9,"kb_mn2o",(19,ng)).to(dv)
    
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    refrat_planck = chi[1, 9] / chi[6, 9]   # CHI_MLS(1,9)/CHI_MLS(6,9)
    refrat_m = chi[1, 3] / chi[6, 3]          # CHI_MLS(1,3)/CHI_MLS(6,3)
    
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    # ── Lower atmosphere: H2O+CH4 major + N2O minor ──
    Tl_bl = _bl(9,ng,nspa,8.,colh2o,colch4,rat_h2och4,rat_h2och4_1,
                fac00,fac10,fac01,fac11,jp,jt,jt1,
                selffac,selffrac,indself,forfac,forfrac,indfor,
                sr,fr,aa,torch.zeros_like(tauaer[...,8:9]))
    
    # N2O adjustment: Fortran CHI_MLS(4, K_JP+1) → 1-based chi[4, jp+1]
    jp_n2o = (jp + 1).clamp(1, 59).long()
    chi_n2o = coln2o / coldry.clamp(min=1e-30)
    zratn2o = 1.0e20 * chi_n2o / chi[4, jp_n2o]
    adjfac = torch.where(zratn2o > 1.5, 0.5 + (zratn2o - 0.5)**0.65, torch.ones_like(zratn2o))
    adjcoln2o = torch.where(zratn2o > 1.5,
        adjfac * chi[4, jp_n2o] * coldry * 1.0e-20, coln2o)
    
    # N2O minor gas: KA_MN2O — (9, 19, 12), species+temp interpolation
    sc_mn2o = colh2o + refrat_m * colch4
    sp_mn2o = (colh2o / sc_mn2o.clamp(min=1e-30)).clamp(max=0.999999)
    sm_mn2o = 8.0 * sp_mn2o
    jmn2o = (1 + _trunc_int(sm_mn2o)).clamp(1, 9).long()
    f_mn2o = sm_mn2o - (jmn2o - 1).to(dt)
    im_flat = (indminor.clamp(1, 19).long() - 1).reshape(n*nl)
    jm0_flat = (jmn2o - 1).clamp(0, 8).reshape(n*nl)
    jm1_flat = (jmn2o).clamp(0, 8).reshape(n*nl)  # 0-based JMN2O+1
    im1_flat = (im_flat + 1).clamp(0, 18)
    zn2om1 = (ka[jm0_flat, im_flat, :] + f_mn2o.reshape(n*nl, 1) * (ka[jm1_flat, im_flat, :] - ka[jm0_flat, im_flat, :])).reshape(n, nl, ng)
    zn2om2 = (ka[jm0_flat, im1_flat, :] + f_mn2o.reshape(n*nl, 1) * (ka[jm1_flat, im1_flat, :] - ka[jm0_flat, im1_flat, :])).reshape(n, nl, ng)
    zabsn2o = zn2om1 + minorfrac.unsqueeze(-1) * (zn2om2 - zn2om1)
    
    Tl = Tl_bl + adjcoln2o.unsqueeze(-1) * zabsn2o + tauaer[...,8:9]
    T=torch.where(lo.unsqueeze(-1), Tl, T)
    
    # Planck fraction lower
    P_lo = _planck_frac(fa, colh2o, colch4, refrat_planck, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    
    # ── Upper atmosphere: CH4 * ABSB + N2O minor (KB_MN2O, simple temp interpolation) ──
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u = _a4(ab, i0u, i1u, fac00, fac10, fac01, fac11)
    
    # KB_MN2O: (19, 12) — simple 1D temperature interpolation, no species
    zabsn2o_u = _lin(kb, indminor, minorfrac)
    
    Tu = colch4.unsqueeze(-1) * a4u + adjcoln2o.unsqueeze(-1) * zabsn2o_u + tauaer[...,8:9]
    T=torch.where((~lo).unsqueeze(-1), Tu, T)
    P=torch.where((~lo).unsqueeze(-1), fb.view(1,1,ng).expand(n,nl,ng), P)
    
    return T,P

# ===== BANDS 10-16 (simpler patterns) =====
def rrtm_taumol10(tauaer, fac00, fac01, fac10, fac11,
                  forfac, forfrac, indfor, jp, jt, jt1,
                  colh2o, laytrop, selffac, selffrac, indself):
    ng=6; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(10,"absa",(65,ng)).to(dv); ab=_load_band(10,"absb",(235,ng)).to(dv)
    sr=_load_band(10,"selfref",(10,ng)).to(dv); fr=_load_band(10,"forref",(4,ng)).to(dv)
    fa=_load_band(10,"fracrefa",(ng,)).to(dv); fb=_load_band(10,"fracrefb",(ng,)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    i0=((jp-1)*5+(jt-1))+1; i1=(jp*5+(jt1-1))+1
    a4=_a4(aa,i0,i1,fac00,fac10,fac01,fac11)
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    Tl=colh2o.unsqueeze(-1)*a4+ts+tf+tauaer[...,9:10]
    T=torch.where(lo.unsqueeze(-1),Tl,T)
    # Upper atmosphere: uses ABSB, foreign continuum only
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u=_a4(ab,i0u,i1u,fac00,fac10,fac01,fac11)
    tfu=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    Tu=colh2o.unsqueeze(-1)*a4u+tfu+tauaer[...,9:10]
    T=torch.where((~lo).unsqueeze(-1),Tu,T)
    P=torch.where(lo.unsqueeze(-1),fa.view(1,1,ng).expand(n,nl,ng),
                                 fb.view(1,1,ng).expand(n,nl,ng))
    return T,P

def rrtm_taumol11(tauaer, fac00, fac01, fac10, fac11,
                  forfac, forfrac, indfor, jp, jt, jt1,
                  colh2o, colo2, laytrop,
                  selffac, selffrac, indself, minorfrac, indminor, scaleminor):
    ng=8; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(11,"absa",(65,ng)).to(dv); ab=_load_band(11,"absb",(235,ng)).to(dv)
    sr=_load_band(11,"selfref",(10,ng)).to(dv); fr=_load_band(11,"forref",(4,ng)).to(dv)
    fa=_load_band(11,"fracrefa",(ng,)).to(dv); fb=_load_band(11,"fracrefb",(ng,)).to(dv)
    km=_load_band(11,"ka_mo2",(19,ng)).to(dv); kbm=_load_band(11,"kb_mo2",(19,ng)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    i0=((jp-1)*5+(jt-1))+1; i1=(jp*5+(jt1-1))+1
    a4=_a4(aa,i0,i1,fac00,fac10,fac01,fac11)
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    sc=colo2*scaleminor; tn=sc.unsqueeze(-1)*_lin(km,indminor,minorfrac)
    Tl=colh2o.unsqueeze(-1)*a4+ts+tf+tn+tauaer[...,10:11]
    T=torch.where(lo.unsqueeze(-1),Tl,T); P=torch.where(lo.unsqueeze(-1),fa.view(1,1,ng),P)
    # upper
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u=_a4(ab,i0u,i1u,fac00,fac10,fac01,fac11)
    tfu=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    tnu=sc.unsqueeze(-1)*_lin(kbm,indminor,minorfrac)
    Tu=colh2o.unsqueeze(-1)*a4u+tfu+tnu+tauaer[...,10:11]
    T=torch.where((~lo).unsqueeze(-1),Tu,T); P=torch.where((~lo).unsqueeze(-1),fb.view(1,1,ng).expand(n,nl,ng),P)
    return T,P

def rrtm_taumol12(tauaer, fac00, fac01, fac10, fac11,
                  forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                  colh2o, colco2, laytrop, selffac, selffrac, indself,
                  rat_h2oco2, rat_h2oco2_1):
    ng=8; nspa=9; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(12,"absa",(585,ng)).to(dv); sr=_load_band(12,"selfref",(10,ng)).to(dv)
    fr=_load_band(12,"forref",(4,ng)).to(dv); fa=_load_band(12,"fracrefa",(ng,9)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    Tl=_bl(12,ng,nspa,8.,colh2o,colco2,rat_h2oco2,rat_h2oco2_1,
           fac00,fac10,fac01,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           sr,fr,aa,tauaer[...,11:12])  # separate_sc always True
    T=torch.where(lo.unsqueeze(-1),Tl,T)
    
    # Planck fraction: ZREFRAT_PLANCK_A = CHI_MLS(1,10)/CHI_MLS(2,10)
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    refrat_planck = chi[1, 10] / chi[2, 10]
    P_lo = _planck_frac(fa, colh2o, colco2, refrat_planck, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    # Upper: PFRAC=0 (Fortran) — already zero
    return T,P

def rrtm_taumol13(tauaer, fac00, fac01, fac10, fac11,
                  forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                  colh2o, coln2o, colco2, colo3, coldry, laytrop,
                  selffac, selffrac, indself,
                  rat_h2on2o, rat_h2on2o_1, minorfrac, indminor):
    """Band 13: 2080-2250 cm-1 (low-H2O,N2O+CO2+CO; high-O3 minor).
    
    Direct Fortran port with CO2 minor (KA_MCO2), CO minor (KA_MCO), and O3 upper (KB_MO3).
    """
    ng=4; nspa=9; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(13,"absa",(585,ng)).to(dv); sr=_load_band(13,"selfref",(10,ng)).to(dv)
    fr=_load_band(13,"forref",(4,ng)).to(dv); fa=_load_band(13,"fracrefa",(ng,9)).to(dv)
    fb=_load_band(13,"fracrefb",(ng,)).to(dv)
    kmc=_load_band(13,"ka_mco2",(9,19,ng)).to(dv); kmo=_load_band(13,"ka_mco",(9,19,ng)).to(dv)
    kbm=_load_band(13,"kb_mo3",(19,ng)).to(dv)
    
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    refrat_planck = chi[1, 5] / chi[4, 5]   # CHI_MLS(1,5)/CHI_MLS(4,5)
    refrat_m_a = chi[1, 1] / chi[4, 1]       # CHI_MLS(1,1)/CHI_MLS(4,1)
    refrat_m_a3 = chi[1, 3] / chi[4, 3]      # CHI_MLS(1,3)/CHI_MLS(4,3)
    
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    # ── Lower atmosphere: H2O+N2O major + CO2 minor + CO minor ──
    Tl_bl = _bl(13,ng,nspa,8.,colh2o,coln2o,rat_h2on2o,rat_h2on2o_1,
                fac00,fac10,fac01,fac11,jp,jt,jt1,
                selffac,selffrac,indself,forfac,forfrac,indfor,
                sr,fr,aa,torch.zeros_like(tauaer[...,12:13]))
    
    # CO2 adjustment (B13-specific: uses fixed reference 3.55e-4, not CHI_MLS)
    chi_co2 = colco2 / coldry.clamp(min=1e-30)
    zratco2 = 1.0e20 * chi_co2 / 3.55e-4
    adjfac_co2 = torch.where(zratco2 > 3.0, 2.0 + (zratco2 - 2.0)**0.68, torch.ones_like(zratco2))
    adjcolco2 = torch.where(zratco2 > 3.0,
        adjfac_co2 * 3.55e-4 * coldry * 1.0e-20, colco2)
    
    # Common indexing for minor gas tables
    im_flat = (indminor.clamp(1, 19).long() - 1).reshape(n*nl)
    im1_flat = (im_flat + 1).clamp(0, 18)
    
    # CO2 minor gas: KA_MCO2 (species+temp, 3D table)
    sc_mco2 = colh2o + refrat_m_a * coln2o
    sp_mco2 = (colh2o / sc_mco2.clamp(min=1e-30)).clamp(max=0.999999)
    sm_mco2 = 8.0 * sp_mco2
    jmco2 = (1 + _trunc_int(sm_mco2)).clamp(1, 9).long()
    f_mco2 = sm_mco2 - (jmco2 - 1).to(dt)
    jm0_c = (jmco2 - 1).clamp(0, 8).reshape(n*nl)
    jm1_c = (jmco2).clamp(0, 8).reshape(n*nl)  # 0-based JMCO2+1
    zco2m1 = (kmc[jm0_c, im_flat, :] + f_mco2.reshape(n*nl, 1) * (kmc[jm1_c, im_flat, :] - kmc[jm0_c, im_flat, :])).reshape(n, nl, ng)
    zco2m2 = (kmc[jm0_c, im1_flat, :] + f_mco2.reshape(n*nl, 1) * (kmc[jm1_c, im1_flat, :] - kmc[jm0_c, im1_flat, :])).reshape(n, nl, ng)
    zabsco2 = zco2m1 + minorfrac.unsqueeze(-1) * (zco2m2 - zco2m1)
    
    # CO minor gas: KA_MCO (species+temp, 3D table)
    sc_mco = colh2o + refrat_m_a3 * coln2o
    sp_mco = (colh2o / sc_mco.clamp(min=1e-30)).clamp(max=0.999999)
    sm_mco = 8.0 * sp_mco
    jmco = (1 + _trunc_int(sm_mco)).clamp(1, 9).long()
    f_mco = sm_mco - (jmco - 1).to(dt)
    jm0_co = (jmco - 1).clamp(0, 8).reshape(n*nl)
    jm1_co = (jmco).clamp(0, 8).reshape(n*nl)  # 0-based JMCO+1
    zcom1 = (kmo[jm0_co, im_flat, :] + f_mco.reshape(n*nl, 1) * (kmo[jm1_co, im_flat, :] - kmo[jm0_co, im_flat, :])).reshape(n, nl, ng)
    zcom2 = (kmo[jm0_co, im1_flat, :] + f_mco.reshape(n*nl, 1) * (kmo[jm1_co, im1_flat, :] - kmo[jm0_co, im1_flat, :])).reshape(n, nl, ng)
    zabsco = zcom1 + minorfrac.unsqueeze(-1) * (zcom2 - zcom1)
    
    # CO column is zero (not passed from GASABS) — Fortran sets Z_COLCO = 0
    zcolco = torch.zeros_like(colco2)
    
    Tl = Tl_bl + adjcolco2.unsqueeze(-1)*zabsco2 + zcolco.unsqueeze(-1)*zabsco + tauaer[...,12:13]
    T=torch.where(lo.unsqueeze(-1), Tl, T)
    
    # Planck fraction lower
    P_lo = _planck_frac(fa, colh2o, coln2o, refrat_planck, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    
    # ── Upper atmosphere: O3 * KB_MO3 (simple temp interpolation, no species) ──
    zabso3 = _lin(kbm, indminor, minorfrac)
    Tu = colo3.unsqueeze(-1) * zabso3 + tauaer[...,12:13]
    T=torch.where((~lo).unsqueeze(-1), Tu, T)
    P=torch.where((~lo).unsqueeze(-1), fb.view(1,1,ng).expand(n,nl,ng), P)
    
    return T,P

def rrtm_taumol14(tauaer, fac00, fac01, fac10, fac11,
                  forfac, forfrac, indfor, jp, jt, jt1,
                  colco2, laytrop, selffac, selffrac, indself):
    ng=2; n,nl=colco2.shape; dt=colco2.dtype; dv=colco2.device
    aa=_load_band(14,"absa",(65,ng)).to(dv); ab=_load_band(14,"absb",(235,ng)).to(dv)
    sr=_load_band(14,"selfref",(10,ng)).to(dv); fr=_load_band(14,"forref",(4,ng)).to(dv)
    fa=_load_band(14,"fracrefa",(ng,)).to(dv); fb=_load_band(14,"fracrefb",(ng,)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    i0=((jp-1)*5+(jt-1))+1; i1=(jp*5+(jt1-1))+1
    a4=_a4(aa,i0,i1,fac00,fac10,fac01,fac11)
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    Tl=colco2.unsqueeze(-1)*a4+ts+tf+tauaer[...,13:14]
    T=torch.where(lo.unsqueeze(-1),Tl,T); P=torch.where(lo.unsqueeze(-1),fa.view(1,1,ng),P)
    # upper: CO2 only
    i0u=((jp-13)*5+(jt-1))+1; i1u=((jp-12)*5+(jt1-1))+1
    a4u=_a4(ab,i0u,i1u,fac00,fac10,fac01,fac11)
    Tu=colco2.unsqueeze(-1)*a4u+tauaer[...,13:14]
    T=torch.where((~lo).unsqueeze(-1),Tu,T); P=torch.where((~lo).unsqueeze(-1),fb.view(1,1,ng).expand(n,nl,ng),P)
    return T,P

def rrtm_taumol15(tauaer, fac00, fac01, fac10, fac11,
                  forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                  colh2o, colco2, coln2o, laytrop, selffac, selffrac, indself,
                  rat_n2oco2, rat_n2oco2_1, minorfrac, indminor, scaleminor, colbrd):
    """Band 15: 2380-2600 cm-1 (low - N2O,CO2 + N2 minor; high - nothing).
    
    Direct Fortran port. Key differences from generic _bl:
    - N2 minor gas continuum (KA_MN2) with species+temperature interpolation
    - Planck fraction interpolation using H2O reference ratio
    """
    ng=2; nspa=9; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(15,"absa",(585,ng)).to(dv); sr=_load_band(15,"selfref",(10,ng)).to(dv)
    fr=_load_band(15,"forref",(4,ng)).to(dv); fa=_load_band(15,"fracrefa",(ng,9)).to(dv)
    km=_load_band(15,"ka_mn2",(9,19,ng)).to(dv)  # (nspa, ntemp, ng)
    
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    # ZREFRAT_PLANCK_A = ZREFRAT_M_A = CHI_MLS(4,1)/CHI_MLS(2,1)
    refrat_planck = chi[4, 1] / chi[2, 1]
    
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    # ── Lower atmosphere (same as _bl but with N2 minor gas) ──
    # Species combination 1: N2O + rat_n2oco2 * CO2
    sc=coln2o+rat_n2oco2*colco2; sp=(coln2o/sc.clamp(min=1e-30)).clamp(max=0.999999)
    sm_=8.0*sp; js=1+_trunc_int(sm_); fs=sm_-(js-1).to(dt)
    # Species combination 2: N2O + rat_n2oco2_1 * CO2
    sc1=coln2o+rat_n2oco2_1*colco2; sp1=(coln2o/sc1.clamp(min=1e-30)).clamp(max=0.999999)
    sm1=8.0*sp1; js1=1+_trunc_int(sm1); fs1=sm1-(js1-1).to(dt)
    # ABSA indices
    i0=((jp-1)*5+(jt-1))*nspa+js-1; i1=(jp*5+(jt1-1))*nspa+js1-1
    # BSW weights
    w0=_bsw(sp,fs,fac00,fac10,fac00,fac10); w1=_bsw(sp1,fs1,fac01,fac11,fac01,fac11)
    # Major gas: ABSA interpolation
    b0=_ba(aa,i0,nspa,w0); b1=_ba(aa,i1,nspa,w1)
    # Self + foreign continuum
    ts=selffac.unsqueeze(-1)*_lin(sr,indself,selffrac)
    tf=forfac.unsqueeze(-1)*_lin(fr,indfor,forfrac)
    
    # N2 minor gas: KA_MN2(JMN2, INDM, IG) with species interpolation
    # Z_SPECCOMB_MN2 = coln2o + refrat_planck * colco2
    sc_mn2 = coln2o + refrat_planck * colco2
    sp_mn2 = (coln2o / sc_mn2.clamp(min=1e-30)).clamp(max=0.999999)
    sm_mn2 = 8.0 * sp_mn2
    jmn2 = (1 + _trunc_int(sm_mn2)).clamp(1, 9).long()  # 1-based, ∈ [1,9]
    f_mn2 = sm_mn2 - (jmn2 - 1).to(dt)  # MOD ∈ [0,1)
    
    # KA_MN2 is (9, 19, ng): species * temp * g-point
    # Fortran: ZN2M1 = KA_MN2(JMN2, INDM, IG) + Z_FMN2 * (KA_MN2(JMN2+1, INDM, IG) - KA_MN2(JMN2, INDM, IG))
    # Fortran: ZN2M2 = KA_MN2(JMN2, INDM+1, IG) + Z_FMN2 * (KA_MN2(JMN2+1, INDM+1, IG) - KA_MN2(JMN2, INDM+1, IG))
    # Use 2D gather: flatten (n*nl) then reshape back
    im0_flat = (indminor.clamp(1, 19).long() - 1).reshape(n*nl)  # 0-based temp, (n*nl,)
    jm0_flat = (jmn2 - 1).clamp(0, 8).reshape(n*nl)               # 0-based species
    jm1_flat = (jmn2).clamp(0, 8).reshape(n*nl)                   # 0-based JMN2+1
    im1_flat = (im0_flat + 1).clamp(0, 18)                         # next temp
    
    # Gather: km[jm, im, :] → (n*nl, ng)
    zn2m1_flat = km[jm0_flat, im0_flat, :] + f_mn2.reshape(n*nl, 1) * (km[jm1_flat, im0_flat, :] - km[jm0_flat, im0_flat, :])
    zn2m2_flat = km[jm0_flat, im1_flat, :] + f_mn2.reshape(n*nl, 1) * (km[jm1_flat, im1_flat, :] - km[jm0_flat, im1_flat, :])
    zn2m1 = zn2m1_flat.reshape(n, nl, ng)
    zn2m2 = zn2m2_flat.reshape(n, nl, ng)
    
    zscalen2 = colbrd * scaleminor
    tn2 = zscalen2.unsqueeze(-1) * (zn2m1 + minorfrac.unsqueeze(-1) * (zn2m2 - zn2m1))
    
    # Total lower tau
    Tl = sc.unsqueeze(-1)*b0 + sc1.unsqueeze(-1)*b1 + ts + tf + tn2 + tauaer[...,14:15]
    T=torch.where(lo.unsqueeze(-1), Tl, T)
    
    # Planck fraction
    P_lo = _planck_frac(fa, coln2o, colco2, refrat_planck, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    # Upper: PFRAC=0, tau=tauaer — already zero from init
    return T,P

def rrtm_taumol16(tauaer, fac00, fac01, fac10, fac11,
                  forfac, forfrac, indfor, jp, jt, jt1, oneminus,
                  colh2o, colch4, laytrop, selffac, selffrac, indself,
                  rat_h2och4, rat_h2och4_1):
    ng=2; nspa=9; nspb=0; n,nl=colh2o.shape; dt=colh2o.dtype; dv=colh2o.device
    aa=_load_band(16,"absa",(585,ng)).to(dv); ab=_load_band(16,"absb",(235,ng)).to(dv)
    sr=_load_band(16,"selfref",(10,ng)).to(dv); fr=_load_band(16,"forref",(4,ng)).to(dv)
    fa=_load_band(16,"fracrefa",(ng,9)).to(dv); fb=_load_band(16,"fracrefb",(ng,)).to(dv)
    T=torch.zeros(n,nl,ng,dtype=dt,device=dv); P=torch.zeros_like(T)
    lo=torch.arange(n,device=dv).unsqueeze(1)<laytrop.unsqueeze(0)
    
    from .setcoef import CHI_MLS
    chi = CHI_MLS.to(dv)
    # ZREFRAT_PLANCK_A = CHI_MLS(1,6)/CHI_MLS(6,6)
    refrat_planck = chi[1, 6] / chi[6, 6]
    
    Tl=_bl(16,ng,nspa,8.,colh2o,colch4,rat_h2och4,rat_h2och4_1,
           fac00,fac10,fac01,fac11,jp,jt,jt1,
           selffac,selffrac,indself,forfac,forfrac,indfor,
           sr,fr,aa,tauaer[...,15:16])  # separate_sc always True
    T=torch.where(lo.unsqueeze(-1),Tl,T)
    
    # Planck fraction: FRACREFA(IG, JPL)
    P_lo = _planck_frac(fa, colh2o, colch4, refrat_planck, oneminus, sm=8.0)
    P=torch.where(lo.unsqueeze(-1), P_lo, P)
    
    # Upper: CH4 * ABSB + tauaer.  NSPB=0 → IND0=IND1=1 (fixed).
    a4u = fac00.unsqueeze(-1)*ab[0,:] + fac10.unsqueeze(-1)*ab[1,:] \
        + fac01.unsqueeze(-1)*ab[0,:] + fac11.unsqueeze(-1)*ab[1,:]
    Tu=colch4.unsqueeze(-1)*a4u+tauaer[...,15:16]
    T=torch.where((~lo).unsqueeze(-1),Tu,T)
    P=torch.where((~lo).unsqueeze(-1),fb.view(1,1,ng).expand(n,nl,ng),P)
    return T,P

# ===== GASABS1A DRIVER =====
def rrtm_gasabs1a_140gp(
    pavel, coldry, colbrd, wx, tauaer,
    fac00, fac01, fac10, fac11,
    forfac, forfrac, indfor, jp, jt, jt1, oneminus,
    colh2o, colco2, colo3, coln2o, colch4, colo2, co2mult,
    laytrop, layswtch, laylow,
    selffac, selffrac, indself,
    minorfrac, indminor, scaleminor, scaleminorn2,
    rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1,
    rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1,
    rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1,
):
    J=140; n,nl=pavel.shape; dt=pavel.dtype; dv=pavel.device
    T=torch.zeros(n,nl,J,dtype=dt,device=dv); P=torch.zeros_like(T)
    off=[0,10,22,38,52,68,76,88,96,108,114,122,130,134,136,138,140]
    def put(b,t,p):
        T[...,off[b-1]:off[b]]=t
        if p is not None: P[...,off[b-1]:off[b]]=p

    t1,p1=rrtm_taumol1(pavel,tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,
                        colh2o,laytrop,selffac,selffrac,indself,minorfrac,indminor,scaleminorn2,colbrd); put(1,t1,p1)
    t2,p2=rrtm_taumol2(pavel,coldry,tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,
                        colh2o,laytrop,selffac,selffrac,indself); put(2,t2,p2)
    t3,p3=rrtm_taumol3(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                        colh2o,colco2,coln2o,coldry,laytrop,selffac,selffrac,indself,
                        rat_h2oco2,rat_h2oco2_1,minorfrac,indminor); put(3,t3,p3)
    t4,p4=rrtm_taumol4(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                        colh2o,colco2,colo3,laytrop,selffac,selffrac,indself,
                        rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1); put(4,t4,p4)
    t5,p5=rrtm_taumol5(wx,tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                        colh2o,colco2,colo3,laytrop,selffac,selffrac,indself,
                        rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,minorfrac,indminor); put(5,t5,p5)
    t6,p6=rrtm_taumol6(wx,tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,
                        colh2o,colco2,coldry,laytrop,selffac,selffrac,indself,minorfrac,indminor); put(6,t6,p6)
    t7,p7=rrtm_taumol7(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                        colh2o,colo3,colco2,coldry,laytrop,selffac,selffrac,indself,
                        rat_h2oo3,rat_h2oo3_1,minorfrac,indminor); put(7,t7,p7)
    t8,p8=rrtm_taumol8(wx,tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,
                        colh2o,colo3,coln2o,colco2,coldry,laytrop,selffac,selffrac,indself,
                        minorfrac,indminor); put(8,t8,p8)
    t9,p9=rrtm_taumol9(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                        colh2o,coln2o,colch4,coldry,laytrop,layswtch,laylow,selffac,selffrac,indself,
                        rat_h2och4,rat_h2och4_1,minorfrac,indminor); put(9,t9,p9)
    t10,p10=rrtm_taumol10(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,
                           colh2o,laytrop,selffac,selffrac,indself); put(10,t10,p10)
    t11,p11=rrtm_taumol11(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,
                           colh2o,colo2,laytrop,selffac,selffrac,indself,minorfrac,indminor,scaleminor); put(11,t11,p11)
    t12,p12=rrtm_taumol12(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                           colh2o,colco2,laytrop,selffac,selffrac,indself,
                           rat_h2oco2,rat_h2oco2_1); put(12,t12,p12)
    t13,p13=rrtm_taumol13(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                           colh2o,coln2o,colco2,colo3,coldry,laytrop,selffac,selffrac,indself,
                           rat_h2on2o,rat_h2on2o_1,minorfrac,indminor); put(13,t13,p13)
    t14,p14=rrtm_taumol14(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,
                           colco2,laytrop,selffac,selffrac,indself); put(14,t14,p14)
    t15,p15=rrtm_taumol15(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                           colh2o,colco2,coln2o,laytrop,selffac,selffrac,indself,
                           rat_n2oco2,rat_n2oco2_1,minorfrac,indminor,scaleminor,colbrd); put(15,t15,p15)
    t16,p16=rrtm_taumol16(tauaer,fac00,fac01,fac10,fac11,forfac,forfrac,indfor,jp,jt,jt1,oneminus,
                           colh2o,colch4,laytrop,selffac,selffrac,indself,
                           rat_h2och4,rat_h2och4_1); put(16,t16,p16)

    secang=1.66; trans_tbl=_load_common("trans",(5001,)).to(dv)
    bpade=float(_load_common("bpade",()).item())
    od=secang*T.clamp(-1e10,1e10); ztf=(od/(bpade+od)).clamp(0,1)
    itr=_trunc_int(5000.*ztf+.5).clamp(0,5000)
    patr1=1.-trans_tbl[itr]
    return patr1, od, ztf, P

__all__ = [f"rrtm_taumol{i}" for i in range(1,17)] + ["rrtm_gasabs1a_140gp"]
