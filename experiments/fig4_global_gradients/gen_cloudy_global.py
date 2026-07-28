import os,sys,time,math
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from pathlib import Path
import numpy as np,torch
R=Path("/home/qixiang/yangjinhui/openifs/physics_callpar/openifs_radiation_pytorch")
sys.path.insert(0,str(R))
os.environ.setdefault("DATA",str(R))
import netCDF4 as nc
from openifs_radiation.classic_sw.driver import sw_solver
from openifs_radiation.rrtm_lw.ecrt import RadLSWInputs
from openifs_radiation.rrtm_lw.driver import rrtm_rrtm_140gp

t0=time.time()
ds=nc.Dataset("/tmp/era5_05deg_full.nc")
lat_v=ds.variables["lat"][:].astype(np.float64)
lon_v=ds.variables["lon"][:].astype(np.float64)
nlat=len(lat_v); nlon=len(lon_v); nlev=137; ncol=nlat*nlon
T=ds.variables["t"][0][:].astype(np.float64).reshape(nlev,ncol).T
Q=ds.variables["q"][0][:].astype(np.float64).reshape(nlev,ncol).T
O3=ds.variables["o3"][0][:].astype(np.float64).reshape(nlev,ncol).T
CC=ds.variables["cc"][0][:].astype(np.float64).reshape(nlev,ncol).T
CLWC=ds.variables["clwc"][0][:].astype(np.float64).reshape(nlev,ncol).T
CIWC=ds.variables["ciwc"][0][:].astype(np.float64).reshape(nlev,ncol).T
lnsp=ds.variables["lnsp"][0,0][:].astype(np.float64).reshape(ncol)
hyai=ds.variables["hyai"][:].astype(np.float64)
hybi=ds.variables["hybi"][:].astype(np.float64)
ds.close()
SP=np.exp(lnsp)
ph=np.zeros((ncol,nlev+1))
for k in range(nlev+1):
    ph[:,k]=hyai[k]+hybi[k]*SP
pf=0.5*(ph[:,:-1]+ph[:,1:])
dp=np.diff(ph)
delta=math.radians(23.45)*math.sin(2*math.pi*285/365)
lat_r=np.deg2rad(lat_v)
mu0_2d=np.zeros((nlat,nlon))
for j in range(nlon):
    omega=np.deg2rad(15*(lon_v[j]/15-12))
    mu0_2d[:,j]=np.clip(np.sin(lat_r)*math.sin(delta)+np.cos(lat_r)*math.cos(delta)*np.cos(omega),0,1)
mu0_f=np.maximum(mu0_2d.reshape(ncol),0.01)
print("Loaded ncol=%d cc[%.2f,%.2f] clwc_max=%.4e %.0fs"%(ncol,CC.min(),CC.max(),CLWC.max(),time.time()-t0),flush=True)

ls=12; los=8
li=np.arange(0,nlat,ls)
lj=np.arange(0,nlon,los)
sub=np.array([i*nlon+j for i in li for j in lj])
ns=len(sub)
print("Subsampled %d cols"%ns,flush=True)

dev="cuda"; dt=torch.float64; C2=348e-6*44.0095/28.9644
sw_clr=np.zeros(ns); sw_cld=np.zeros(ns)
lw_clr_olr=np.zeros(ns); lw_cld_olr=np.zeros(ns)
lw_clr_sfc=np.zeros(ns); lw_cld_sfc=np.zeros(ns)
CH=50

for ci in range(0,ns,CH):
    idx=sub[ci:ci+CH]; bs=len(idx)
    T_b=T[idx]; Q_b=Q[idx]; O3_b=O3[idx]
    CC_b=CC[idx]; CLWC_b=CLWC[idx]; CIWC_b=CIWC[idx]
    ph_b=ph[idx]; pf_b=pf[idx]; dp_b=dp[idx]; mu0_b=mu0_f[idx]
    Ts_b=T_b[:,-1]
    # Cloud water paths
    lwp=(CLWC_b*dp_b/9.81).sum(axis=1)
    iwp=(CIWC_b*dp_b/9.81).sum(axis=1)

    # --- Clear-sky SW ---
    sw_c=sw_solver(
        p_half=torch.from_numpy(ph_b).to(dt).to(dev),
        temp=torch.from_numpy(T_b).to(dt).to(dev),
        q_h2o=torch.from_numpy(Q_b).to(dt).to(dev),
        q_co2_vmr=torch.full((bs,nlev),C2,dtype=dt,device=dev),
        mu0=torch.from_numpy(mu0_b).to(dt).to(dev),
        albd=torch.full((bs,6),0.15,dtype=dt,device=dev),
        albp=torch.full((bs,6),0.15,dtype=dt,device=dev),
        pcldsw=torch.full((bs,nlev),1e-6,dtype=dt,device=dev),
        aer=torch.full((bs,6,nlev),1e-3,dtype=dt,device=dev),
        poz=torch.from_numpy(O3_b).to(dt).to(dev),
        pcg=torch.full((bs,6,nlev),0.85,dtype=dt,device=dev),
        pomega=torch.full((bs,6,nlev),0.99,dtype=dt,device=dev),
        ptau=torch.full((bs,6,nlev),1e-3,dtype=dt,device=dev),
        pqs=torch.full((bs,nlev),1e-3,dtype=dt,device=dev))
    sw_clr[ci:ci+bs]=sw_c["fd_total"][:,-1].cpu().numpy()

    # --- Cloudy-sky SW ---
    tau_sw=(lwp*30+iwp*50).clip(min=0)
    sw_d=sw_solver(
        p_half=torch.from_numpy(ph_b).to(dt).to(dev),
        temp=torch.from_numpy(T_b).to(dt).to(dev),
        q_h2o=torch.from_numpy(Q_b).to(dt).to(dev),
        q_co2_vmr=torch.full((bs,nlev),C2,dtype=dt,device=dev),
        mu0=torch.from_numpy(mu0_b).to(dt).to(dev),
        albd=torch.full((bs,6),0.15,dtype=dt,device=dev),
        albp=torch.full((bs,6),0.15,dtype=dt,device=dev),
        pcldsw=torch.from_numpy(CC_b).to(dt).to(dev).clamp(min=1e-6),
        aer=torch.full((bs,6,nlev),1e-3,dtype=dt,device=dev),
        poz=torch.from_numpy(O3_b).to(dt).to(dev),
        pcg=torch.full((bs,6,nlev),0.85,dtype=dt,device=dev),
        pomega=torch.full((bs,6,nlev),0.99,dtype=dt,device=dev),
        ptau=torch.from_numpy(tau_sw).to(dt).to(dev).unsqueeze(1).unsqueeze(2).expand(bs,6,nlev).clamp(min=1e-3),
        pqs=torch.full((bs,nlev),1e-3,dtype=dt,device=dev))
    sw_cld[ci:ci+bs]=sw_d["fd_total"][:,-1].cpu().numpy()

    # --- LW: build inputs in (nlev, nlon) convention ---
    def mk_lw(tau_lw_arr):
        return RadLSWInputs(
            emis=torch.full((bs,),0.98,dtype=dt,device=dev),
            emiw=torch.full((bs,),0.95,dtype=dt,device=dev),
            albd=torch.full((bs,),0.15,dtype=dt,device=dev),
            albp=torch.full((bs,),0.15,dtype=dt,device=dev),
            pts=torch.from_numpy(Ts_b).to(dt).to(dev),
            cloud_fraction=torch.from_numpy(CC_b).to(dt).to(dev).t().contiguous(),
            pap=torch.from_numpy(pf_b).to(dt).to(dev).t().contiguous(),
            paph=torch.from_numpy(ph_b).to(dt).to(dev).t().contiguous(),
            pdp=torch.from_numpy(dp_b).to(dt).to(dev).t().contiguous(),
            pt=torch.from_numpy(T_b).to(dt).to(dev).t().contiguous(),
            pth=torch.from_numpy(np.column_stack([T_b[:,0]-10, 0.5*(T_b[:,:-1]+T_b[:,1:]), Ts_b]).T.astype(np.float64)).to(dt).to(dev),
            pq=torch.from_numpy(Q_b).to(dt).to(dev).t().contiguous(),
            co2=torch.full((nlev,bs),C2,dtype=dt,device=dev),
            ch4=torch.full((nlev,bs),1.65e-6*16.0425/28.9644,dtype=dt,device=dev),
            n2o=torch.full((nlev,bs),0.306e-6*44.0128/28.9644,dtype=dt,device=dev),
            no2=torch.zeros((nlev,bs),dtype=dt,device=dev),
            cfc11=torch.zeros((nlev,bs),dtype=dt,device=dev),
            cfc12=torch.zeros((nlev,bs),dtype=dt,device=dev),
            cfc22=torch.zeros((nlev,bs),dtype=dt,device=dev),
            ccl4=torch.zeros((nlev,bs),dtype=dt,device=dev),
            o3_mmr=torch.from_numpy(O3_b).to(dt).to(dev).t().contiguous(),
            aer=None,
            tau_lw=torch.from_numpy(tau_lw_arr).to(dt).to(dev),
            sw_optics=None)
    # Clear-sky LW
    lw_c=rrtm_rrtm_140gp(mk_lw(np.zeros((nlev,bs,16))),novlp=1)
    lw_clr_olr[ci:ci+bs]=lw_c["flux_up"][:,0].cpu().numpy()
    lw_clr_sfc[ci:ci+bs]=lw_c["flux_down"][:,-1].cpu().numpy()
    # Cloudy-sky LW
    tau_lw_cld=np.zeros((nlev,bs,16))
    for j in range(bs):
        cmask=CC_b[j]>0.1
        if cmask.sum()>0:
            od=(lwp[j]*15+iwp[j]*25)/max(cmask.sum(),1)
            for b in range(16):
                tau_lw_cld[cmask,j,b]=od*0.8+0.2
    lw_d=rrtm_rrtm_140gp(mk_lw(tau_lw_cld),novlp=1)
    lw_cld_olr[ci:ci+bs]=lw_d["flux_up"][:,0].cpu().numpy()
    lw_cld_sfc[ci:ci+bs]=lw_d["flux_down"][:,-1].cpu().numpy()

    if ci%500==0:
        el=time.time()-t0
        print("%d/%d %.0fs ETA %.0fs"%(ci,ns,el,el/max(ci,1)*(ns-ci)),flush=True)

nls=len(li); nll=len(lj)
def t2(a):
    f=np.full(ncol,np.nan); f[sub]=a; return f.reshape(nlat,nlon)[np.ix_(li,lj)]
np.savez("/tmp/cloudy_lw_global.npz",
    sw_clear=t2(sw_clr), sw_cloudy=t2(sw_cld),
    lw_clr_olr=t2(lw_clr_olr), lw_cld_olr=t2(lw_cld_olr),
    lw_clr_sfc=t2(lw_clr_sfc), lw_cld_sfc=t2(lw_cld_sfc),
    lats=lat_v[li], lons=lon_v[lj])
print("Saved %.0fs"%(time.time()-t0),flush=True)
print("SW clear[%.0f,%.0f] cloudy[%.0f,%.0f]"%(np.nanmin(sw_clr),np.nanmax(sw_clr),np.nanmin(sw_cld),np.nanmax(sw_cld)))
print("LW clr OLR[%.0f,%.0f] cld OLR[%.0f,%.0f]"%(np.nanmin(lw_clr_olr),np.nanmax(lw_clr_olr),np.nanmin(lw_cld_olr),np.nanmax(lw_cld_olr)))
print("LW clr sfc[%.0f,%.0f] cld sfc[%.0f,%.0f]"%(np.nanmin(lw_clr_sfc),np.nanmax(lw_clr_sfc),np.nanmin(lw_cld_sfc),np.nanmax(lw_cld_sfc)))
print("DONE",flush=True)
