"""T95 单步物理过程 Fortran vs torch 对比验证。

从 T95 callpar dump 的 PRE 状态出发，运行 torch RADHEATN 和
（如果可用）其他 torch 物理方案，与 Fortran POST 输出对比。

Dump 格式: big-endian int32 header + big-endian float32 body (SP build)。
精确字节偏移表来自 ec_phys.F90 CALLPAR_DUMP 的 WRITE 顺序验证。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent


# ── 精确偏移表 (klon=17, klev=137) ──────────────────────────────────────────
# 每个条目: (name, byte_offset, count, shape)
# count = number of float32 values; shape for reshape(order='F')

def _build_offsets(klon=17, klev=137):
    """Build the exact byte-offset table from the CALLPAR_DUMP write order."""
    k, kv, k4, kh = klon, klon*klev, klon*klev*4, klon*(klev+1)
    o = 20  # after header
    fields = []

    def add(name, n, shape=None):
        nonlocal o
        fields.append((name, o, n, shape))
        o += n * 4

    # PRE geom (5×klon)
    for n in ["pgelam","pgelat","pgemu","porog","pmu0"]:
        add(n, k, (klon,))
    # PRE full (5×klon×klev)
    for n in ["paprsf","paphif","pgeom1","prsf1","pdelp"]:
        add(n, kv, (klon,klev))
    # PRE half (4×klon×(klev+1))
    for n in ["paprs","paphi","pgeomh","prs1"]:
        add(n, kh, (klon,klev+1))
    # PRE state (7×klon×klev + cld klon×klev×4)
    for n in ["u","v","t","o3","q","a","tke"]:
        add(f"state_{n}", kv, (klon,klev))
    add("state_cld", k4, (klon,klev,4))
    # PRE gfl (5×klon×klev)
    for n in ["q","l","i","a","o3"]:
        add(f"gfl_{n}", kv, (klon,klev))
    # PRE tcml (5×klon×klev + cld)
    for n in ["u","v","t","q","a"]:
        add(f"tcml_{n}", kv, (klon,klev))
    add("tcml_cld", k4, (klon,klev,4))
    # PRE tdyn (4×klon×klev)
    for n in ["u","v","t","q"]:
        add(f"tdyn_{n}", kv, (klon,klev))
    # PRE rad (2×klon×klev)
    add("phrsw", kv, (klon,klev))
    add("phrlw", kv, (klon,klev))
    # POST tloc (5+cld)
    for n in ["u","v","t","q","a"]:
        add(f"tloc_{n}", kv, (klon,klev))
    add("tloc_cld", k4, (klon,klev,4))
    # POST tcml2 (5+cld)
    for n in ["u","v","t","q","a"]:
        add(f"tcml2_{n}", kv, (klon,klev))
    add("tcml2_cld", k4, (klon,klev,4))
    # POST diag (2)
    add("pcovptot", kv, (klon,klev))
    add("pqsat", kv, (klon,klev))
    # POST flux (2×klon×2)
    add("pfrthc", klon*2, (klon,2))
    add("pfrsoc", klon*2, (klon,2))
    # POST rad_extra v3: pedro,pts (klon×2) + 5×(klon×(klev+1)) + 4×klon
    add("pedro", k, (klon,))
    add("pts", k, (klon,))
    for n in ["pemtd","ptrsw","pemtc","ptrsc","derivlw"]:
        add(n, kh, (klon,klev+1))
    for n in ["psrswd","psrlwdc","pfdir","pcdir"]:
        add(n, k, (klon,))

    return fields, o


def read_dump(path):
    """Read the v3 T95 callpar dump using exact byte offsets."""
    raw = open(path, "rb").read()
    magic, klon, klev, kstglo, nstep = np.frombuffer(raw[:20], dtype=">i4")
    fields, expected_size = _build_offsets(int(klon), int(klev))
    assert expected_size == len(raw), f"size mismatch: {expected_size} vs {len(raw)}"

    data = {}
    for name, offset, count, shape in fields:
        d = np.frombuffer(raw[offset:offset+count*4], dtype=">f4").astype(np.float64)
        if shape:
            d = d.reshape(shape, order='F')
        data[name] = d
    return int(klon), int(klev), data


def bu2td(arr):
    """Bottom-up (klon, klev) → top-down (klev, klon) for torch.
    IFS bottom-up: index 0 = surface, klev-1 = TOA.
    torch top-down: index 0 = TOA, klev-1 = surface.
    Transpose to (klev, klon) and flip vertical axis.
    """
    return torch.from_numpy(arr.T[::-1].copy()).double()


def sh_half(arr):
    """(klon, 0:klev) half-level array → top-down (klev+1, klon) for torch.
    Dump: index 0=above-TOA(unused), 1=TOA, ..., klev=surface.
    radheatn needs: index 0=TOA(small P), ..., klev=surface(large P), increasing.
    Map: skip index 0, transpose+flip to top-down.
    Actually radheatn takes (nlon, nlev+1) with increasing pressure.
    Dump[1:]=TOA→surface (increasing). So just skip index 0 and keep order.
    """
    return torch.from_numpy(arr[:, 1:].T.copy()).double()  # (klev+1-1=klev, klon)... need klev+1


def stats(name, torch_val, fort_val):
    """Print comparison stats for 1D arrays."""
    t = torch_val.detach().cpu().numpy().ravel() if hasattr(torch_val, 'detach') else np.asarray(torch_val).ravel()
    f = np.asarray(fort_val).ravel()
    d = np.abs(t - f)
    if np.std(t) > 1e-30 and np.std(f) > 1e-30:
        corr = np.corrcoef(t, f)[0, 1]
    else:
        corr = float('nan')
    print(f"  {name:30s}: mean|Δ|={d.mean():.4e}  max|Δ|={d.max():.4e}  "
          f"corr={corr:.4f}  fort=[{f.min():.3e},{f.max():.3e}]  "
          f"torch=[{t.min():.3e},{t.max():.3e}]")
    return d.mean(), d.max(), corr


def main():
    DUMP = "/data/yangjinhui/openifs/run_oifs/T95/work_physics_dump/callpar_state_step1.bin"
    if len(sys.argv) > 1:
        DUMP = sys.argv[1]

    print(f"Loading: {DUMP}")
    klon, klev, p = read_dump(DUMP)
    print(f"  klon={klon}, klev={klev}")
    print(f"  PTS (skin T): {p['pts'][:5]}")
    print(f"  PMU0: {p['pmu0'][:5]} (cos sza, 0=night)")

    # ══════════════════════════════════════════════════════════════════════
    # 1. RADHEATN: 每步辐射通量插值
    # ══════════════════════════════════════════════════════════════════════
    sys.path.insert(0, str(_HERE.parent / "openifs_radiation_pytorch"))
    from openifs_radiation.radheatn import radheatn, RadHeatState

    dt = torch.float64
    nlon = klon

    # Build RadHeatState from dump's stored radiation fields.
    # Dump half-level arrays are (klon, 0:klev), index 0=unused, 1=TOA, klev=surface.
    # radheatn expects (nlon, klev+1) with index 0=TOA, klev=surface (increasing pressure).
    # Map: torch[i] = dump[i+1] for i=0..klev-1, torch[klev] = dump[klev] (duplicate surface).
    def sh(arr):
        """(klon, klev+1) dump → (nlon, klev+1) torch, skip index 0, pad surface."""
        out = np.zeros((nlon, klev+1), dtype=np.float64)
        out[:, :-1] = arr[:, 1:]   # dump[1:] → torch[0:klev] = TOA→surface
        out[:, -1] = arr[:, -1]    # duplicate surface
        return torch.from_numpy(out).to(dt)

    # paprs for radheatn: (nlon, klev+1), index 0=TOA, klev=surface.
    # Dump paprs: index 0=0(unused), 1=TOA(small), klev=surface(large).
    # Use dump directly: dp[jk] = paprs[jk+1]-paprs[jk] > 0 for all jk.
    paphm1 = torch.from_numpy(p["paprs"].copy()).to(dt)

    rad_state = RadHeatState(
        ptrsol=sh(p["ptrsw"]),
        ptrsoc=sh(p["ptrsc"]),
        ptrsod=torch.from_numpy(p["psrswd"]).to(dt),
        ptrsodc=torch.from_numpy(p["psrswd"]).to(dt) * 0.95,
        pfdiri=torch.from_numpy(p["pfdir"]).to(dt).clamp(min=1e-10),
        pcdiri=torch.from_numpy(p["pcdir"]).to(dt).clamp(min=1e-10),
        pemtd=sh(p["pemtd"]),
        pemtec=sh(p["pemtc"]),
        ptrthd=torch.from_numpy(p["psrlwdc"]).to(dt),
        ptrthdc=torch.from_numpy(p["psrlwdc"]).to(dt) * 0.95,
        plwderivative=sh(p["derivlw"]),
        pmu0m=torch.zeros(nlon, dtype=dt),  # last radiation call was at night
        puvdfi=torch.zeros(nlon, dtype=dt),
        pparfi=torch.zeros(nlon, dtype=dt),
        pparcfi=torch.zeros(nlon, dtype=dt),
        ptincfi=torch.zeros(nlon, dtype=dt),
        pemis=torch.full((nlon,), 0.98, dtype=dt),
    )

    # q: bottom-up (klon, klev) → top-down (klev, klon) → radheatn takes (nlon, klev)?
    # radheatn signature: pqm1: (nlon, nlev). But our dump q is bottom-up.
    # radheatn §4.2 loops jk=0..nlev-1 with pqm1[:,jk] and paphm1[:,jk:jk+2].
    # paphm1 is dump[0:klev+1] (bottom-up: 0=above-TOA, 1=TOA, ..., klev=surface).
    # So paphm1 increases with index (TOA→surface). pqm1 must match this order.
    # Our q (klon, klev) is bottom-up: index 0=surface. Need to reverse to match
    # paphm1's TOA-first order. But paphm1 index 0=0(unused), index 1=TOA.
    # pqm1 should be at full-level positions matching paphm1 half-levels.
    # radheatn: dp = paphm1[:,jk+1] - paphm1[:,jk]. jk=0: dp=paphm1[1]-paphm1[0].
    # paphm1[0]=0(unused), paphm1[1]=TOA pressure. dp[0]=TOA pressure ≈ 2 Pa.
    # pqm1[0] should be the q at the first model layer (near TOA).
    # Our q bottom-up: q[0]=surface. Reverse: q[::-1] gives TOA first.
    pqm1 = torch.from_numpy(p["state_q"][:, ::-1].copy()).to(dt)  # (nlon, klev) TOA-first
    pts = torch.from_numpy(p["pts"]).to(dt)
    pmu0 = torch.from_numpy(p["pmu0"]).to(dt)

    rad_out = radheatn(rad_state, paphm1, pqm1, pts, pmu0,
                       approxlw_update=True, manners_sw_update=False)

    print("\n" + "="*70)
    print("1. 辐射通量插值 (RADHEATN)")
    print("="*70)
    stats("LW TOA net flux", rad_out.pfrth[:, 0], p["pfrthc"][:, 0])
    stats("LW SFC net flux", rad_out.pfrth[:, -1], p["pfrthc"][:, 1])
    stats("SW TOA net flux", rad_out.pfrso[:, 0], p["pfrsoc"][:, 0])
    stats("SW SFC net flux", rad_out.pfrso[:, -1], p["pfrsoc"][:, 1])

    # Heating rates: Fortran phrlw is bottom-up; radheatn output is TOA-first
    phrlw_td = p["phrlw"][:, ::-1].copy()  # bu→td
    phrsw_td = p["phrsw"][:, ::-1].copy()
    stats("LW heating (col-mean)", rad_out.phrlw.mean(dim=0), phrlw_td.mean(axis=0))
    stats("SW heating (col-mean)", rad_out.phrsw.mean(dim=0), phrsw_td.mean(axis=0))

    # ══════════════════════════════════════════════════════════════════════
    # 2. Fortran callpar POST 倾向诊断
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("2. Fortran callpar 倾向诊断 (POST tendency_loc)")
    print("="*70)

    for var in ["u", "v", "t", "q", "a"]:
        tl = p[f"tloc_{var}"]
        print(f"  tloc_{var}: range=[{tl.min():.4e}, {tl.max():.4e}]"
              f"  mean={tl.mean():.4e}  nonzero={np.count_nonzero(tl)}/{tl.size}")

    # Callpar total tendency = POST tcml - PRE tcml
    print("\n  Callpar 总增量 (POST tcml - PRE tcml):")
    for var in ["u", "v", "t", "q"]:
        d = p[f"tcml2_{var}"] - p[f"tcml_{var}"]
        print(f"  d{var.upper()}/dt: range=[{d.min():.4e}, {d.max():.4e}]"
              f"  mean={d.mean():.4e}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. 辐射加热率剖面对比 (col 0, top-down)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("3. 加热率剖面对比 (col 0, top-down, every 20 levels)")
    print("="*70)
    print(f"  {'level':>5s}  {'fort_LW':>12s}  {'torch_LW':>12s}  {'fort_SW':>12s}  {'torch_SW':>12s}")
    for jk in range(0, klev, 20):
        f_lw = phrlw_td[0, jk]
        t_lw = rad_out.phrlw[0, jk].item()
        f_sw = phrsw_td[0, jk]
        t_sw = rad_out.phrsw[0, jk].item()
        print(f"  {jk:5d}  {f_lw:12.4e}  {t_lw:12.4e}  {f_sw:12.4e}  {t_sw:12.4e}")

    # ══════════════════════════════════════════════════════════════════════
    # 4. 降水和云物理诊断
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("4. 降水和云物理诊断")
    print("="*70)
    tloc_cld = p["tloc_cld"]  # (klon, klev, 4): ql, qi, qr, qs
    pdelp = p["pdelp"]  # (klon, klev) bottom-up
    rain_prod = -(tloc_cld[:,:,2] * pdelp).sum(axis=1) / 9.80665 * 86400  # mm/day
    snow_prod = -(tloc_cld[:,:,3] * pdelp).sum(axis=1) / 9.80665 * 86400
    print(f"  Rain production (mm/day): {rain_prod[:5]}")
    print(f"  Snow production (mm/day): {snow_prod[:5]}")
    print(f"  Cloud cover (pcovptot) max: {p['pcovptot'].max():.4f}")
    print(f"  Cloud water (tloc_cld liquid) max: {tloc_cld[:,:,0].max():.4e}")
    print(f"  Cloud ice   (tloc_cld ice)   max: {tloc_cld[:,:,1].max():.4e}")

    # ══════════════════════════════════════════════════════════════════════
    # 5. 状态量诊断
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("5. 大气状态诊断 (PRE)")
    print("="*70)
    t = p["state_t"]
    q = p["state_q"]
    print(f"  T: range=[{t.min():.2f}, {t.max():.2f}] K")
    print(f"  Q: range=[{q.min():.4e}, {q.max():.4e}] kg/kg")
    print(f"  U: range=[{p['state_u'].min():.2f}, {p['state_u'].max():.2f}] m/s")
    print(f"  Cloud fraction (a): range=[{p['state_a'].min():.4f}, {p['state_a'].max():.4f}]")
    print(f"  Surface pressure: {p['paprs'][:,-1][:3]} Pa")
    print(f"  TOA pressure: {p['paprs'][:,1][:3]} Pa")

    print("\n" + "="*70)
    print("验证完成")
    print("="*70)


if __name__ == "__main__":
    main()
