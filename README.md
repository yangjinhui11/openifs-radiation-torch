# 可微分 OpenIFS 辐射参数化（Differentiable OpenIFS Radiation）

ECMWF OpenIFS 48r1 短波（6 波段）与长波（RRTM-G）辐射参数化的 PyTorch 逐位精确移植，支持自动微分，可用于梯度敏感性分析与参数校准。

## 目录结构

```
radi_paper/
├── AGU_JAMES_AGUTeX_Article/     # 论文 LaTeX 源码与编译产物
│   ├── agujournaltemplate.tex     # 主文稿
│   ├── agujournal2019.cls         # AGU 文档类
│   └── agusample.bib             # 参考文献
├── radiation_solver/              # 辐射参数化 PyTorch 代码
│   └── openifs_radiation/
│       ├── classic_sw/            # 短波 6 波段方案（SWU/SWCLR/SWR/SWDE/SWTT/SW1S/SWNI）
│       └── rrtm_lw/               # 长波 RRTM-G 方案
├── experiments/                   # 实验脚本与数据
│   ├── e01_bitexact_verification/ # 逐内核 bit-exact 验证
│   ├── e02_era5_fluxes/           # ERA5 驱动的通量与加热率
│   ├── e03_gradient_sensitivity/  # autograd 梯度敏感性 + SW+LW 联合反馈核
│   ├── e04_parameter_calibration/ # 梯度参数校准
│   ├── fig4_global_gradients/     # 全球 SW+LW 敏感性场 + LW 验证
│   ├── calibration/               # 校准实验数据与绘图
│   └── perf_benchmark/            # CPU/GPU 性能基准
├── paper_figure/                  # 论文用图（PDF + PNG）
├── presentation/                  # 本科生讲义（HTML）
└── data/                          # 大数据文件说明
```

## 核心特性

- **逐位精确**：SW 12 个内核 + LW RRTM-G 全链相对 Fortran 参考达成 ULP 级一致（max |Δ| < 10⁻¹⁴）
- **自动可微**：SW + LW 完整辐射链可经 PyTorch autograd 反向传播，FD 交叉检验 rel_err < 10⁻¹⁰
- **GPU 加速**：同一代码在 NVIDIA A100 上运行，约 8000 列 × 137 层 SW+LW 计算在 2 秒内完成

## 环境依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0（建议 CUDA 支持）
- NumPy, Matplotlib, netCDF4
- gfortran（编译 Fortran 参考库）
- LaTeX（编译论文；推荐 MiKTeX，AGU `agujournal2019.cls`）

## 数据文件说明

实验使用的 ERA5 再分析数据来自 Copernicus Climate Data Store（公开），通过 CDO 工具从谱空间 GRIB 转换为 0.5° 网格点数据。因体积过大（5.5 GB）未纳入仓库。各实验目录内的 `*.json` / `*.npz` 为已产出的最终结果，可直接用于绘图复现。从原始 ERA5 GRIB 重建完整数据的说明见 `e02_era5_fluxes/gen_data.py`。

## 许可

- OpenIFS 源码遵循 ECMWF OpenIFS 许可证
- PyTorch 移植代码与实验脚本按 MIT 许可发布
- ERA5 数据来自 Copernicus Climate Data Store（Hersbach et al. 2020）

## 联系

Yang Jinhui — National University of Defense Technology, Changsha, China
通讯作者：Zhao Juan — zhaojuan@nudt.edu.cn
