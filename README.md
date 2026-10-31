# SoilEvaporationLBM Native

基于 **C++17 + CUDA** 的三维孔隙尺度蒸发 LBM 程序，包含独立的**等温**与**非等温** Windows 原生 EXE。等温版本由原 Taichi 实现迁移而来；非等温版本在独立 CUDA 程序中增加温度场、热传输及潜热相关控制。运行不需要 Python/Taichi。非等温程序的明文源代码不在本仓库公开。

> 本仓库公开提供可执行程序及示例配置；非等温明文源码不公开。两个版本分别使用自己的 TXT 配置和 BAT 启动脚本。程序与文档适用仓库中的 **Three-Day Evaluation License（三天试用许可）**。

## 1. 等温版本主要功能

求解器目前包含：

- D3Q19 三维格子玻尔兹曼模型
- MRT 多松弛时间碰撞
- EDM forcing
- Peng–Robinson EOS
- Shan–Chen 赝势两相作用
- 基于 `G_ads` 的固液润湿作用
- X/Y 周期边界
- Z 方向非周期边界
- 顶部恒密度 Zou–He 干燥开放边界
- 稀疏孔隙节点存储
- 多样品批处理
- 多 `G_ads` 扫描
- 自动跳过已完成工况
- 配置变化自动归档旧结果
- checkpoint 与断点恢复
- `stats.csv`、`DONE.json`、`batch_summary.csv`
- VTK 输出
- 简洁/完整两种控制台输出模式

程序运行时不需要安装 Python、Taichi、Visual Studio 或 CUDA Toolkit，但需要正常安装 NVIDIA 显卡驱动。

## 2. 等温版本已完成的数值验证

Native CUDA 版本已使用以下基准工况与原 Taichi 版本进行 2000 步对照：

- 网格：`200 × 200 × 200`
- buffer：10
- 样品：`soil_sample_21`
- `G_ads=-0.20`
- `rho_dry=0.37`
- 真实孔隙节点：2,452,000
- 总流体节点：2,852,000

2000 步时主要结果：

| 指标 | Taichi | Native CUDA |
|---|---:|---:|
| Sat_eq | 0.960280 | 0.960287 |
| ERliq | 2.494483e-03 | 2.494235e-03 |
| Jsoil | 2.513832e-03 | 2.513567e-03 |
| Jtop | 2.576577e-03 | 2.576602e-03 |
| rho_min | 0.35655 | 0.35655 |
| rho_max | 7.88907 | 7.88912 |

主要状态量和通量的差异约为 `10^-4` 量级或更低，满足当前迁移验证要求。

> 性能比较需要在相同 VTK、checkpoint 和统计输出条件下进行。开发阶段观察到 Native CUDA 显著快于 Taichi，但不应直接把不同 I/O 配置下的耗时作为严格加速比。

## 3. GPU 支持

当前 Release 使用 CUDA 12.8 编译，并嵌入以下原生 CUDA 架构：

- `sm_75`：Turing
- `sm_80`：Ampere A100 等
- `sm_86`：Ampere RTX 30 系
- `sm_89`：Ada RTX 40 系
- `sm_90`：Hopper
- `sm_120`：Blackwell RTX 50 系

当前主要开发与验证平台为 RTX 4060 Laptop GPU。

仅支持 NVIDIA CUDA GPU。AMD Radeon、Intel Arc 或纯 CPU 环境不能运行当前 CUDA Release。

## 4. 发布包

标准发布目录：

```text
SoilEvaporationLBM.exe
config.txt
run.bat
```

双击：

```text
run.bat
```

即可按 `config.txt` 启动计算。

## 5. 等温配置文件（config.txt）

示例：

```ini
[PATHS]
geometry_root=F:\LBM\connected_pores
phase_file=F:\LBM\phase_200.txt
output_root=F:\LBM\result
geometry_pattern=**/*_connected_Z_THROUGH*.txt

[SAMPLES]
only=21
exclude=

[GRID]
nx=200
ny=200
nz_geo=200
n_buffer=10
pore_value=0

[PHYSICS]
niu=0.20
G_int=-1.0
beta_sc=1.16
Tr=0.86
rho_liq=6.498946
rho_gas=0.379679
rho_l_eq=6.498946
rho_g_eq=0.379679
rho_dry=0.37
G_ads=-0.20

[RUN]
total_steps=2000
record_interval=100
print_interval=100
vtk_interval=100
checkpoint_interval=1000
keep_checkpoints=2
save_vtk=1
device_memory_fraction=0.86
wait_seconds=3.0

[CONTROL]
console_mode=minimal
force=0
dry_run=0
pause_when_finished=1
```

### PATHS

`geometry_root`：孔隙结构文件所在目录。

`phase_file`：初始相分布文件。

`output_root`：结果输出根目录。

`geometry_pattern`：几何文件匹配规则。

路径支持中文、空格、绝对路径和相对路径。

### SAMPLES

`only`：只运行指定样品。数字 `21` 会识别为 `soil_sample_21`。

多个样品可使用逗号分隔，例如：

```ini
only=1,2,3,21
```

留空表示运行所有匹配样品。

`exclude`：排除指定样品。

### GRID

`nx`、`ny`、`nz_geo`：真实几何尺寸。

`n_buffer`：顶部气相 buffer 厚度。

`pore_value`：几何文件中代表孔隙的数值。当前数据通常使用：

```ini
pore_value=0
```

程序不固定为 200³，可通过配置切换到 300³ 等尺寸，但输入文件元素数量必须与 `nx × ny × nz_geo` 完全一致。

### PHYSICS

`niu`：运动黏度。

`G_int`：流体–流体相互作用强度。

`beta_sc`：改进 Shan–Chen 参数。

`Tr`：约化温度。

`rho_liq`、`rho_gas`：初始化液相/气相密度。

`rho_l_eq`、`rho_g_eq`：当前 EOS 与参数条件下用于等效液相体积计算的共存密度。

`rho_dry`：顶部干燥边界密度。

`G_ads`：固液润湿参数，可一次给多个值：

```ini
G_ads=-0.20,-0.15,-0.10,-0.05,0,0.05,0.10
```

程序会对每个样品 × 每个 `G_ads` 组合依次运行。

### RUN

`total_steps`：总迭代步数。

`record_interval`：写入统计结果的间隔。

`print_interval`：完整输出模式下打印迭代信息的间隔。

`vtk_interval`：VTK 输出间隔。

`checkpoint_interval`：checkpoint 保存间隔。

`keep_checkpoints`：每个工况保留最近几个 checkpoint，建议至少为 2。

`save_vtk`：

```ini
save_vtk=1
```

表示输出 VTK。

```ini
save_vtk=0
```

表示不输出 VTK。

### CONTROL

控制台输出：

```ini
console_mode=minimal
```

只显示每个工况成功或失败以及最终汇总，适合正式批量计算。

```ini
console_mode=full
```

显示 GPU、初始化、迭代、checkpoint、恢复等完整信息，适合调试和检查。

`force=1`：强制重算，即使已有完成结果。

`dry_run=1`：只列出将运行的工况，不启动 GPU。

`pause_when_finished=1`：运行结束后等待按键关闭窗口。

## 6. 输入数据格式

### Geometry

Geometry 为纯文本数值文件，共包含：

```text
nx × ny × nz_geo
```

个元素。

当前数据按照与原 Python/NumPy 实现一致的 Fortran-order 解释：

```text
flat_index = i + nx × (j + ny × k)
```

当 `pore_value=0` 时：

- 0 = pore
- 非 0 = solid

### Phase

Phase 文件同样包含：

```text
nx × ny × nz_geo
```

个元素。

判定规则：

```text
phase > 0.5  -> liquid
phase <= 0.5 -> gas
```

顶部 buffer 不从 phase 文件读取，而是使用 `rho_dry` 初始化。

## 7. 输出目录

典型输出结构：

```text
result/
├─ batch_summary.csv
└─ soil_sample_21/
   └─ Gads_m0p2/
      ├─ stats.csv
      ├─ DONE.json
      ├─ run_signature.txt
      ├─ *_checkpoint_001000.bin
      ├─ *_checkpoint_002000.bin
      └─ *.vtk
```

### stats.csv

记录主要质量、饱和度和通量指标，包括：

- mass_soil
- mass_buffer
- mass_domain
- saturation_threshold
- saturation_equiv
- liquid_volume_equiv
- liquid_mass_equiv
- J_soil_lu
- J_out_mass_lu
- J_top_direct_lu
- ER_liquid_equiv_lu
- mass balance diagnostics
- rho_min
- rho_max

### DONE.json

一个工况正常完成后生成，用于判断该工况是否已经完成。

### batch_summary.csv

汇总本批次所有工况的完成状态和最终关键结果。

## 8. Checkpoint 与断点恢复

Native 版本使用二进制 checkpoint：

```text
*_checkpoint_XXXXXX.bin
```

程序再次运行相同配置时，会优先恢复最近的有效 checkpoint。

为避免磁盘占用过大，仅保留最近 `keep_checkpoints` 个 checkpoint。

Native `.bin` checkpoint 与原 Taichi `.npz` checkpoint 不互相兼容。

## 9. 结果归档与重复运行

程序会根据输入文件和关键参数生成运行签名。

如果检测到同一工况已经完整计算并且配置一致，则自动跳过。

如果输入文件或关键参数发生变化，旧结果目录不会被覆盖，而是自动重命名为：

```text
Gads_m0p2__archive_config_YYYYMMDD-HHMMSS
```

然后创建新的结果目录重新计算。

## 10. 构建

源码构建依赖：

- Windows x64
- Visual Studio 2022 / MSVC
- CMake
- CUDA Toolkit 12.8

Release 使用静态 MSVC runtime 和静态 CUDA runtime。

GitHub Actions 会自动生成 Windows Release artifact。

正式分发时不建议包含：

```text
.cpp
.cu
.cuh
.pdb
.obj
.lib
.ptx
```

## 11. 注意事项

1. `rho_l_eq` 和 `rho_g_eq` 应与当前 EOS、`Tr`、`G_int`、`beta_sc` 等参数的标定结果对应。
2. 修改网格尺寸时必须同步更换 geometry 和 phase 文件。
3. 300³ 等大尺寸工况需要足够显存；程序采用稀疏流体节点布局以降低显存需求。
4. VTK 输出可能产生大量磁盘 I/O。正式长时间计算时应合理设置 `vtk_interval`。
5. 若用于严谨性能对比，应保证不同实现的 VTK、checkpoint、统计间隔完全一致。

## 12. 等温版本当前状态

当前 Native CUDA 迁移已完成核心数值验证，可用于后续正式批量计算和发布测试。

验证平台：

```text
NVIDIA GeForce RTX 4060 Laptop GPU
Compute Capability 8.9
Windows x64
```

核心求解结果已与原 Taichi 基准完成 2000 步对照。


## 13. 非等温版本（Non-Isothermal Native CUDA）

非等温计算使用 `SoilEvaporationLBM_NonIsothermal.exe`，独立读取 `config_nonisothermal.txt`。不要将等温配置文件直接替代非等温配置文件：非等温程序需要额外设置热学参数。可在 `release/` 目录双击 `run_nonisothermal.bat` 启动。

### 非等温配置结构

```ini
[PATHS]
geometry_root=F:\LBM\connected_pores
phase_file=F:\LBM\phase_200.txt
nonisothermal_output_root=F:\LBM\result_nonisothermal
output_root=F:\LBM\result_nonisothermal

[SAMPLES]
only=21
exclude=

[GRID]
nx=200
ny=200
nz_geo=200
n_buffer=10
pore_value=0

[PHYSICS]
niu=0.20
G_int=-1.0
beta_sc=1.16
rho_liq=6.498946
rho_gas=0.379679
rho_l_eq=6.498946
rho_g_eq=0.379679
rho_dry=0.37
G_ads=-0.20

[THERMAL]
rhocv_s=1.0
T_initial_ratio=0.86
T_bottom_ratio=0.86
T_surface_ratio=0.90
enable_heater=0
q_heat=0.0
enable_latent=1
latent_heat=1.0
latent_relaxation=1.0
interface_frac=0.5
velocity_ref=1.0
adv_correction=1
use_transport_weight=1
boundary_monotonic_guard=1
fo_safety=0.25
cfl_safety=0.25
min_substeps=1
max_substeps=64

[RUN]
total_steps=2000
record_interval=100
print_interval=100
vtk_interval=100
checkpoint_interval=1000
save_vtk=1
wait_seconds=3.0

[CONTROL]
console_mode=minimal
force=0
pause_when_finished=1
```

**重要**：上述 `[THERMAL]` 数值仅用于展示配置格式，**不是经数值标定的推荐物理参数**。正式计算前，必须核对原非等温模型采用的格子单位、热容、潜热、边界温度与热源输入，不能把摄氏温度直接填入 `T_*_ratio` 字段。等温配置中的 `Tr` 并不能替代非等温的温度场设置。

### 热学控制项

| 配置键 | 含义 |
|---|---|
| `rhocv_s` | 固相体积热容参数 |
| `T_initial_ratio` | 初始温度比 |
| `T_bottom_ratio` | 底部温度比 |
| `T_surface_ratio` | 表面温度比 |
| `enable_heater`, `q_heat` | 加热开关与热量输入 |
| `enable_latent`, `latent_heat`, `latent_relaxation` | 潜热项开关及其控制参数 |
| `interface_frac` | 界面识别相关阈值 |
| `velocity_ref`, `adv_correction`, `use_transport_weight` | 热平流及输运权重控制 |
| `boundary_monotonic_guard` | 边界温度限制控制 |
| `fo_safety`, `cfl_safety` | 热扩散与平流稳定性系数 |
| `min_substeps`, `max_substeps` | 热求解子步数范围 |

非等温程序通过 `[PATHS]` 指定与等温程序相同格式的几何和初始相分布输入，使用 `nonisothermal_output_root` 保存独立结果。统计结果包含饱和度、质量损失、温度极值和均值、能量与热通量诊断等字段，并可输出包含温度场的 VTK 文件。

非等温程序已在 GitHub Actions 上成功完成 Windows CUDA 编译，但**不能仅凭编译成功认定与原始非等温版本的数值结果完全一致**。用户应以经验证的物理参数及基准算例检查结果。

## 14. 三天试用许可与发布说明

等温和非等温 Windows 可执行程序均链接同一套试用期校验模块 `trial_guard.cpp`。更新后的版本从首次运行起按连续 **72 小时**限制使用，且在长时间计算期间持续检查许可状态。已存在的试用期记录不会因重新下载程序而重置。详细使用权限见仓库根目录 `LICENSE`。

GitHub Actions 对两个版本分别编译，并可将新生成的 EXE 同步到 `release/`。不要将旧的构建产物与新许可代码混用；应检查发布 EXE 对应的构建记录。非等温程序的加密构建输入仅用于受控编译，不应直接公开明文源文件。
