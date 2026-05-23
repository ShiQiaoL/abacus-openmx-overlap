# ABACUS/OpenMX Direct Overlap

中文：从 `POSCAR` 和数值原子轨道 basis 直接生成 DeepH/DeepX 可用的 `overlap.h5`，并提供 spin、Fermi、`deeph-infer` 和 `dock compute eigen calc-band` 的防呆检查。

English: Generate DeepH/DeepX-compatible `overlap.h5` directly from a `POSCAR` and numerical atomic orbital basis files, with guard rails for spin, Fermi energy, `deeph-infer`, and `dock compute eigen calc-band`.

---

## 目录 / Table of Contents

- [这个项目解决什么问题 / What This Solves](#这个项目解决什么问题--what-this-solves)
- [一句话用法 / One-Line Workflows](#一句话用法--one-line-workflows)
- [安装 / Installation](#安装--installation)
- [输入文件 / Required Inputs](#输入文件--required-inputs)
- [命令总览 / Command Overview](#命令总览--command-overview)
- [ABACUS basis 用法 / ABACUS Basis Usage](#abacus-basis-用法--abacus-basis-usage)
- [OpenMX basis 用法 / OpenMX Basis Usage](#openmx-basis-用法--openmx-basis-usage)
- [Spin 选择指南 / Spin Decision Guide](#spin-选择指南--spin-decision-guide)
- [Fermi energy 指南 / Fermi Energy Guide](#fermi-energy-指南--fermi-energy-guide)
- [DeepH inference 前检查 / Pre-Inference Check](#deeph-inference-前检查--pre-inference-check)
- [Band 计算防呆 / Band Calculation Guard](#band-计算防呆--band-calculation-guard)
- [输出文件 / Output Files](#输出文件--output-files)
- [常见错误和解决办法 / Troubleshooting](#常见错误和解决办法--troubleshooting)
- [大体系建议 / Large-System Notes](#大体系建议--large-system-notes)
- [开发和验证 / Development and Validation](#开发和验证--development-and-validation)
- [许可证 / License](#许可证--license)

---

## 这个项目解决什么问题 / What This Solves

中文：在 DeepH/DeepX 工作流中，经常需要一个和结构、basis 一致的 `overlap.h5`。传统做法可能需要跑 ABACUS `get_S` 或 OpenMX 输出矩阵。对于大体系，这会带来几个问题：

- OpenMX/DFT SCF 可能非常贵，尤其是 SOC/noncollinear 大体系。
- 只为了 overlap 去跑完整 SCF 不划算。
- `overlap.h5`、`hamiltonian.h5`、`info.json` 的 spin 约定很容易混乱。
- `deeph-infer` 和 `dock compute eigen calc-band` 对 spinful overlap 的使用场景不完全一样。
- Fermi energy 经常被误以为可以从 overlap-only 计算得到。

English: DeepH/DeepX workflows often need an `overlap.h5` consistent with the structure and basis. Running ABACUS `get_S` or OpenMX matrix output can be expensive or confusing, especially for large SOC/noncollinear systems. This project provides a direct overlap path and explicit validation tools.

本项目做什么：

- 从 `POSCAR` + ABACUS `.orb` 直接生成 DeepH-format `overlap.h5`。
- 从 `POSCAR` + OpenMX `.pao` 直接生成 DeepH-format `overlap.h5`。
- 生成 `info.json`、`POSCAR` 和 `direct_overlap_manifest.json`。
- 检查 spinful/spinless block shape 是否和下游模型一致。
- 为 `dock compute eigen calc-band` 自动准备干净的 `band_ready/` 目录。
- 明确记录 Fermi energy 的来源，避免把 placeholder 当作物理值。

本项目不做什么：

- 不运行 ABACUS SCF。
- 不运行 OpenMX SCF。
- 不生成 Hamiltonian。
- 不生成 SOC Hamiltonian。
- 不从 overlap-only 数据推断真实 Fermi energy。
- 不把未收敛 OpenMX 输出当成可靠物理证据。

---

## 一句话用法 / One-Line Workflows

### 1. 只生成 overlap.h5

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au11.0-s3p2d2f1 \
  --openmx-basis Mo=Mo11.0-s3p2d2 \
  --openmx-basis S=S9.0-s2p2d1 \
  --output-dir OUT_DIR \
  --ecut 300 \
  --overwrite
```

### 2. 生成后检查 deeph-infer 是否会 spin mismatch

```bash
direct-overlap-check OUT_DIR --expect-spin spinless
```

或者：

```bash
direct-overlap-check OUT_DIR --expect-spin spinful
```

### 3. 为 SOC/spinful Hamiltonian 准备 calc-band 目录

```bash
direct-overlap-band-prep CASE_ROOT \
  --out-dir CASE_ROOT/band_ready \
  --overwrite \
  --yes

cd CASE_ROOT/band_ready
dock compute eigen calc-band ./ --parallel-num 5 --thread-num 1
```

### 4. 只诊断，不写文件

```bash
direct-overlap-band-prep CASE_ROOT --diagnose-only
```

---

## 安装 / Installation

### 推荐安装方式 / Recommended

中文：推荐在已经有 Python、MPI runtime、`numpy`、`scipy`、`h5py` 支持的 HPC 环境里安装。仓库提供一个简单安装脚本：

English: Install inside an HPC Python environment that already has a working MPI/scientific stack.

```bash
git clone https://github.com/ShiQiaoL/abacus-openmx-overlap.git
cd abacus-openmx-overlap
bash install.sh
source .venv/bin/activate
```

检查命令是否可用：

```bash
direct-overlap --help
direct-overlap-check --help
direct-overlap-band-prep --help
```

### 已有 Python 环境 / Existing Environment

如果你已经有自己的 `deeph-python` 或集群 Python 环境，也可以直接安装：

```bash
git clone https://github.com/ShiQiaoL/abacus-openmx-overlap.git
cd abacus-openmx-overlap
python -m pip install -e .
```

### 依赖 / Dependencies

`pyproject.toml` 中固定的主要依赖：

- `numpy`
- `scipy`
- `h5py`
- `deepx-dock==0.9.11`
- `hpro==0.3.0`

如果集群需要手动加载 MPI 或编译环境，请先运行对应命令，例如：

```bash
module load mpi
source /path/to/env.sh
source .venv/bin/activate
```

这里的 `/path/to/env.sh` 是占位符。不要把私有账号路径直接写进公开脚本。

---

## 输入文件 / Required Inputs

### 必需文件 / Required

`DATA_DIR` 必须包含：

```text
DATA_DIR/
  POSCAR
```

`POSCAR` 必须是 VASP 5 格式，也就是第 6 行有明确元素名：

```text
comment
1.0
a1x a1y a1z
a2x a2y a2z
a3x a3y a3z
Au S Mo
16 8 4
Direct
...
```

如果 `POSCAR` 没有元素行，程序会拒绝继续，因为元素顺序错了会直接导致 basis 和 HDF5 block shape 错配。

### Basis 文件 / Basis Files

ABACUS backend 需要 `.orb`：

```text
BASIS_DIR/
  Au_*.orb
  S_*.orb
  Mo_*.orb
```

OpenMX backend 需要 `.pao`：

```text
BASIS_DIR/
  Au*.pao
  S*.pao
  Mo*.pao
```

OpenMX 必须显式告诉程序每个元素使用哪个 basis label：

```bash
--openmx-basis Au=Au11.0-s3p2d2f1
--openmx-basis Mo=Mo11.0-s3p2d2
--openmx-basis S=S9.0-s2p2d1
```

重要：basis label 必须和下游模型训练时的 basis 完全一致。`S9.0-s2p2d1` 和 `S9.0-s3p2d1` 不是同一个 basis；后者会让 S 的 orbital count 从 13 变成 14，后续 band/inference 会失败。

---

## 命令总览 / Command Overview

### `direct-overlap`

用途：从结构和 basis 直接生成 DeepH-format overlap。

```bash
direct-overlap DATA_DIR BASIS_DIR [options]
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--basis-code abacus` | 使用 ABACUS `.orb` |
| `--basis-code openmx` | 使用 OpenMX `.pao` |
| `--openmx-basis ELEMENT=BASIS` | 指定 OpenMX basis label |
| `--basis-map ELEMENT=PATH` | 手动指定某个元素的 basis 文件 |
| `--output-dir OUT_DIR` | 输出目录 |
| `--ecut VALUE` | HPRO overlap 积分 cutoff |
| `--spinful` | 写 spinful overlap，用于特定 DeepH inference 场景 |
| `--expect-spin spinless/spinful` | 防止脚本里 spin 选项写错 |
| `--fermi-energy-ev VALUE` | 显式写入 Fermi energy |
| `--fermi-log PATH` | 从收敛 SCF log 解析 Fermi energy |
| `--require-fermi-energy` | 没有显式 Fermi 就失败 |
| `--expect-formula Au16S8Mo4` | 检查结构 formula |
| `--expect-natoms 28` | 检查 atom count |
| `--overwrite` | 允许覆盖输出文件 |
| `--dry-run` | 只检查和写 manifest，不计算 |

### `direct-overlap-check`

用途：检查一个 overlap 目录是否和下游 DeepH inference 的 spin 模式一致。

```bash
direct-overlap-check OUT_DIR --expect-spin spinful
```

### `direct-overlap-band-prep`

用途：为 `dock compute eigen calc-band` 生成一致的 `band_ready/` 目录，提前阻止 `52x52 -> 26x26` 这类错误。

```bash
direct-overlap-band-prep CASE_ROOT --out-dir band_ready --overwrite --yes
```

---

## ABACUS Basis 用法 / ABACUS Basis Usage

### 最小例子 / Minimal Example

```bash
direct-overlap DATA_DIR ABACUS_ORB_DIR \
  --basis-code abacus \
  --output-dir OUT_DIR \
  --ecut 100 \
  --overwrite
```

### 推荐加结构保护 / Recommended Guards

```bash
direct-overlap DATA_DIR ABACUS_ORB_DIR \
  --basis-code abacus \
  --output-dir OUT_DIR \
  --ecut 100 \
  --expect-formula Au16S8Mo4 \
  --expect-natoms 28 \
  --overwrite
```

### 手动指定 basis 文件 / Manual Basis Map

如果目录里有多个 `Mo_*.orb`，自动识别可能不安全。可以手动指定：

```bash
direct-overlap DATA_DIR ABACUS_ORB_DIR \
  --basis-code abacus \
  --basis-map Au=/path/to/Au_gga_8au_100Ry_4s2p2d1f.orb \
  --basis-map Mo=/path/to/Mo_gga_8au_100Ry_4s2p2d1f.orb \
  --basis-map S=/path/to/S_gga_8au_100Ry_2s2p1d.orb \
  --output-dir OUT_DIR \
  --ecut 100 \
  --overwrite
```

---

## OpenMX Basis 用法 / OpenMX Basis Usage

### 最小例子 / Minimal Example

```bash
direct-overlap DATA_DIR OPENMX_PAO_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au11.0-s3p2d2f1 \
  --openmx-basis Mo=Mo11.0-s3p2d2 \
  --openmx-basis S=S9.0-s2p2d1 \
  --output-dir OUT_DIR \
  --ecut 300 \
  --overwrite
```

### 为什么 OpenMX 必须写 `--openmx-basis`

OpenMX `.pao` 文件包含一组 radial functions，但具体使用多少个 `s/p/d/f` 轨道由 label 决定。例如：

```text
S9.0-s2p2d1  -> S: 2 s + 2 p + 1 d -> 2*1 + 2*3 + 1*5 = 13 orbitals
S9.0-s3p2d1  -> S: 3 s + 2 p + 1 d -> 3*1 + 2*3 + 1*5 = 14 orbitals
```

这一个 orbital 的差异会在后续造成 HDF5 block shape mismatch。

### OpenMX normalization warning

OpenMX 官方 `.pao` 的 radial functions 不一定逐个严格归一化。默认行为是警告并继续：

```bash
--allow-unnormalized-orbitals
```

如果你希望任何 normalization deviation 都直接失败：

```bash
--strict-norm-check
```

---

## Spin 选择指南 / Spin Decision Guide

### 先区分两个问题

1. 你的下游模型/数据格式需要什么 shape？
2. 你是在做 `deeph-infer`，还是在做 `dock compute eigen calc-band`？

这两个问题不能混在一起。

### 生成 overlap 给 `deeph-infer`

| 下游模型 | 推荐 overlap |
|---|---|
| spinless 模型 | 不加 `--spinful` |
| spinful/spinor 模型，且模型图期待 doubled overlap block | 加 `--spinful` |

例子：

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au11.0-s3p2d2f1 \
  --openmx-basis Mo=Mo11.0-s3p2d2 \
  --openmx-basis S=S9.0-s2p2d1 \
  --output-dir DFT_OUT \
  --spinful \
  --expect-spin spinful \
  --overwrite
```

### 生成 overlap 给 `dock compute eigen calc-band`

如果你已经有 spinful/SOC `hamiltonian.h5`，不要手动喂一个 doubled spinful `overlap.h5` 给 `calc-band`。正确组合是：

```text
info.json:       spinful = true
orbits_quantity: spatial orbital count, not doubled
hamiltonian.h5:  spinful/spinor block, doubled
overlap.h5:      spatial/spinless block, not doubled
```

直接用防呆命令：

```bash
direct-overlap-band-prep CASE_ROOT --out-dir CASE_ROOT/band_ready --overwrite --yes
```

### `--spinful` 到底做了什么

`--spinful` 只做：

```text
S_spin = I_2 ⊗ S_spatial
```

它不做：

- 不生成 SOC Hamiltonian。
- 不生成 spin-mixing overlap。
- 不改变 basis 的物理径向函数。
- 不让 Fermi energy 变得更真实。

---

## Fermi Energy 指南 / Fermi Energy Guide

### overlap.h5 不能决定 Fermi energy

`overlap.h5` 只包含 basis overlap matrix `S`。没有 Hamiltonian `H`，就不能决定电子本征值、占据和 chemical potential。

因此，direct-overlap 默认只能写：

```json
"fermi_energy_eV": 0.0
```

这只是 placeholder，不是物理费米能级。

### 可靠来源

可靠 Fermi 来源按优先级：

1. 同一体系或科学上等价体系的收敛 SCF。
2. 经过说明和校准的代表性小胞/界面模型。
3. DeepH/DeepX 推理得到 `hamiltonian.h5` 后，通过 `H` 和 `S` 解本征值并按电子数求 Fermi。

不可靠来源：

- `scf.maxIter 1` 的 OpenMX 输出。
- OOM/killed 任务残留的 log。
- 没有收敛的 SCF。
- 只为了导出 overlap 的非自洽任务。

### 从 log 读取 Fermi

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au11.0-s3p2d2f1 \
  --openmx-basis Mo=Mo11.0-s3p2d2 \
  --openmx-basis S=S9.0-s2p2d1 \
  --output-dir OUT_DIR \
  --fermi-log converged_openmx.out \
  --require-fermi-energy \
  --overwrite
```

### 直接指定 eV

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au11.0-s3p2d2f1 \
  --openmx-basis Mo=Mo11.0-s3p2d2 \
  --openmx-basis S=S9.0-s2p2d1 \
  --output-dir OUT_DIR \
  --fermi-energy-ev -4.253040486253099 \
  --require-fermi-energy \
  --overwrite
```

---

## DeepH Inference 前检查 / Pre-Inference Check

### 为什么要检查

典型报错：

```text
TypeError: mul got incompatible shapes for broadcasting: (16, 2704), (16, 676)
```

这里：

```text
2704 = 52 x 52
676  = 26 x 26
```

这通常说明 spinful/spinless 数据混用，而不是 JAX 本身的问题。

### 检查 spinless 模型

```bash
direct-overlap-check DFT_OUT_DIR --expect-spin spinless
```

### 检查 spinful 模型

```bash
direct-overlap-check DFT_OUT_DIR --expect-spin spinful
```

### 同时要求 Fermi provenance

```bash
direct-overlap-check DFT_OUT_DIR \
  --expect-spin spinful \
  --require-fermi-energy
```

### 检查指定 Fermi 值

```bash
direct-overlap-check DFT_OUT_DIR \
  --expect-spin spinful \
  --require-fermi-energy \
  --expect-fermi-energy-ev -4.253040486253099
```

---

## Band Calculation Guard

### 目标

防止这类 `calc-band` 报错：

```text
ComplexWarning: Casting complex values to real discards the imaginary part
ValueError: could not broadcast input array from shape (52,52) into shape (26,26)
```

含义：

- `hamiltonian.h5` 是 complex spinful。
- `info.json` 或 `overlap.h5` 却按 spinless 或错误 basis 处理。
- `dock` 尝试把 `52 x 52` Hamiltonian block 塞进 `26 x 26` slice。

### 推荐用法

```bash
direct-overlap-band-prep CASE_ROOT \
  --out-dir CASE_ROOT/band_ready \
  --overwrite \
  --yes
```

然后：

```bash
cd CASE_ROOT/band_ready
dock compute eigen calc-band ./ --parallel-num 5 --thread-num 1
```

### 交互模式

不加 `--yes` 时，如果在终端运行，会先显示诊断并询问：

```bash
direct-overlap-band-prep CASE_ROOT --out-dir CASE_ROOT/band_ready --overwrite
```

你会看到类似：

```text
检测结果 / Diagnosis:
- Hamiltonian source: ...
- Overlap source: ...
- Band spin mode: spinful
- info.json spatial orbits_quantity: 596
- per-element spatial orbitals: {'Au': 26, 'S': 13, 'Mo': 19}
- hamiltonian.h5 dtype: complex64, max block shape: [52, 52]
- overlap.h5 dtype: float64, max block shape: [26, 26]

说明 / Note:
- 对 SOC/spinful band 计算，hamiltonian.h5 是双倍 spinor block。
- 但 overlap.h5 应保持 spatial/spinless，dock 会在内部扩展 S。
- 不要把 orbits_quantity 改成两倍；它仍然是 spatial orbital 总数。
```

### 只诊断

```bash
direct-overlap-band-prep CASE_ROOT --diagnose-only
```

### 自动运行 calc-band

```bash
direct-overlap-band-prep CASE_ROOT \
  --out-dir CASE_ROOT/band_ready \
  --overwrite \
  --yes \
  --run-calc-band \
  --parallel-num 5 \
  --thread-num 1
```

### 输出目录

`band_ready/` 会包含：

```text
band_ready/
  POSCAR
  K_PATH
  info.json
  overlap.h5
  hamiltonian.h5
  band_prepare_manifest.json
```

---

## 输出文件 / Output Files

### direct-overlap 输出

```text
OUT_DIR/
  POSCAR
  info.json
  overlap.h5
  direct_overlap_manifest.json
```

### info.json 关键字段

```json
{
  "atoms_quantity": 28,
  "orbits_quantity": 596,
  "orthogonal_basis": false,
  "spinful": true,
  "fermi_energy_eV": -4.253040486253099,
  "elements_orbital_map": {
    "Au": [0, 0, 0, 1, 1, 2, 2, 3],
    "S": [0, 0, 1, 1, 2],
    "Mo": [0, 0, 0, 1, 1, 2, 2]
  }
}
```

注意：

- `orbits_quantity` 是 spatial orbital 总数。
- 即使 `spinful=true`，`orbits_quantity` 也不翻倍。
- spinful 翻倍体现在 matrix block shape 中。

### HDF5 matrix 文件结构

`overlap.h5` 和 `hamiltonian.h5` 通常包含：

```text
atom_pairs
chunk_boundaries
chunk_shapes
entries
```

`chunk_shapes` 是最重要的防呆对象。例如：

```text
spatial Au block: 26 x 26
spinful Au block: 52 x 52
```

---

## 常见错误和解决办法 / Troubleshooting

### 1. `2704 vs 676`

报错：

```text
mul got incompatible shapes for broadcasting: (16, 2704), (16, 676)
```

原因：

- `2704 = 52 x 52`
- `676 = 26 x 26`
- spinful 和 spinless overlap/model 数据混用了。

解决：

```bash
direct-overlap-check DFT_OUT_DIR --expect-spin spinful
```

或重新生成：

```bash
direct-overlap DATA_DIR BASIS_DIR ... --spinful --expect-spin spinful --overwrite
```

### 2. `52x52 -> 26x26`

报错：

```text
ValueError: could not broadcast input array from shape (52,52) into shape (26,26)
```

原因：

- `hamiltonian.h5` 是 spinful。
- `info.json` 写成了 `spinful=false`，或者 basis map 不匹配。

解决：

```bash
direct-overlap-band-prep CASE_ROOT --out-dir CASE_ROOT/band_ready --overwrite --yes
cd CASE_ROOT/band_ready
dock compute eigen calc-band ./ --parallel-num 5 --thread-num 1
```

### 3. S 原子 13 vs 14 orbitals

现象：

```text
H shape expects S=13 spatial orbitals
overlap/info imply S=14 spatial orbitals
```

常见原因：

```text
正确: S9.0-s2p2d1 -> 13
错误: S9.0-s3p2d1 -> 14
```

解决：使用和训练模型完全一致的 OpenMX basis label。

### 4. `orbits_quantity mismatch`

原因：

- `info.json` 的 `orbits_quantity` 被错误翻倍。
- 或者 `elements_orbital_map` 和 POSCAR 不一致。

规则：

```text
orbits_quantity = spatial orbital count
not spinful matrix dimension
```

### 5. Fermi energy 不可信

如果没有提供 `--fermi-energy-ev` 或 `--fermi-log`，`info.json` 里的 Fermi 可能只是 `0.0 eV` placeholder。

如果下游依赖 Fermi，请使用：

```bash
--require-fermi-energy
```

### 6. OpenMX normalization warning

OpenMX `.pao` 可能出现 normalization warning。默认允许继续，是因为 OpenMX 官方 basis 不一定逐函数严格归一化。

严格模式：

```bash
--strict-norm-check
```

---

## 大体系建议 / Large-System Notes

对于上千原子的 SOC/noncollinear 大体系：

- 不建议为了 overlap 跑完整 OpenMX SCF。
- `overlap.h5` 可以直接由结构和 basis 生成。
- Fermi energy 不能从 overlap-only 得到。
- 如果要 band，需要下游预测或计算得到 `hamiltonian.h5`。
- `dock compute eigen calc-band` 会构造 k-space 矩阵，仍然可能有内存压力。

运行前建议估算：

- atom count
- spatial orbital count
- spinful dimension
- `hamiltonian.h5` size
- `overlap.h5` size
- dense matrix memory lower bound

简单估算：

```text
real dense matrix memory    ≈ N^2 * 8 bytes
complex dense matrix memory ≈ N^2 * 16 bytes
```

其中 `N` 是 matrix dimension。spinful/SOC 时通常是 `2 * spatial_orbits_quantity`。

---

## 推荐工作流 / Recommended Workflows

### Workflow A: overlap for spinless inference

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au11.0-s3p2d2f1 \
  --openmx-basis Mo=Mo11.0-s3p2d2 \
  --openmx-basis S=S9.0-s2p2d1 \
  --output-dir DFT_OUT \
  --expect-spin spinless \
  --overwrite

direct-overlap-check DFT_OUT --expect-spin spinless
```

### Workflow B: overlap for spinful inference

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au11.0-s3p2d2f1 \
  --openmx-basis Mo=Mo11.0-s3p2d2 \
  --openmx-basis S=S9.0-s2p2d1 \
  --output-dir DFT_OUT \
  --spinful \
  --expect-spin spinful \
  --overwrite

direct-overlap-check DFT_OUT --expect-spin spinful
```

### Workflow C: band from predicted spinful Hamiltonian

```bash
direct-overlap-band-prep CASE_ROOT \
  --out-dir CASE_ROOT/band_ready \
  --overwrite \
  --yes

cd CASE_ROOT/band_ready
dock compute eigen calc-band ./ --parallel-num 5 --thread-num 1
```

---

## 开发和验证 / Development and Validation

### 本地语法检查

```bash
python -m py_compile src/direct_overlap/cli.py src/direct_overlap/__init__.py
```

### CLI smoke test

```bash
direct-overlap --help
direct-overlap-check --help
direct-overlap-band-prep --help
```

### Band-prep smoke test

```bash
direct-overlap-band-prep CASE_ROOT --diagnose-only
direct-overlap-band-prep CASE_ROOT --out-dir CASE_ROOT/band_ready --overwrite --yes
```

### 版本检查

```bash
python - <<'PY'
import importlib.metadata as m
print(m.version("abacus-openmx-overlap"))
PY
```

---

## 设计原则 / Design Principles

- Fail early: 在 `deeph-infer` 或 `calc-band` 崩溃前先检查。
- Keep provenance: 记录输入 hash、basis hash、软件版本、Fermi 来源。
- Do not fake physics: 不伪造 Fermi energy，不把未收敛输出当真值。
- Keep spin explicit: spinful/spinless 必须显式写入命令和 manifest。
- Keep band convention separate: `calc-band` 的 overlap 约定和某些 inference 场景不同，不能混用。

---

## 许可证 / License

中文：本项目使用 MIT License。

English: This project is released under the MIT License.
