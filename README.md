# ABACUS/OpenMX Direct Overlap

## 简介 / Overview

中文：这个项目从 `POSCAR` 和数值原子轨道直接生成 DeepH 格式的 `overlap.h5`，支持 ABACUS `.orb` 和 OpenMX `.pao` 两种 basis。

English: This project generates a DeepH-format `overlap.h5` directly from a `POSCAR` and numerical atomic orbitals. It supports ABACUS `.orb` and OpenMX `.pao` basis files.

中文：它不运行 ABACUS `get_S`，不运行 OpenMX SCF，也不做 Hamiltonian 转换；目标只是计算 overlap matrix。

English: It does not run ABACUS `get_S`, does not run an OpenMX SCF calculation, and does not convert Hamiltonians. It only computes the overlap matrix.

## 安装 / Installation

中文：推荐在已经有 MPI、`h5py`、`numpy`、`scipy` 的 HPC Python 环境里安装。安装命令保持一条：

English: Installation is intended to be simple. On an HPC system with a working Python/MPI/scientific stack, run one command:

```bash
bash install.sh
```

中文：然后激活环境：

English: Then activate the environment:

```bash
source .venv/bin/activate
direct-overlap --help
```

中文：如果你的集群需要先加载 MPI 环境，请在运行 `direct-overlap` 前先 `module load` 或 `source` 对应环境脚本。

English: If your cluster requires an MPI runtime to be loaded first, run the proper `module load` or `source` command before calling `direct-overlap`.

## ABACUS 用法 / ABACUS Usage

中文：`DATA_DIR` 必须包含 `POSCAR`；`ABACUS_ORB_DIR` 里每个元素应有唯一的 `ELEMENT_*.orb` 文件。

English: `DATA_DIR` must contain `POSCAR`; `ABACUS_ORB_DIR` should contain exactly one `ELEMENT_*.orb` file for each element.

```bash
direct-overlap DATA_DIR ABACUS_ORB_DIR \
  --basis-code abacus \
  --output-dir OUT_DIR \
  --ecut 100
```

## OpenMX 用法 / OpenMX Usage

中文：OpenMX `.pao` 文件只包含 radial tables，所以必须明确写出每个元素使用的 OpenMX basis label。

English: OpenMX `.pao` files contain radial tables, so the OpenMX basis label must be provided explicitly for each element.

```bash
direct-overlap DATA_DIR OPENMX_PAO_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au7.0-s2p2d2f1 \
  --openmx-basis Mo=Mo7.0-s3p2d2f1 \
  --openmx-basis S=S7.0-s3p3d2f1 \
  --output-dir OUT_DIR \
  --ecut 100
```

中文：如果 `.pao` 文件不在默认位置，可以用 `--basis-map` 指定：

English: If a `.pao` file is stored elsewhere, point to it with `--basis-map`:

```bash
--basis-map Au=/path/to/Au7.0.pao
```

## 结构保护 / Structure Guards

中文：为了避免误用结构，建议对固定体系加 formula 和 atom-count 检查：

English: To avoid using the wrong structure, add formula and atom-count guards for fixed systems:

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code abacus \
  --output-dir OUT_DIR \
  --expect-formula Au16S8Mo4 \
  --expect-natoms 28
```

## Spin / 自旋

中文：默认输出是 spinless spatial overlap。这对普通 collinear spin、没有 SOC/noncollinear 的情况通常足够，因为 overlap 不依赖自旋，spin-up 和 spin-down 共用同一个空间 overlap。

English: The default output is a spinless spatial overlap. This is normally sufficient for collinear spin calculations without SOC/noncollinearity because the overlap is spin independent and shared by spin-up and spin-down channels.

中文：如果你的下游 DeepH 数据格式需要 spinful/spinor matrix，可以加 `--spinful`。它会按照 `deepx-dock` 的原生策略把空间 overlap 复制到自旋对角块，也就是 `S_spin = I_2 \otimes S`。

English: If your downstream DeepH format expects a spinful/spinor matrix, add `--spinful`. It follows the native `deepx-dock` strategy and duplicates the spatial overlap onto the spin diagonal blocks, i.e. `S_spin = I_2 \otimes S`.

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code abacus \
  --output-dir OUT_DIR \
  --spinful \
  --expect-spin spinful
```

中文：注意，这不会生成 SOC Hamiltonian，也不会引入 spin-mixing overlap；它只把 spin-independent overlap 写成 spinful matrix 形状。

English: Note that this does not generate an SOC Hamiltonian and does not add spin-mixing overlap terms. It only writes the spin-independent overlap in a spinful matrix shape.

## Fermi Level / 费米能级

中文：`overlap.h5` 只由结构和 basis 决定，不能从 overlap-only 计算中得到 SCF Fermi level。直接生成时如果不提供 Fermi energy，`info.json` 只能写入 `0.0 eV` 占位值，这不应被当作收敛的费米能级。

English: `overlap.h5` is determined only by the structure and basis. An SCF Fermi level cannot be derived from an overlap-only calculation. If no Fermi energy is provided, `info.json` can only contain a `0.0 eV` placeholder, which should not be treated as a converged Fermi level.

推荐从收敛的 OpenMX/DFT 输出读取：

Recommended usage with a converged OpenMX/DFT output:

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --output-dir OUT_DIR \
  --fermi-log openmx.out \
  --require-fermi-energy
```

也可以直接给 eV：

You can also pass the value in eV directly:

```bash
direct-overlap DATA_DIR BASIS_DIR \
  --basis-code openmx \
  --output-dir OUT_DIR \
  --fermi-energy-ev 1.2345 \
  --require-fermi-energy
```

中文：OpenMX 输出中的 `Chemical potential (Hartree)` 会被自动转换为 eV。

English: `Chemical potential (Hartree)` in OpenMX output is automatically converted to eV.

## 推理前检查 / Pre-Inference Check

中文：为了避免 `deeph-infer` 里出现类似 `2704 vs 676` 的 JAX shape 错误，建议在推理前检查 overlap 目录的 spin mode 和 HDF5 block shape。

English: To avoid JAX shape errors such as `2704 vs 676` in `deeph-infer`, check the overlap directory spin mode and HDF5 block shapes before inference.

如果模型是 spinless：

```bash
direct-overlap-check DFT_OUT_DIR --expect-spin spinless
```

如果模型是 spinful/spinor：

```bash
direct-overlap-check DFT_OUT_DIR --expect-spin spinful
```

同时要求 Fermi energy 必须来自显式输入：

Also require the Fermi energy to come from an explicit input:

```bash
direct-overlap-check DFT_OUT_DIR \
  --expect-spin spinful \
  --require-fermi-energy
```

中文：`direct-overlap` 生成文件后也会自动做同样的自洽检查，并把结果写入 `direct_overlap_manifest.json`。

English: `direct-overlap` runs the same consistency check after generation and stores the report in `direct_overlap_manifest.json`.

## 输出 / Outputs

中文：输出目录会包含：

English: The output directory contains:

- `overlap.h5`
- `info.json`
- `POSCAR`
- `direct_overlap_manifest.json`

中文：`direct_overlap_manifest.json` 会记录输入哈希、basis 哈希、软件版本、参数和 HDF5 dataset shapes，便于复现。

English: `direct_overlap_manifest.json` records input hashes, basis hashes, software versions, parameters, and HDF5 dataset shapes for reproducibility.

## 说明 / Notes

中文：OpenMX 官方 `.pao` 的 optimized radial functions 不一定像 ABACUS `.orb` 一样逐个严格归一化；OpenMX backend 默认只警告并记录 normalization deviation。需要强制失败时加 `--strict-norm-check`。

English: Optimized radial functions in official OpenMX `.pao` files are not always individually normalized as strictly as ABACUS `.orb` files. The OpenMX backend warns and records normalization deviations by default. Use `--strict-norm-check` to make this fatal.

## 许可证 / License

中文：本项目使用 MIT License。

English: This project is released under the MIT License.
