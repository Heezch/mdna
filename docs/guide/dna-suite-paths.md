# 标准 DNA suite 显式帧路径筛选

## 用途和边界

`DNAFragmentLibrary` 从来源可追溯的实验 DNA 二核苷酸构象中读取 suite。调用者提供每个碱基的目标姿态；库按序列和姿态残差筛选 suite，再按共享残基坐标连续性选出一条确定性路径。

当前路径库来自 1KX5 的两条 DNA 链，共 288 个内部 suite，覆盖 15/16 种序列上下文，缺少 `CG`。两条链来自同一核小体结构家族；该来源集不能证明跨家族构象泛化。

工作区还包含独立条目 1BNA 的来源数据 `artifacts/gap_benchmarks/1bna_dna_suites.json`。数据含 18 个内部 suite，覆盖 `AA/AT/CG/GA/GC/TC/TT` 七种上下文。该 1.6 Å X-ray B-DNA 十二聚体来自 `mdna/atomic/ddd/1bna.pdb`，文件 SHA-256 为 `4f74ccaff0a1ede01a2fb08184a0fc308bceeba42fbe636a044515fb754a4c7d`。提取和组件/键图检查已通过，但盲化缺口求解和正式结构家族分类尚未完成；目前只可作为候选留出来源。

该 API 返回分别放置的 suite 及路径诊断。`assemble_suite_path()` 可在显式共享原子位移和来源键长变化上限内合并为连续重原子段，并能输出 MDTraj/PDB；两端保留开放延续端口，不补端基。路径筛选只比较目标基帧；通过限值不等于全链化学几何或结构科学验收。

## 帧约定

每个目标帧形状为 `(4, 3)`。第 1 行是碱基原点，后 3 行依次是 MDNA `ReferenceBase` 的 `b_D`、`b_L`、`b_N` 单位轴。原点单位由 `coordinate_unit` 指定；轴无量纲。`angstrom` 与 `nm` 均可用。

路径中所有碱基都需有显式姿态。API 不会从中心线推断碱基滚转或倾斜。

## 最小调用

```python
from pathlib import Path

from mdna import DNAFragmentLibrary

library = DNAFragmentLibrary.from_json(
    Path("artifacts/gap_benchmarks/1kx5_dna_suites.json")
)
# frames 由调用方提供，shape 为 (len(sequence), 4, 3)。
path = library.select_suite_path(
    "TCA",
    frames,
    coordinate_unit="angstrom",
    max_translation_residual_A=0.5,
    max_rotation_residual_degrees=5.0,
    max_shared_residue_rmsd_A=0.5,
)
```

三个容差都必须由调用方显式给出，并且是有限正数。上述数值仅说明 API 用法，不是经验证的推荐阈值。

## 检查结果

- `library.available_sequences` 返回库内实际出现的二核苷酸组合。
- `library.get_suites("CG")` 抛出 `UnsupportedDNASuiteError`，错误信息列出当前覆盖序列。
- 每个路径步骤的下游平移与旋转残差保存在 `path.placements`。
- 相邻 suite 的共享残基 RMSD 保存在 `path.shared_residue_rmsd_A`。
- `path.suite_ids` 和 `path.normalized_score` 支持复核确定性选择结果。
- `path.to_report()` 返回 JSON 兼容报告，记录算法、输入姿态帧 SHA-256、来源文件哈希、调用方限值、suite ID 和逐段残差。
- 无 suite 满足显式姿态或共享坐标限值时，API 抛出 `NoCompatibleDNASuitePathError`。

## 合并和导出

```python
assembly = library.assemble_suite_path(
    path,
    max_shared_atom_displacement_A=0.02,
    max_bond_length_deviation_A=0.02,
)
trajectory = assembly.to_mdtraj()
assembly.save_pdb("assembled.pdb")
```

限值必须由调用方给出，示例数字不构成推荐值。装配检查 suite 的来源 ID、序列、原子清单和键图；共享原子超限或装配使任一来源键长变化超限时拒绝。接受时保留先出现的共享坐标，不平均。结果记录来源 suite ID、共享界面 RMSD/最大位移、最大来源键长变化以及 5′ `P` 和 3′ `O3'` 开放端口。MDTraj/PDB 导出包含重原子共价图及显式 `CONECT` 连接记录；不生成氢或端羟基。

WSL 回归使用同一 1KX5 来源的两个相邻 suite，验证序列连续性、跨残基 `O3'–P` 键和 MDTraj→PDB→MDTraj 键图往返。该回归只验证软件行为。回归未评估独立结构家族泛化、糖环、碱基配对、整链几何、环境冲突或实验符合度。

构造库时，API 会核对来源哈希和标准 DNA 字母。
API 也会核对原子清单、组件模板 SHA-256、分子内键图、跨残基 `O3'–P` 键和右手刚体变换。
几何或身份不符的数据会被拒绝。

## 复算来源

- 数据提取脚本：`scripts/gap_benchmarks/extract_1kx5_dna_suites.py`
- JSON 构象集：`artifacts/gap_benchmarks/1kx5_dna_suites.json`
- 候选留出构象集：`artifacts/gap_benchmarks/1bna_dna_suites.json`（由 `--input mdna/atomic/ddd/1bna.pdb --output artifacts/gap_benchmarks/1bna_dna_suites.json --entry-id 1BNA` 提取）
- 组件注册：`mdna/chemistry.py`
- API：`mdna/dna_fragments.py`
- 自动化检查：`tests/test_dna_suite_extraction.py` 与 `tests/test_dna_fragment_placement.py`

路径/装配 API 尚未把多来源 suite 合并成一个已验证构象库。使用者宜保存输入结构、输入帧、限值、suite ID 和 JSON 来源文件哈希。真实末端化学、双锚闭合、遮蔽 1BNA 的独立条目评估、正式结构家族验证和全结构冲突检查仍未完成。
