# 标准 RNA 短缺口闭合

`mdna.close_rna_gap_from_anchors()` 可在已登记的标准 RNA dinucleotide suite 中，搜索满足两端锚定条件的短单链片段。当前实现是受预算限制的确定性枚举原型；可选的环境筛选按调用方给定的统一重原子距离阈值排除严重重叠。该筛选不是元素特异的范德华判据，也不替代扭转角优化、分子力学松弛或整结构验收。

## 输入合同

`sequence` 包含缺口两侧的锚定残基和中间缺失残基，至少 3 个标准碱基。例如 `CGU` 表示 C 和 U 为固定边界，中间缺失 G。锚点必须位于同一条线性链内。

调用方提供两端基帧、两端锚残基的重原子坐标映射，以及以下参数：

- 坐标单位；
- 端点平移和旋转残差上限；
- 共享残基 RMSD 上限；
- 共享原子位移、内部键长变化、原锚端口回填后的 `O3'–P` 键长偏差及锚原子位移上限；
- 最大 suite 展开数和最多返回候选数。

帧原点使用 `coordinate_unit` 指定的单位；锚原子坐标始终使用 Å。锚原子映射至少含 3 个原子名及其三维坐标。每个候选都必须包含调用方指定的锚原子名，否则该候选会被拒绝。

环境筛选为可选项。启用时，调用方同时传入 `environment_atoms` 和正数 `min_environment_distance_A`。每个环境原子记录包含唯一 `atom_id`、非氢元素符号和 `xyz_A` 坐标。求解器只比较缺失残基重原子与环境重原子；当任一距离小于阈值时拒绝该候选。阈值对所有元素相同，因此只适合作为明确设定的粗筛条件。`excluded_environment_pairs` 可列出已知共价原子对，格式为 `(缺失残基索引, 缺失原子名, 环境 atom_id)`；求解器不从坐标自动推断共价排除。环境坐标、身份和排除列表会进入 SHA-256 报告。

## 调用示例

阈值必须由调用方按研究问题、输入分辨率和验证方案设定。以下数值仅演示参数形式，不是通用推荐值。

```python
import mdna
import numpy as np
from mdna import RNAFragmentLibrary

library = RNAFragmentLibrary.from_json("rna_suite_library.json")

# frame: 原点以及右手正交 x/y/z 基向量，形状为 (4, 3)。
start_frame = np.asarray(start_frame_A, dtype=float)
end_frame = np.asarray(end_frame_A, dtype=float)

result = mdna.close_rna_gap_from_anchors(
    library,
    "CGU",  # 两端锚残基加中间缺失的 G
    start_frame,
    end_frame,
    start_anchor_atoms,  # {"C1'": [x, y, z], ...}，坐标单位 Å
    end_anchor_atoms,
    coordinate_unit="angstrom",
    max_translation_residual_A=0.5,
    max_rotation_residual_degrees=10.0,
    max_shared_residue_rmsd_A=1.0,
    max_shared_atom_displacement_A=0.5,
    max_bond_length_deviation_A=0.1,
    max_anchor_atom_displacement_A=0.5,
    max_anchor_connection_length_deviation_A=0.2,
    max_search_states=100_000,
    max_returned_candidates=25,
    # 可选粗筛；阈值需由研究者按具体结构和目的设定。
    environment_atoms=environment_atoms,
    min_environment_distance_A=1.5,
)

report = result.to_report()
if result.candidates:
    best = result.candidates[0]
    print(best.suite_ids, best.endpoint_translation_residual_A)
```

## 搜索和结果解释

求解器按 suite ID 排序，以深度优先方式枚举内部态片段，并从 5′ 锚逐段传播下游基帧。路径必须满足共享残基连续性、末端位置/方向、锚原子位移、内部装配键长变化，以及使用原始固定锚原子坐标计算的两条 `O3'–P` 连接键偏差等硬限值。启用环境筛选时，报告每个保留候选的最小环境距离，并在 `rejections.environment_clash` 记录因阈值被拒绝的候选数。该距离不进入候选评分，避免把单一统一阈值解释为物理能量或普适排序依据。

`status` 有 3 种值：

- `complete`：有限 suite 库搜索已穷尽，并找到至少一个符合硬限值的候选；
- `no_solution_found`：有限 suite 库搜索已穷尽，未找到符合硬限值的候选；
- `search_budget_exhausted`：达到调用方设定的展开上限。若结果包含候选，这些候选满足硬限值，但搜索未穷尽，不能称为全局最优。

未覆盖的二核苷酸会抛出 `UnsupportedSuiteSequenceError`。参数、帧或锚原子格式错误会抛出 `ValueError`。报告含来源哈希、两端帧和锚原子哈希、搜索数量、候选数量、限值、拒绝原因计数及各候选的评分分项。

返回的 `assembly` 包含两端锚残基和缺失残基的候选原子。集成到原模型时，只插入 `result.missing_residue_indices` 指向的残基，保留原结构中的锚坐标。该 API 不自动加载环境、不推断共价排除，不检验配对、堆叠、糖环适配、密度或力场能量，也不移动候选锚点来掩盖闭合误差。环境筛选的阈值、原子范围和排除列表由调用方负责。该 API 只生成软件/局部几何候选，不代表结构科学验收。

## 本地条目留一评估

执行以下命令可复算一个单残基缺口案例：

```bash
conda run -n dip2 python scripts/rna_pilot/evaluate_rna_gap_closure_loo.py
```

求解器使用 4U3R chain A 的两端锚点和预先固定的阈值。生成候选前，脚本从 suite 库中排除整个 4U3R 条目；缺失 G 的坐标只用于候选生成后的 RMSD 评价。完整结果见 `artifacts/rna_pilot/gap_benchmarks/4u3r_A_CGU_single_residue.json`。

该条目与其余训练源同属一个 8-mer 离子条件结构组。复算时将目标链 A 完全排除于环境坐标，仅用其他链的 208 个重原子做 1.5 Å 统一距离粗筛；33 个几何候选中没有候选触发该阈值，最佳候选的最小跨链距离为 2.610 Å。该筛查没有元素特异半径，也未包含目标链其余残基，因此不能作为完整非键冲突验收。结果只验证单条目留出下的搜索行为和局部坐标重建，不是独立结构家族盲测，不证明一般 RNA 构象泛化，也不含地图证据。

### 可选：返回坐标精确固定在实验锚点

设置 `pin_anchor_coordinates=True` 时，调用者须提供装配模板涉及的所有锚原子。枚举和初筛保持原逻辑；候选装配后写入原始观测锚坐标，再检查受影响的键相对未固定装配的变化，及最终两条 O3′–P 键相对各自来源片段的长度偏差。超限候选计入 `rejections.pinned_anchor_geometry`。报告的 `anchors_pinned` 指示此模式。默认仍保持既有行为。该模式没有优化连接键角、糖手性和能量，使用前需对返回坐标做更完整的结构质控。

### 固定双锚后的局部刚体修复对照

用 WSL dip2 运行：`conda run -n dip2 python -B scripts/rna_pilot/evaluate_1rna_transfer.py --entry 1RNA --supplement 157D --sensitivity`。诊断先使用探索性的 2 Å 预筛，再精确固定锚坐标，只允许缺失残基作刚体调整，最大位移超过 0.50 Å 时拒绝。4U3L 的局部连接几何提供靶值；遮蔽的 1RNA 缺失残基坐标只参与事后评分。两例中一例通过现有键长、键角和近接触诊断，另一例仍有 88.58° 的低角而被拒绝。该结果不足以验证通用 RNA 补缺。

可选磷酸转动的开发对照随同一命令输出到 `phosphate_gap_trials`：B 链的低角从刚体方案的 88.58° 提升到 92.29°，但遮蔽重原子 RMSD 从 0.280 Å 增至 0.423 Å。先保留已经通过诊断的刚体候选；新增自由度只在必要时评估，仍需用未参与开发的结构验证。

### 独立条目 438D

运行 `conda run -n dip2 python -B scripts/rna_pilot/evaluate_1rna_transfer.py --entry 438D --supplement 157D --sensitivity`。在 14 个窗口中，4 个进入几何搜索，B 链 13 CGC 的严格固定锚点方案返回 13 个候选（含重复），排序首选相对遮蔽坐标的 RMSD 为 0.412 Å。候选选择按几何优先保留刚体，磷酸转动只作为诊断后备，实验遮蔽坐标仅用于评价。

### 按缺失区坐标去重

可选 `deduplicate_missing_geometry=True` 在排序和返回限额内保留每组几何的首选。键由缺失区的残基索引、原子名和 0.001 Å 网格坐标构成；不会对齐模型，也不会跨网格边界合并。报告保留原始通过路径数 `total_compatible_closures`，另给出 `distinct_missing_geometries`。438D 的同一严格固定锚点窗口由 13 个通过路径归并为 7 个坐标候选，首选 RMSD 不变。

### 2G3S 条目留出

冻结容差后运行：`conda run -n dip2 python -B scripts/rna_pilot/evaluate_1rna_transfer.py --entry 2G3S --supplement 157D --sensitivity`。60 个窗口中 20 个有模板，4 个 CGU 窗口返回精确固定锚点候选；这些窗口来自同一晶体条目。四个首选中一个有明显连接角超差，故候选返回数不能替代最终几何验收。该条目的显式帧装配无候选。

### 可选：筛查最终锚定连接角

与 `pin_anchor_coordinates=True` 一起传入 `anchor_angle_windows_deg`。键名可为 `c3_o3_p_deg`、`o3_p_o5_deg`、`p_o5_c5_deg`；每个值为包含上下界的度数区间。筛查以精确固定锚点后的最终装配坐标为准，两端任一请求的角度超差、缺失或退化均拒绝，计入 `rejections.pinned_anchor_angles`。筛查在排序和去重之前执行。未提供时保留原行为；条目留出使用 100–140°、90–130°、95–135°，仅作当前对照的宽松诊断。
