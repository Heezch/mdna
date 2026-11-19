# 使用显式碱基帧构建 RNA 单链片段

## 适用范围

`mdna.build_rna_chain_from_frames()` 按调用者提供的每残基参考帧，从有来源的 RNA 二核苷酸 suite 库筛选并装配单链重原子坐标。该 API 当前只接受标准 RNA `A/C/G/U`，且每种连续二核苷酸和所需端基状态都必须有库记录。

该 API 不从中心线推导碱基方向或螺旋相位，不生成互补链，不移动目标帧，也不闭合缺口。该函数不生成氢坐标，不执行能量最小化、碰撞筛选或密度拟合。调用方需要先提供具有明确化学含义的碱基帧。

## 输入帧合同

`base_frames` 的形状必须为 `(len(sequence), 4, 3)`。每个 `(4, 3)` 矩阵按行存放碱基帧原点和三个基向量：

1. 第 0 行是碱基帧原点；
2. 第 1–3 行是右手、正交归一的 x/y/z 基向量；
3. `coordinate_unit` 显式声明原点坐标使用 `angstrom` 或 `nm`。

API 将坐标及残差统一用 Å 报告。基向量无量纲，不能随坐标单位缩放。

## 构建并保存

下面示例要求调用者已把结构帧放入 `base_frames`。路径容差由调用者明确给出；这些值不会被解释成普适 RNA 几何阈值。

```python
import json
from pathlib import Path

import mdna

root = Path.cwd()
library = mdna.RNAFragmentLibrary.from_json(
    root / "artifacts/rna_pilot/reference/rna_suite_library.json",
    terminal_path=root / "artifacts/rna_pilot/reference/rna_terminal_suite_library.json",
)

result = mdna.build_rna_chain_from_frames(
    library,
    "UCGUACGA",
    base_frames,
    coordinate_unit="angstrom",
    max_translation_residual_A=0.5,
    max_rotation_residual_degrees=6.0,
    max_shared_residue_rmsd_A=0.5,
    max_shared_atom_displacement_A=0.5,
    max_bond_length_deviation_A=0.1,
    include_terminal_suites=True,
)

result.assembly.save_pdb(root / "artifacts/rna_pilot/output/UCGUACGA.pdb")
(root / "artifacts/rna_pilot/output/UCGUACGA.build.json").write_text(
    json.dumps(result.to_report(), indent=2) + "\n",
    encoding="utf-8",
)
```

示例中的阈值仅用于说明参数单位和调用形式，不是推荐值或已验证的通用设置。正式运行前须按结构问题预先登记阈值，并用独立来源结构检查输出。

## 端部与失败处理

`include_terminal_suites=True` 时，序列必须至少有 3 个残基。第一个 dinucleotide 要匹配 5′-OH suite，最后一个要匹配 3′-OH suite，中间步骤使用内部态 suite。当前归档端部库只覆盖 5′ `UC` 和 3′ `GA`。缺少序列或端基覆盖时，API 抛出 `UnsupportedSuiteSequenceError`；目标帧不相容或片段不能连续装配时，API 抛出 `NoCompatibleRNAFragmentPathError` 或 `ValueError`。

`include_terminal_suites=False` 表示输出片段两端保持开放磷酸二酯延续端口；这不等于完整自然链端。

## 可复现记录和验收边界

`result.to_report()` 返回 JSON 兼容记录，其中包含：序列、坐标单位、目标帧 SHA-256、suite 来源清单哈希、化学注册表哈希、选择算法标识、suite ID、全部限值、帧残差、共享原子位移和边界状态。PDB 保存原子坐标；构建报告保存输入和筛选诊断。两者应一起归档。

库内 suite 来源、帧及目标结构若来自同一 PDB 条目，只能验证软件重放和格式行为，不能作为独立验证。几何质量、整链冲突、RNA 双链配对、电子密度支持和双锚缺口闭合需要分别验收。未出现的碱基组合必须补充有来源的数据，不能以其它序列的 suite 替代。

### 导出装配片段的 mmCIF

已装配的标准 DNA/RNA 单链片段均可调用 `assembly.save_mmcif(path)`。该方法保留组装顺序、标准组件名、Å 坐标及显式共价边及 PDBx 连接表可表示的键级。芳香键保留连接，键级标为未知；原始键级仍见装配对象。链与残基重新从 A/1 编号；该接口不是原始坐标文件的无损转换器，不保留输入作者编号、实验占有率或温度因子。片段两端的延续口及片段来源仍以装配对象和 JSON 报告为准。文件用于科研交换，未做 OneDep 字典验收。连接表参照 [wwPDB 的连接字段说明](https://mmcif.wwpdb.org/docs/user-guide/resources/ligands.html)。

### 双端帧拟合（实验选项）

`build_rna_chain_from_frames(..., balance_frames=True)` 对每个套件施加一次刚体变换，让两端分担帧偏差。平移和旋转限值分别约束两端中的较大残差；相邻片段还须在排序前通过共享原子和键长变化检查，最终装配会再复核。目标帧不会被重写，返回坐标也不保证精确落在目标帧。默认仍沿用上游帧对齐。

独立 438D 的 4 个覆盖窗口中，B 链从残基 12 开始的 GCG 通过，两个 O3′–P 距离为 1.624 和 1.632 Å，片段内无低于 2 Å 的非键合重原子近接触；另外三个窗口未通过。1RNA 的 24 个覆盖窗口仍全部拒绝。该结果限于三残基片段，不证明全链、双链、任意中心线或沉积质量。

438D B12–14 片段放回沉积不对称单元后，12 个糖环手性符号均与参考一致，没有低于 2 Å 的外部非键合近接触；两端预期共价接口独立量得 1.864 Å 和 1.761 Å。该检查不包含晶体对称邻居、不评价糖环构象，也不能替代固定两端后的装配验收。

同一窗口的边界刚体试验以 4U3L 的 1.603 Å 中位数为参考，保持周围坐标固定，使两个接口缩至 1.608/1.603 Å。最大原子位移 0.315 Å，无新增近接触。报告保存前后键角与扭转；试验没有复验调整后的目标帧误差，不作为可交付路径模型。

复核最终坐标时，三个碱基帧的最坏偏差为 0.346 Å 和 3.301°，仍满足 0.50 Å/6° 限值。加入逐帧约束的求解给出近同结果，两个接口仍为 1.608/1.603 Å。以上属于单窗联合诊断；下一步用冻结的预算测试新的独立窗口。

### 定位装配失败阶段

调用时传入空字典 `diagnostics={}`，成功或兼容性失败后均可读取已走过的步骤。每步记录已有模板数、仅平移失败、仅旋转失败、两项都失败、通过数，以及按最大归一化超差排序的最近套件及成对残差。拼接阶段记录已尝试组合、连续性和装配拒绝数。计数只覆盖实际检查到的步骤；遇到首次无可用候选即停止。诊断不改变原阈值或选取策略。

### 共享残基刚体协调（验证脚本）

三残基留出对照新增 `shared_residue_rigid_probe`：保留一套共享残基内部几何，以刚体姿态协调两个片段的重叠坐标，再检查两份来源的键长变化、最终帧偏差和片段内近接触。1RNA 新增 1 个通过窗口，2G3S 新增 3 个；仍需将这些片段接回实验结构复核。不改变默认装配。

追加完整上下文复核后，2G3S 的三个新增三残基窗口均通过帧、接口键长/键角、糖手性和不对称单元近接触的联合诊断；1RNA 链末窗口因唯一接口角为 83.03° 被拒。两者仍只覆盖三残基局部窗口，且未评价晶体对称邻居。

### 四至六残基多界面对照

使用 `--long-paths-only` 复跑固定的 1RNA、438D、2G3S 对照。总计 210 窗，72 窗有模板覆盖；两种完整共享残基起点各有 26 条路径进入最终装配，均因 O3′–P 键长变化超限被拒。逐路径失败信息附带超差原子对。后续需要同时求解多个界面，不能直接把三残基局部阳性推为整链支持。

联合调整多个共享残基后，同一面板 72 个覆盖窗中有 5 个达到片段级限值；1RNA 和 2G3S 合计 3 个四残基窗口进一步通过接回检查。五残基窗口有片段级候选，但未通过接回；六残基仍无候选。对照报告同时保留两个独立协调基线。

把 0.50 Å / 6° 容差直接加入共享残基联合求解后，原 72 个覆盖窗的通过数未变：5 个片段级阳性、3 个四残基上下文阳性。新结果见长路径报告的 `joint_frame_probe`。已有五残基候选仍需同时解决外部接口和近接触。

长路径报告中的 `joint_boundary_probe` 同次求解共享残基和外部接口，1RNA、2G3S 共三个四残基窗口无需第二次调整即可通过现有联合检查。五残基和六残基无通过结果；该检验与先装配再调接口的基线分别保存。

追加 [8FEP](https://www.rcsb.org/structure/8FEP) 条目留出，保持模板来源和阈值不变。四、五、六残基窗口的模板覆盖为 6/13、5/12、4/11。所有 15 个覆盖窗至少有一步缺少合格帧候选，未进入完整路径评价。未覆盖的 21 个窗口缺少 CU 或 UC。复跑使用 `--entry 8FEP --supplement 157D --long-paths-only`，报告区分缺少序列模板、缺少帧候选和进入完整路径评价的数量。

加入独立于目标坐标的 [438D](https://www.rcsb.org/structure/438D) 后，12 个新增内部套件使 8FEP 的序列模板覆盖从 15/36 升至 36/36。两轮使用相同的帧容差，所有窗口仍至少有一步超差，进入完整路径评价的数量仍为零。追加 `--supplement 438D` 即可复算；多来源工件保留每个来源的坐标哈希和合并清单哈希。这是针对前一轮发现的覆盖缺口做的工程对照，后续泛化评价需使用新留出条目。

再加入 [255D](https://www.rcsb.org/structure/255D) 的 9 个套件后，8FEP 的 7 个超差相邻步中有 1 个通过原帧容差，但所有 36 个窗口仍至少缺一个逐步候选。随后在同一套冻结条件下评估 [420D](https://www.rcsb.org/structure/420D)，48/72 窗有序列模板，唯一逐步帧均有候选的四残基窗卡在共享残基重叠约束。复算可在上条命令追加 `--supplement 255D`，并将 `--entry` 分别设为 `8FEP` 和 `420D`。

对 420D B26–29 的 AAUC 窗口，新增逐组重叠 RMSD 和相对两份源坐标的最小原子位移下界。两种刚体起点各有一条路径通过位移下界预筛，随后均未找到同时满足实际原子位移、来源键长变化、目标帧和外部接口约束的坐标。报告字段为 `closest_rejected_overlap_by_step`、`displacement_bound_probe` 和 `reverse_displacement_bound_probe`。

分开求解接口后，AAUC 的下游键长可在现有预算内达到 1.679 Å。上游接口至少需要 0.464 Å 的原子移动，超出允许的 0.350 Å 整体位移。这个下界适用于局部坐标固定的末端，后续需检验上游磷酸的局部调整。

将上游两条骨架扭转和下游 O3′ 的有界转动联合求解后，420D B26–29 的 AAUC 通过现有内部键、帧、手性、接口键长与键角、不对称单元重原子近接触筛查。分析坐标和键图见报告的 `terminal_backbone_angle_probe.best.candidate_geometry`。这是开发用窗口的局部验证，待冻结方法后复查新窗口。
