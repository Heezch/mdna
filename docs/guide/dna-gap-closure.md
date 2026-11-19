# 标准 DNA 单残基双锚缺口

## 能力边界

`mdna.close_dna_gap_from_anchors()` 是一个有界候选枚举原型：输入标准 A/C/G/T 线性链的三残基片段（上游固定锚、一个缺失位点、下游固定锚），以及两端锚残基的基帧和完整注册重原子坐标。求解器从库中寻找两段连续 dinucleotide suite，检查帧闭合、共享残基连续性、锚点偏移、两条 `O3'–P` 连接以及恢复锚点原坐标后的来源键长漂移。

该接口不处理多残基缺口、RNA、混合糖化学、分支/环、环境碰撞、双链配对约束或局部能量优化。返回候选是结构几何假设；需额外检查实验环境、配对/堆叠、键角、手性、糖环和地图证据。

## 输入与阈值

- `sequence` 长度必须为 3，顺序为 5′→3′，且每个字母为 A/C/G/T；首尾为固定锚，中心位点为缺失碱基。
- `start_frame` 与 `end_frame` 是两侧锚碱基的 `(4, 3)` 帧，首行为原点，后 3 行为右手正交基。`coordinate_unit` 必须显式指定 `angstrom` 或 `nm`。
- `start_anchor_atoms` 和 `end_anchor_atoms` 提供组件注册表要求的完整重原子名称到坐标映射。缺失或重复原子不得通过删减清单静默绕过。
- 所有几何硬限值和 `max_search_states`、`max_returned_candidates` 必须由调用者显式提供。限值表示筛选约束，不是通用推荐值。

```python
from mdna import DNAFragmentLibrary, close_dna_gap_from_anchors

library = DNAFragmentLibrary.from_json("artifacts/gap_benchmarks/1kx5_dna_suites.json")
result = close_dna_gap_from_anchors(
    library,
    "GAA",  # 固定 G、缺失 A、固定 A
    start_frame,
    end_frame,
    start_anchor_atoms,
    end_anchor_atoms,
    coordinate_unit="angstrom",
    max_translation_residual_A=0.75,
    max_rotation_residual_degrees=12.0,
    max_shared_residue_rmsd_A=0.8,
    max_shared_atom_displacement_A=0.8,
    max_bond_length_deviation_A=0.25,
    max_anchor_atom_displacement_A=0.8,
    max_anchor_connection_length_deviation_A=0.25,
    max_search_states=10000,
    max_returned_candidates=30,
)

report = result.to_report()
```

候选记录中的锚点偏移是在回填原锚坐标之前计算的；最终 `assembly` 会把锚点原子恢复到输入坐标，并重新核验来源键长。`search_budget_exhausted` 表示搜索未穷尽，不能解读为无解；`no_solution_found` 仅表示在当前库与当前阈值下穷尽搜索未找到候选。报告中的环境筛查为 `not_evaluated`。

无解报告还包含 `trial_metric_minima`、`trial_metric_evaluated_counts` 和 `assembly_rejection_reasons`。前两项记录每个搜索阶段实际观测到的单项最小残差及到达该阶段的组合数；不同指标的最小值可能来自不同组合，不能拼成一个虚构候选。最后一项保留装配阶段按原始错误信息分组的拒绝数。缺少某个指标表示没有组合到达对应阶段。

## 当前结构评估

`scripts/gap_benchmarks/evaluate_1bna_dna_gap_loo.py` 用 1KX5 suite 库作模板、1BNA 作条目留出。脚本先确认来源哈希及模板不含目标条目，再仅使用缺口两侧锚帧和完整锚重原子坐标生成每条链候选；模型候选通过预设 Watson–Crick 原子接触筛查后，才读取被遮蔽缺失残基坐标用于 RMSD 评价。

2026-11-10 默认阈值复跑留下 4 个符合模板上下文与侧翼配对审计的双链缺口窗口。四个窗口中两条单链均为 `no_solution_found`，所以双链配对候选均为 0。分阶段诊断显示：A 链窗口 4 的 400 个组合全部被起始锚原子位移门限淘汰，最近值为 1.045 Å；B 链窗口 6 的 650 个组合也全部在该门限淘汰，最近值为 0.907 Å。其余部分组合会进入下游阶段，分别因末端锚原子偏移、共享残基连续性或连接几何被拒绝。仅把窗口 6 的 A 链锚连接长度偏差限值从 0.25 Å 探索性提高到 0.30 Å 后，唯一到达装配阶段的组合仍因共享残基原子位移 1.242 Å 超过 0.8 Å 而失败。它说明多个结构门限共同限制刚性 suite 转移；不能靠放宽单一参数宣称成功。该诊断不是科学验收，也不支持结构泛化。敏感性运行不应用来替换正式留出结果或挑选“最佳”结果。

在项目根目录使用已有 Gemmi 的 Python 环境复算。

```bash
python scripts/gap_benchmarks/evaluate_1bna_dna_gap_loo.py --output /tmp/mdna-1bna-dna-gap-loo-check.json
```

回归测试使用 pytest。1BNA 解析需要 Gemmi。
