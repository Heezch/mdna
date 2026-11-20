# 开放路径急弯诊断与发布边界

## 本轮可复现结论

固定种子 20261119 生成一条非平面的开放曲线，在其中人为加大三个连续控制点的横向跳动。按等弧长采样，81 点下最大离散转角由 79.45° 降至 40.95°，95 分位由 22.45° 降至 18.39°。同弧长比例位置最大偏移 0.5185 nm，两端不动。弧长从 14.2604 nm 缩至 12.7668 nm。用 48、161、401 点重测，最大转角均下降，数值随采样尺度变化。这是一个合成曲线对照。

```powershell
python -B tests/test_open_path_geometry.py
python -B scripts/gap_benchmarks/evaluate_open_path_bends.py --curve-only --output artifacts/gap_benchmarks/open_path_bends_curve.json --plot artifacts/gap_benchmarks/open_path_bends_curve.png
```

无需原子建模依赖的结果在 `artifacts/gap_benchmarks/open_path_bends_curve.json`。在具备项目完整依赖的环境中，去掉 `--curve-only` 并改写不同的 `--output` 路径，即可运行直线/原始/平滑的同序列标准 DNA 双链原子对照。保留原曲线工件。

## 原子级对照

同一条 48 bp 合成序列在 WSL 中通过公开 `make()` 构建。报告：`artifacts/gap_benchmarks/open_path_bends_atomic.json`。

| 模型 | 最大帧间转角 | O3′–P 距离最小–最大（Å） | 距离筛查失败（共 94 条） | <2 Å 重原子近接触 |
|---|---:|---:|---:|---:|
| 直线 | ≈0° | 0.91–2.31 | 70 | 0 |
| 原始曲线 | 98.62° | 0.23–14.38 | 70 | 183 |
| 调整曲线 | 45.17° | 0.31–7.71 | 67 | 23 |

三组的 94 条预期 O3′–P 拓扑键都存在。上表采用 1.30–2.00 Å 的宽松异常筛查范围；在范围内也不等于满足化学构象标准。三组都未通过筛查。JSON 保留每条连接的两侧键角、每链异常计数和明确的失败原因。调整后，重新按目标序列长度缩放使末端移动 1.497 nm，超过输入空间的 0.90 nm 调整预算；有硬锚点的应用不能把此选项当作固定端点的求解器。

## 入口与限制

- `mdna.regularize_open_path(control_points, max_deviation=...)`：返回端点固定的控制点与路径诊断；预算只约束控制点和同弧长比例处的离散样点，单位和输入一致。采样无法证明连续曲线全程满足偏移预算。没有有效改进时保留输入。
- `mdna.make(..., path_deviation_nm=...)`：显式选择该预处理，限开放路径和显式控制点。原有 `make` 根据碱基对数量重新缩放位置。若输入的绝对位置和端点必须固定，应使用独立预处理入口评估几何，不应把该 `make` 选项用于硬锚定。
- `mdna.structure_qc.measure_open_chain_geometry`：报告逐链 O3′–P 拓扑键、距离、两侧键角与拓扑排除后的近距离重原子接触数；阈值是距离筛查，不是 MolProbity clashscore。Windows 专项单测检查人工拓扑，真实 `MDTraj` 对照已在 WSL 运行。

## 技能知识到代码的映射

| 知识 | 已固化 | 尚待验收 |
|---|---|---|
| `bio-geometric-analysis` 的距离、角度、接触量测 | `structure_qc.py` 区分距离与拓扑键；真实原子对照证实部分改善，但仍有明显坏连接。 | 基于化学与构象类别的角度、二面角、糖环和手性参考；全原子冲突。 |
| `bio-structure-io` 的身份和格式约束 | 既有 DNA/RNA suite 装配和 PDB 往返测试已提供部分证据；保留链序、组件/原子名和拓扑键图。 | 新路径的实际原子构建及 PDB/mmCIF 往返，须分别核对坐标、末端化学及非标准连接。 |
| RNA suite 构象依据 | 既有 `RNAFragmentLibrary` 使用显式帧、片段来源及 atom/link 合同；历史同源 `UCGUACGA` 回放。 | 独立序列/结构家族、RNA 局部异常分类、密度复核。参照 Richardson 等（DOI `10.1261/rna.657708`）。 |

## 建议发布口径

当前工作树以实验分支对待。标准 DNA 双链模板路径仍是既有入口；RNA 是有限库的显式帧单链构建；DNA/RNA 双锚补缺是单残基原型。历史跨条目 DNA 九窗口基准未找到默认门槛下的可接受候选。现有证据不够支持“任意核酸链”，也不足以确认平滑后原子模型稳定。下一步优先按碱基步分解直线对照的连接误差，再验证可保持糖和碱基局部构象的连接重建方案；同时把最终模型的路径偏差与输入端点作为单独的验收量。

## 直线路径上的模板连接诊断

WSL 从项目根目录运行：`python -B scripts/gap_benchmarks/audit_straight_dna_steps.py --output artifacts/gap_benchmarks/straight_dna_step_audit.json`。17 bp 合成序列覆盖全部 16 种有向相邻碱基步；10 种在两条链上都超出宽松范围，旋转直线方向后结论不变。保持模板和步距，扫描统一的扭转角，最优采样角仍有 6 种步超差。该结果支持优先改进连接构象；本轮不调整默认扭转。

## 固定糖和碱基的磷酸旋转对照

从项目根目录在 WSL 运行：`python -B scripts/gap_benchmarks/probe_rigid_phosphate.py --output artifacts/gap_benchmarks/rigid_phosphate_probe.json`。P/OP1/OP2 绕固定 O5′ 刚体旋转，其他原子保持原位。结果显示：16 步直线上全部 32 条连接都可以达到 1.60 Å 诊断靶值，但增加两对近接触；平滑曲线仍有 13 条在固定锚点下不可达。重复序列直线通过当前宽松筛查。报告保留逐条几何和失败原因；该探针未约束键角或扭转，不用于自动修复或发布验收。

### 用近接触和键角筛选取向

在固定锚点下，对可达到 1.60 Å 的连接，分别绕 O5′–O3′ 和 O5′–P 以 15° 步长采样。按近接触数、三个角度诊断窗口的平方超差以及原子位移依次选取，顺向、逆向、顺向各扫描一次。候选保留上一轮取向，避免以局部几何改善为代价增加该步骤的接触数。这是有限网格上的逐组选择，不是全局最优搜索。

复跑：`python -B scripts/gap_benchmarks/probe_rigid_phosphate.py --search --output artifacts/gap_benchmarks/rigid_phosphate_search.json`。

| 模型 | 最近取向 → 新搜索：重原子近接触 | 不可达 | 可达但角度超差 |
|---|---:|---:|---:|
| 16 步直线 | 2 → 0 | 0 | 18/32 |
| 48 bp 直线 | 0 → 0 | 0 | 46/94 |
| 48 bp 原始曲线 | 181 → 141 | 18 | 43/76 |
| 48 bp 平滑曲线 | 25 → 14 | 13 | 42/81 |

四组均未通过包含键角的诊断。平滑曲线的磷酸原子最大位移为 4.02 Å，其他原子位移为零；目前没有位移门槛。诊断窗口不是实验验证标准，放松窗口不能解决实际角度偏差。下一步需要允许有限的骨架自由度，并同时登记角度、近接触和位移边界。

### 限制 O5′ 位移后的可达性

复跑：`python -B scripts/gap_benchmarks/probe_rigid_phosphate.py --search --anchor-budget-A 0.5 --output artifacts/gap_benchmarks/bounded_o5_probe.json`。0.50 Å 预算约束 O5′ 位移。平滑曲线恢复 4/13 条原不可达的键长；余下 6 条超自由平移下界、3 条在所选转轴的连续范围内不可达。新增四条均有键角超差。糖环与碱基保持原位，曲线仍未通过组合诊断。

### 实验坐标的连接角与扭转对照

用相同的 QC 函数度量模型和实验坐标，报告 O3′–P 距离、三个键角与四个有符号连接扭转（沿 5′→3′ 顺序）。先复算带扭转的模型，随后用已有的 WSL dip2 环境运行参考审计：

~~~bash
python -B scripts/gap_benchmarks/probe_rigid_phosphate.py --search --anchor-budget-A 0.5 --output artifacts/gap_benchmarks/bounded_o5_with_torsions.json
conda run -n dip2 python -B scripts/gap_benchmarks/audit_backbone_references.py --probe artifacts/gap_benchmarks/bounded_o5_with_torsions.json --output artifacts/gap_benchmarks/backbone_reference_audit.json
~~~

本地 119D、1WQY、4U3L 分别纳入 22、18、14 条实验连接，三个角度窗口的超差计数均为零。它们是少量实验对照，不能据此为整个 DNA/RNA 化学类别制定门槛。脚本记录坐标来源哈希，拒绝含所需原子的替代构象、缺失坐标及非连续作者编号；不把所有序号相邻记录自动解释为有实验验证的拓扑。

用两份 DNA 对照的 40 条连接做参考，计算四角联合周期 RMS 距离、取最近参考条目。平滑模型调整前后的 P90 为 47.73°→53.39°，16 步直线模型为 34.97°→60.17°。这个指标不按碱基步匹配，也不作为发布门槛。结果说明当前磷酸修复可改善键长和部分碰撞，但没有改善这两组模型的扭转尾部；后续候选应显式控制扭转。

### 以独立 DNA 条目评估扭转评分

~~~bash
python -B scripts/gap_benchmarks/probe_rigid_phosphate.py --search --anchor-budget-A 0.5 --torsion-reference artifacts/gap_benchmarks/backbone_reference_audit.json --output artifacts/gap_benchmarks/torsion_ranked_probe.json
~~~

先复现近接触/键角排序的基线，再按有向碱基步使用 119D 的四角联合周期 RMS 距离排序。没有对应步时不更改该连接。保留每组已有的最大位移上限；每次更新不增加该组的近接触或键角窗口超差。1WQY 只用于事后量测，与 119D 仅两个有向步重合。现有两个条目以前均被用于观察分布，此处是条目隔离实验，不是首次盲测或独立家族验收。

| 模型 | 对 119D：基线→再排序中位偏差 | 对未参与评分的 1WQY：基线→再排序中位偏差 |
|---|---:|---:|
| 覆盖 16 步的直线 | 40.50° → 36.43° | 71.23° → 71.23° |
| 48 bp 直线 | 38.44° → 36.01° | 8.60° → 8.60° |
| 原始曲线 | 47.46° → 41.22° | 25.22° → 28.77° |
| 平滑曲线 | 46.10° → 42.26° | 24.92° → 30.61° |

对 1WQY，16 步对照可评价 16/32 条，另三组各 22/94 条；该覆盖包含未重新评分和不可达的连接。平滑曲线的近接触仍是 12 对、不可达仍为 9 条。对参与评分的条目更接近，但未见跨条目收益。保留实验入口，不接入默认建模。后续要扩大并冻结按结构家族拆分的参考库，同时给急弯引入额外骨架自由度，避免把剩余几何不可达误解为打分问题。

### 共享 O3′ / O5′ 位移预算

复跑：`python -B scripts/gap_benchmarks/probe_rigid_phosphate.py --search --anchor-budget-A 0.5 --dual-anchor --output artifacts/gap_benchmarks/bounded_dual_anchor_probe.json`。在 O5′ 单轴方案失败后，允许 O3′ 绕 C4′–C3′ 转动；两端位移之和仍限制在 0.50 Å。无扭转评分的匹配对照中，平滑曲线剩余不可达从 9 降至 7、近接触从 12 降至 11。新增的两条都仅移动了 O3′，仍有角度超差。无转轴搜索失败等同无解的假设；仅自由位移下界超过预算的 6 条可按当前预算排除。

### 固定端点的小幅帧平移对照

在 WSL 运行 `OPENBLAS_NUM_THREADS=1 python -B scripts/gap_benchmarks/probe_frame_translation.py --output artifacts/gap_benchmarks/frame_translation_probe.json`。每个内帧相对已构建路径的位移最多 0.05 nm，两端固定。平滑模型平移后不可达连接从 7 增至 9，近接触从 11 增至 23；使用四分之一位移时仍有 9 条不可达和 13 对近接触。两条曲线所有非零候选均未优于原对照。距离下界仅用于提出候选；实际连接、键角和近接触决定是否接受。

### 固定中心重新估计帧朝向

复跑：`OPENBLAS_NUM_THREADS=1 python -B scripts/gap_benchmarks/probe_frame_orientation.py --output artifacts/gap_benchmarks/frame_orientation_probe.json`。平滑模型在两帧跨度、15° 旋转上限下，不可达 7→5、近接触 11→7、可达但角度超差 48→46，全部路径帧中心保持原位。30° 候选有更少的近接触（3 对），仍有 6 条不可达。所有候选都未通过组合诊断；原始曲线没有三项同时不恶化的候选。已构建路径的初始缩放偏差须单独评估。
## 上游与开发快照对照

在项目根目录运行 `python -B scripts/gap_benchmarks/compare_open_path_revisions.py --output artifacts/gap_benchmarks/open_path_revision_comparison.json`。脚本默认独立导入上游 `8fa36bf` 和 fork `afc7494` 两个 Git 快照，用同一固定种子、序列、控制点和原子检查复算。

直线和原始急弯的旧调用输出逐原子一致。显式启用平滑后，急弯转角 98.619° → 45.166°，近接触 183 → 23，超距骨架连接 70 → 67。生成后的末端相对上游原始急弯偏移 1.497 nm，超过 0.90 nm 输入路径预算。结果同时记录生成器的长度重缩放效应。

## 保留开放路径端点

`mdna.make(sequence=seq, control_points=points, preserve_path_endpoints=True)` 按输入路径原尺度分布帧。若需限制步距，传入 `max_step_deviation_nm`，逐帧直线距离偏离 0.34 nm 超过该值时会报错。该检查仅针对帧位置，骨架质量仍需独立检查。

复算使用 `python -B scripts/gap_benchmarks/evaluate_open_path_bends.py --preserve-endpoints --output artifacts/gap_benchmarks/open_path_fixed_endpoints.json`。固定种子 48 bp 对照的两端误差均为零。平滑减少了近接触，但导致步距缩短，且 68/94 条连接超出骨架距离窗口。

同时指定平滑与步距限值时，程序从满足两组约束的候选中选择最终帧最大转角较小的曲线，再复查实际帧。对照命令可额外加 `--max-step-deviation-nm 0.065`。在全局平滑对照之后加入单个控制点的局部调整，仍保持零端点误差，最大步距偏差为 0.06485 nm，最终转角由 93.31° 降到 79.41°，近接触由 228 降到 175。仍有 67/94 条骨架连接超出距离窗口。

四条新种子的留出曲线均在同一策略下保持端点并满足 0.065 nm 步距预算。下表以各自固定端点、未平滑模型为基线。

| 种子 | 最大帧间转角 | 重原子近接触 | 超距 O3′–P（共 94 条） |
|---|---:|---:|---:|
| 20261120 | 97.19° → 67.69° | 234 → 152 | 75 → 70 |
| 20261121 | 89.07° → 58.41° | 246 → 158 | 74 → 68 |
| 20261122 | 100.37° → 59.45° | 282 → 111 | 78 → 75 |
| 20261123 | 94.75° → 87.27° | 255 → 226 | 69 → 71 |

用 `--seed <seed>` 复算相应曲线。四例均未通过综合原子筛查。第四例的长连接增加，说明减小转角和近接触仍需结合骨架距离判断。

### 用骨架距离筛选路径候选

实验脚本 `screen_open_path_backbone.py` 比较六档位移预算，在端点、步距、短连接、长连接及近接触均过比较筛查的候选中，选取最大转角较小的一条。此前长连接恶化的 20261123 在 0.15 nm 搜索预算下，长连接从 33 降到 32，短连接从 36 降到 35，近接触从 255 降到 217。试验四例仍有 67–75 条超距连接，全部未通过综合原子筛查。

以冻结条件复查新种子 20261124 和 20261125，两者均无候选通过比较筛查。前者能满足步距但骨架条件失败，后者的六档搜索均未构建出步距合格模型。复算参数用 `--seed 20261124 20261125`，逐项记录见 `artifacts/gap_benchmarks/open_path_backbone_holdout.json`。

### 在原曲线上重新分配帧位置

实验脚本 `probe_open_path_resampling.py` 在原曲线的有序网格上搜索满足端点、步距与配对帧位移约束的帧位置。用 `--seed 20261125 --grid-size 1001` 或 `--grid-size 2001` 并指定 `--output` 复算。2001 点搜索将最大转角从 77.40° 降至 72.33°，最大步距偏差从 0.14874 nm 降至 0.06445 nm，两端误差为零。重原子近接触从 360 降至 348，长 O3′–P 连接从 36 增至 37。两组网格试验均未通过综合原子筛查。该搜索只在基准脚本中使用。
