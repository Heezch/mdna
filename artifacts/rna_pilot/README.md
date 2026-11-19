# 标准 dsRNA 构建原型

## 产物

- `ideal_A_dsRNA_UCGUACGA.pdb`：Coot 生成的理想 A-form 双链 RNA 起始模型。
- `reference/4U3R.cif`：从 RCSB PDB 下载的实验 mmCIF 坐标，SHA-256 记录在审计清单中。
- `reference/4U3R.audit.json`：链、原子、糖化学、端基/内部连接原子状态、连接距离和 Coot 模型比较结果。
- `reference/ccd/{A,C,G,U}.cif`：RCSB CCD 标准 RNA 组件定义原文，包含原子、键和理想坐标。
- `reference/ccd/manifest.json`：CCD 来源、下载日期、文件哈希、原子/键计数和初步字段核查。
- `reference/ccd/audit.json`：Gemmi 逐组件核查结果，含解引号后的原子名、坐标完整性、键端点和哈希。
- `scripts/rna_pilot/audit_ccd_components.py`：只读复算 CCD 组件审计的命令行脚本。
- `scripts/rna_pilot/evaluate_ccd_to_polymer_fit.py`：将 CCD 理想单体按碱基环拟合至实验聚合物，并检查糖/骨架偏差及相邻连接距离。
- `reference/ccd_to_polymer_fit.json`：上述拟合审计的逐残基、逐连接结果和输入 SHA-256。
- `scripts/rna_pilot/extract_rna_suite_library.py`：从实验 mmCIF 提取内部双核苷酸 RNA suite，添加内部态 CCD 键图和 frame-relative 坐标。
- `reference/rna_suite_library.json`：60 个内部 suite 坐标、化学图、局部帧变换、结构与化学来源哈希及覆盖/构象差异汇总。
- `scripts/rna_pilot/audit_terminal_geometry.py`：比较各参考 RNA 链端与同碱基内部残基的局部键长/键角，不改写原坐标。
- `reference/terminal_geometry_audit.json`：六个结构中 12 个 5′ 端和 12 个 3′ 端的局部几何比较、输入哈希和适用限制。
- `reference/terminal_geometry_audit.json` 的当前 SHA-256：`3D8FA70531DB80240F2DAEF77130D833CF1262AE8848BDFCF9617013482ED873`。相同 WSL 脚本重复运行后哈希不变。
- `scripts/rna_pilot/extract_rna_terminal_suites.py`：提取带 5′-OH 或 3′-OH 状态的边界二核苷酸 suite，按 CCD 派生态核验重原子清单与键图。
- `reference/rna_terminal_suite_library.json`：24 个实验端部 suite；现有参考只覆盖 5′ `UC` 与 3′ `GA`，另含坐标、键图、输入文件哈希和状态来源。
- 端部 suite 库 SHA-256：`8BF948C69E11ADBDFD60F88E8D1843CF9F872610FA8832E06BA3DB587C6D4FC9`。
- `scripts/rna_pilot/evaluate_rna_terminal_suite_loo.py`：按 PDB 条目留一，把其余条目的同类候选映射到留出结构上游帧，并报告所有候选的帧残差和原子 RMSD。
- `reference/terminal_suite_loo_evaluation.json`：24 个留出案例、240 个跨条目候选配对的逐候选结果。这是同序列的条件性迁移比较，不是盲测。
- 留一评估 JSON SHA-256：`328A78DE0F1D3606789C6ED443032E02A28F86E86A63EC280239F3EE3F17F3BE`。
- `scripts/rna_pilot/audit_4u3r.py`：用 Gemmi 与 NumPy 复算审计清单的脚本。
- RNA 显式帧构建 API 指南：`docs/guide/rna-path-building.md`。

## 复现

在项目根目录使用 PowerShell 执行：

```powershell
.\scripts\rna_pilot\run_with_wincoot.ps1 --sequence UCGUACGA
```

脚本通过 WinCoot Python API `ideal_nucleic_acid("RNA", "A", 0, sequence)` 生成标准 RNA 双链，并将 Coot 数据与备份目录重定向到本项目的 `artifacts/rna_pilot/.coot-home/`。使用前设置 `MDNA_WINCOOT_ROOT` 为本机 WinCoot 安装目录，并确认所用安装含标准 A/C/G/U 单体字典。

## 本次观察

- Coot 返回 2 条链、16 个残基、340 个原子。
- 每个核苷酸均含 `O2'`；Gemmi 独立读取 PDB 得到两条 8 nt 链，序列均为 `UCGUACGA`。该序列自身是反向互补序列。
- 两条链的相邻残基 `O3'–P` 距离范围为 1.6039–1.6050 Å。
- PDB SHA-256：`4f2b08e521ecd2f015783a2e146ce06434c2d7feb267b3cfb283a426273e98bb`。

这些结果检查了文件可解析性、原子存在和局部连接距离。生成器与距离来源不是独立的能量或实验质量评估，不能据此宣称符合 4U3R、RNA 力场或 OneDep 标准。

## RNA 端基局部几何比较

复算命令：

```bash
python scripts/rna_pilot/audit_terminal_geometry.py
```

脚本比较每个参考链的末端核苷酸与同链最近的同碱基内部核苷酸。报告指标包括 `C5'–O5'`、`C4'–C5'–O5'`、`C3'–O3'` 和 `C4'–C3'–O3'`。六个结构中每端各有 12 个比较。结果显示这些局部键长和键角差异较小；该比较没有评估末端氢坐标、骨架扭转、跨结构坐标转移或生成链的连接质量，因此不能单独支持端基坐标复用。

## 端部 suite 与留一条目迁移检查

复算命令：

```bash
python scripts/rna_pilot/extract_rna_terminal_suites.py
python scripts/rna_pilot/evaluate_rna_terminal_suite_loo.py
```

端部 suite 提取器保留 CCD 派生的 5′-OH/3′-OH 重原子化学图。24 条记录只覆盖 5′ `UC` 与 3′ `GA`，各 12 条。按条目留一后，每个留出案例有 10 个来自其他条目的同类候选。评估把候选放入留出结构的上游帧，以留出结构下游帧计算姿态残差；不按隐藏原子坐标挑选候选。5′ 候选的全重原子 RMSD 中位数/P95/最大值为 0.258/0.451/0.496 Å；3′ 为 0.433/0.708/0.768 Å。结果只描述该同序列结构组中的条件性坐标迁移。六个条目属于同一 8-mer，目标下游姿态也取自参考结构，因此结果不能视为盲测或推广到其他序列。

## 端部与内部 suite 装配原型

`RNAFragmentLibrary.from_json("artifacts/rna_pilot/reference/rna_suite_library.json", terminal_path="artifacts/rna_pilot/reference/rna_terminal_suite_library.json")` 将内部库和端部库读入一个校验后的库。路径调用须显式设置 `include_terminal_suites=True`；这会要求首个 suite 是 5′-OH 态、末个 suite 是 3′-OH 态，并要求中间步骤使用内部 suite。未知序列或端部覆盖缺失时，程序拒绝构建。

`assemble_suite_path()` 仅在来源键图、共享重原子位移和来源键长变化满足调用方阈值时合并片段。`RNAFragmentAssembly.to_mdtraj()` 输出已观测重原子和键图；端部状态记录为 5′-OH/3′-OH，端部氢未生成。4U3L chain A 连续 `UCGUACGA` 的同源来源回放得到 8 残基对象，路径共享原子位移和键长变化限值为 `1e-5 Å`，用于确认端部/内部 suite 的程序组合和化学边界处理。测试从同一条目提取片段并用这些片段构造目标帧，所以不是独立验证或盲测。该结果不证明其他序列可构建，不覆盖 12 种缺失二核苷酸，也不验证双链配对、冲突、密度或缺口闭合。

## 4U3R 参考结构核查

RCSB 4U3R 为 1.701 Å X-ray 结构，记录标题标注与 Co(II) 共结晶。Gemmi 读取本地 mmCIF 后得到 1 个模型、2 条 RNA 链；每条链均为 8 nt `UCGUACGA`，每个残基都有 `O2'`。两条 RNA 链合计 334 个重原子；全结构有 424 个重原子，其中另含 87 个水和 3 个 `CO` 组分。两条链的 7 个链内 `O3'–P` 距离分别在 1.6044–1.6079 Å 与 1.6061–1.6089 Å 之间。

对照 Coot 理想 A-form 模型时，先用链 A 的共同糖/骨架重原子作最小二乘刚体叠合，再计算共同重原子 RMSD。93 个拟合原子 RMSD 为 1.645 Å；167 个共同重原子的 RMSD 为 1.688 Å。该比较只说明此理想化模型与这条实验链的坐标差异，不能归因于某个单独参数，也不能当作能量、密度拟合或一般 RNA 质量判断。

复算命令：

```bash
python scripts/rna_pilot/audit_4u3r.py
```

该命令需要本机 WSL `dip2` 环境中的 Gemmi 和 NumPy。脚本读取 `reference/4U3R.cif` 与 Coot PDB，不改写源结构；命令会更新 `reference/4U3R.audit.json`。

## RNA 组件字典输入

A/C/G/U 的 CCD CIF 从 [RCSB 单组件下载服务](https://www.rcsb.org/docs/programmatic-access/file-download-services) 获取，原文件和 SHA-256 保存在 `reference/ccd/`。Gemmi 0.7.5 审计确认 A/C/G/U 分别含 37/35/38/34 个原子、39/36/40/35 条内部键；键端点可解析，理想坐标完整且有限，四种组件都含核糖 `O2'`。复算命令：

```bash
python scripts/rna_pilot/audit_ccd_components.py --output artifacts/rna_pilot/reference/ccd/audit.json
```

这一结果只核对组件字典记录，不代表聚合物坐标模板或核酸结构构建已通过验证。CCD 本身说明组件化学及理想坐标，不提供完整的 RNA 螺旋/链构象；详情见 [wwPDB CCD 说明](https://www.wwpdb.org/data/ccd)。

4U3R 的更新审计还记录了聚合物端基：两条链的首个 U 均含 `HO5'` 且不含 `P`；末端 A 均含 `HO3'`；内部残基含 `P`，不含 `OP3` 或末端羟基氢。wwPDB 将 `OP3` 归为核酸聚合物连接时的离去原子。[wwPDB processing procedures](https://www.wwpdb.org/documentation/procedure) 因而化学注册表需要区分 CCD 单体定义、链内连接态和两种端基态。详细原子名和标志见 `reference/4U3R.audit.json`。

## CCD 单体坐标与实验聚合物拟合

复算命令：

```bash
python scripts/rna_pilot/evaluate_ccd_to_polymer_fit.py
```

脚本将 CCD 理想碱基环以 proper-rotation Kabsch 拟合至六个参考条目中 96 个 RNA 残基的观测碱基环，再比较其余重原子坐标。该拟合的碱基环 RMSD 均值为 0.047 Å；糖和骨架 RMSD 均值分别为 2.854 Å 和 5.147 Å。将每个单体独立拟合后，84 个相邻 `O3'–P` 距离的均值为 7.142 Å；实验坐标对应距离的均值为 1.606 Å。逐项值、分位数与源文件哈希见 `reference/ccd_to_polymer_fit.json`。

这些结果表明，按碱基环独立刚性放置 CCD 理想单体，不能保持实验聚合物的糖/骨架构象或连接几何。该审计没有优化糖环、核苷键或磷酸二酯扭转，也没有测试内坐标重建；因此不能据此否定使用 CCD 化学定义配合有来源的聚合物构象库。当前 RNA 坐标构建应优先使用实验聚合物片段或经过验证的核酸套件构象库，并对片段端点闭合做显式检查。

## 内部 RNA suite 片段库

复算命令：

```bash
python scripts/rna_pilot/extract_rna_suite_library.py
```

脚本核对每个来源坐标文件与归档 manifest 的 SHA-256，只提取链内两端残基均为内部磷酸二酯态的相邻二核苷酸。脚本使用 `mdna.geometry.ReferenceBase` 的 Tsukuba 原子锚点定义建立局部帧，再将实验重原子坐标变换到上游局部帧。原子键图来自 CCD 内部态，另加上游 `O3'` 到下游 `P` 的连接边。suite 同时记录坐标、帧间刚体变换、源条目/链/残基、参考坐标哈希和化学注册状态哈希。片段不含氢原子坐标。

六条 8-mer 参考共提供 60 个内部 suite，覆盖 AC、CG、GU、UA 4 种二核苷酸序列。16 种标准 A/C/G/U 二核苷酸中有 12 种未观测。按 Tsukuba 上游帧直接比较同类型片段的全重原子 RMSD，中位数为 0.351–0.468 Å，各类型最大值为 0.619–1.176 Å。60 个 suite 内 `O3'–P` 距离为 1.5979–1.6146 Å，均值 1.6062 Å。

该内部库仅覆盖一条自互补 8-mer 的内部结构片段；该库不含末端片段、未配对环或另外 12 种二核苷酸。独立的端部库现已覆盖这条序列的 5′ `UC` 和 3′ `GA`，两库能组合回放 4U3L chain A。回放复用同一来源，不构成盲测；现有数据不能支持通用 RNA 路径或缺口闭合结论。

## 适用范围与下一步

`ideal_A_dsRNA_UCGUACGA.pdb` 是 Coot 后端生成的理想化直线 A-form RNA 起始模型。该模型并非 MDNA 内部生成器的输出。当前尚无任意空间路径构建、双链 RNA 生成或缺口闭合，也没有与 4U3R 电子密度或 wwPDB 验证报告比较。单体拟合审计显示，CCD 理想坐标独立刚性放置后无法保持糖/骨架与 `O3'–P` 几何。内部 RNA suite 库现含 60 个有来源的实验片段，但只覆盖 4/16 种标准二核苷酸；端部库只覆盖 5′ `UC` 和 3′ `GA`。当前下一步是扩展有独立来源的序列/端部构象覆盖，再开发 RNA 路径 API 和来源隔离的几何基准。
