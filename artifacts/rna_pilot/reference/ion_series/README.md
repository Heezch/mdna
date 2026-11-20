# 同序列 RNA 离子条件参考组

## 用途与范围

此目录提供同一条 8 nt 双链 RNA（每条链均为 `UCGUACGA`）在不同金属离子条件下的 RCSB 结构坐标。复算官方验证报告和 2Fo-Fc / Fo-Fc 地图系数时，请按 `manifest.json` 的来源 URL 补充对应输入。该参考组用于检查序列、核糖化学、共价连续性、几何质量和参考结构条件效应。

## 来源和复现

- 数据源：RCSB PDB / wwPDB 验证报告归档；下载 URL、字节数和 SHA-256 见 `manifest.json`。
- 获取日期：2026-10-27 UTC。
- 条目共同文献：[The X-ray Structures of Six Octameric RNA Duplexes in the Presence of Different Di- and Trivalent Cations](https://doi.org/10.3390/ijms17070988)。条目/离子条件可从 [4U3L](https://www.rcsb.org/structure/4U3L)、[4U3O](https://www.rcsb.org/structure/4U3O)、[4U3P](https://www.rcsb.org/structure/4U3P)、[4U3R](https://www.rcsb.org/structure/4U3R)、[4U47](https://www.rcsb.org/structure/4U47)、[4U78](https://www.rcsb.org/structure/4U78) 查看。
- 审计命令（WSL `dip2`）：`python scripts/rna_pilot/audit_ion_series.py`。
- 论文中 X3DNA rise/twist 表格的机器可读转录：`x3dna_published_parameters.json`，逐项保留 DOI、表号、算法和统计定义。
- 可复现汇总：`audit.json`。脚本用 Gemmi 解析坐标和 map coefficient CIF，用 PyMuPDF 提取报告文字；按 3.0 samples/Å 将官方相位/振幅系数转换为有限的密度网格。代码仅总结官方验证报告并确认地图可构建；未对新的 MDNA 模型计算密度拟合分数。
- 探索性坐标指标：运行 `python scripts/rna_pilot/compare_ion_series_geometry.py` 生成 `geometry_comparison.json`。它显式记录反平行配对和碱基对中心定义，但 PCA 轴投影、中心步长、局部弦向扭转均不等同 X3DNA helical rise/twist，因此不作为验收阈值。
- 该脚本在六个条目中均确认，坐标上最近的互补碱基索引与预期反平行配对（chain B 逆序）相符；此项是链配对映射核验，不是氢键能量或唯一配对证明。

## 初步结果

六个条目均有两条 8 nt RNA 链，序列均为 `UCGUACGA`；审计到的每个残基均含 `O2'`。各验证报告均未记录 RNA pucker outlier、链断裂或 RSRZ outlier；RNA 键长和键角均无报告离群。4U3L 与 4U3P 各有一个 RNA backbone outlier，均为 chain B 的 G7。其余四个条目未记录 RNA backbone outlier。该单点标记需保留并结合原子级数据判断，不能把整条参考结构简单判为失败。

| PDB | 离子条件 | 分辨率（Å） | RNA backbone outlier | clashscore | 链内 O3′–P 距离范围（Å） |
|---|---|---:|---|---:|---:|
| 4U3L | Ca(II)，共结晶 | 1.482 | B:G7（1） | 6 | 1.5979–1.6100 |
| 4U3O | Mn(II)，浸泡 | 1.800 | 无 | 4 | 1.5946–1.6131 |
| 4U3P | Sr(II)，共结晶 | 1.866 | B:G7（1） | 4 | 1.5990–1.6146 |
| 4U3R | Co(II)，共结晶 | 1.701 | 无 | 0 | 1.6044–1.6089 |
| 4U47 | Tb(III)，浸泡 | 1.952 | 无 | 2 | 1.6026–1.6125 |
| 4U78 | Cu(II)，浸泡 | 1.501 | 无 | 0 | 1.6029–1.6085 |

该论文用 X3DNA 计算了局部碱基对 helical rise/twist，并报告“均值（标准差）”。将其作为本参考组的**标准定义基线**；机器可读值见 `x3dna_published_parameters.json`：

| PDB | 条件 | X3DNA helical rise（Å） | X3DNA helical twist（°） |
|---|---|---:|---:|
| 4U3L | Ca(II) | 2.40 (0.57) | 36.0 (2.7) |
| 4U3O | Mn(II) | 2.59 (0.47) | 35.0 (2.4) |
| 4U3P | Sr(II) | 2.58 (0.36) | 35.3 (1.8) |
| 4U3R | Co(II) | 2.41 (0.55) | 35.6 (2.9) |
| 4U47 | Tb(III) | 2.56 (0.42) | 35.7 (1.7) |
| 4U78 | Cu(II) | 2.43 (0.54) | 35.8 (3.1) |

文献还说明 4U3O、4U78、4U47 的晶体先在 Ca(II) 条件下生长、随后浸泡其他离子；六条目不是六种完全独立的结晶环境。对金属离子效应的解释需结合这种实验谱系。

脚本 `scripts/rna_pilot/compare_ion_series_geometry.py` 另外输出基于碱基环质心、PCA 全局轴和局部弦向量的探索性指标。这些定义并非 X3DNA helical parameters，数值不可直接与上表比较，也暂不作为建模验收门；脚本的目的只是显式记录当前的几何假设并定位后续实现需求。

## 证据边界

- 六个 mmCIF、PDF 和压缩系数文件均与下载清单 SHA-256 相符；每个 2Fo-Fc 和 Fo-Fc 系数文件均成功解析，并生成有限密度网格。
- 独立从坐标计算相邻核苷酸 `O3'` 到下一残基 `P` 的距离；六个条目的总体范围为 1.5946–1.6146 Å，与各自报告“无键长 outlier”相符。该范围是局部共价几何核查，不是双螺旋或模型质量评分。
- 验证报告里的 RSRZ、torsion、pucker 和 clashscore 是 wwPDB 对存档模型的检查结果，不是对 MDNA 输出的验证。
- 网格生成成功只证明地图系数可读、变换流程可运行；还没有把 MDNA 候选模型放入地图计算 RSCC/RSR 或做差值图分析。
- 这些结构是晶体与离子条件下的观测。路径构建参数应区分几何起始先验与条件依赖的实验构象，不能直接从一个离子条件外推到所有环境。
