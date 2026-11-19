# 非典型碱基：打包数据与支持边界

本页区分“仓库里有一个字母模板”和“软件已经按正确化学身份构建并验证”。以下盘点对应本地核查的代码基线 `8fa36bfd9876c23ac76372b2a5b3a7900cc40579`；在完成模板映射、内部键和糖化学修复前，不应把非典型字母示例当作经过验证的结构支持。

## 当前代码路径

`SequenceGenerator` 从 `mdna/atomic/bases/BDNA_{字母}.h5` 读取模板，按 `D{字母}` 写入残基名，并调用 MDTraj 标准键模板；随后为相邻残基添加 `O3'–P`。代码还没有把“碱基字母、CCD 组件身份、糖化学、聚合物连接”拆成独立注册字段。现有 HDF5 模板均来自脱氧糖结构；这些模板不能单独提供 RNA 模板。

| 序列代码 | 打包模板的 HDF5 残基名 | 组件身份/判断 | 当前限制 |
|---|---|---|---|
| A/C/G/T | DA/DC/DG/DT | 标准脱氧核苷酸 | DNA 构建路径；几何和用途仍须按任务验证 |
| U | U | 模板有 `C2'`，没有 `O2'`，符合脱氧尿嘧啶而非标准核糖尿嘧啶 | 不能用来表示 RNA 链；当前代码将其输出为 `DU` |
| D/E | TC1/2PR | tC 与 2-aminopurine 相关模板 | 残基身份和完整键图尚未通过项目级验收 |
| L/M | LHO/BMN | d5SICS/dNaM 相关模板 | 不能据此推断力场参数或实验结构适用范围 |
| B/S/Z/P | 1WA/1W5/JSP/IGU | 组件分别对应 P/Z/S/B | 四个文件的后缀与化学字母发生置换，见下文 |
| CM/GM | DCM/DGM | 甲基化相关模板 | 审计到的 HDF5 拓扑没有内部键记录，不纳入当前可用字母映射 |

HDF5 文件的原子集合和同名 PDB 文件一致，坐标按单位换算比较后 RMSD 约为 0.0005 Å。这只说明两种打包文件互相对应，不证明原始来源、成键、立体化学或软件构建结果正确。

## 已发现的 Hachimoji 标签错配

官方组件命名与本地模板拓扑显示：

| 化学字母 | 应对应组件 | 当前被 SequenceGenerator 读取的文件 | 文件内组件 |
|---|---|---|---|
| B | isoguanine | `BDNA_B.h5` | 1WA（P） |
| S | 1-methylcytosine | `BDNA_S.h5` | 1W5（Z） |
| Z | 6-amino-5-nitropyridin-2-one | `BDNA_Z.h5` | JSP（S） |
| P | 2-amino-imidazotriazinone | `BDNA_P.h5` | IGU（B） |

因此，代码的 B/S/Z/P 输出会使用错误的化学字母身份。互补表 `B↔S`、`Z↔P` 在当前错配下仍可能把 P:Z 和 S:B 这两类配对几何放在一起，但残基身份与序列字母不相符；这不是可接受的修复。下一步应通过中央组件注册表映射化学字母到 CCD 组件和模板来源，并让生成与突变共用该注册表。

## 可参考的数据来源

- Hachimoji 原始研究将四个扩展碱基与配对关系定义为 B:S 和 P:Z：[Hoshika et al., *Science* (2019)](https://doi.org/10.1126/science.aat0971)。
- wwPDB 组件记录： [1WA](https://www.rcsb.org/ligand/1WA)、[1W5](https://www.rcsb.org/ligand/1W5)、[JSP](https://www.rcsb.org/ligand/JSP)、[IGU](https://www.rcsb.org/ligand/IGU)。这些记录提供组件名称、化学式、原子/键和立体化学字段。
- RNA 原型的参考候选为 [4U3R](https://www.rcsb.org/structure/4U3R)，其 8-mer 序列为 UCGUACGA，分辨率 1.70 Å。本地生成的 Coot 理想模型尚未与 4U3R 坐标比较。

## 使用前判断

- 标准 `A/C/G/T` 可调用现有 DNA 构建路径，但这不等同于针对实验结构的科学验收。
- 输入 `U` 不会自动切换到 RNA 糖化学。
- 在组件标签和模板映射修复并完成独立结构检查前，不要把 B/S/Z/P 结果作为正确标注的 Hachimoji 结构交付。
- 修饰核苷酸、混合 DNA–RNA、XNA、替代骨架和模拟参数必须逐项核实；未知组件应明确拒绝，而不是依据单字母继续构建。
