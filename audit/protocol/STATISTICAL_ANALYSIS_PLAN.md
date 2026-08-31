# V14 统计分析计划

## 1. 文件状态与适用范围

本文件用于在读取 V14 新增实验指标前固定统计口径。它只规定分析方法和结果解释边界，不记录 Condition0、B-lite-FT 或 FreeUV V1.2 的任何质量结果。

- 计划版本：`V14-SAP-1.5`
- 计划状态：`AWAITING_INPUT_HASH_BINDING`
- 适用数据：FaceScape 评价集（D1）和 REALY 评价集（D2）
- 适用新增比较：Condition0、B-lite-FT、FreeUV V1.2
- 既有 V13 结果：作为已观察结果保留，不追溯性改写为本计划下的新增确认性分析
- 冻结要求：所有标记为 `PENDING_BINDING` 的字段完成绑定并生成本文件外部 SHA-256 后，方可读取新增指标汇总

冻结后不得根据结果改变指标、样本范围、聚合顺序、检验方向、比较族、停止条件或图表选择规则。若确需修订，必须另建版本并将修订后的分析明确标为探索性。

## 2. 方法与对照定义

| 标识 | 定义 | 在本计划中的作用 |
|---|---|---|
| Full | FrugalFace3D-Lite；使用 B-lite 初始纹理和纹理特征，并接收坐标—法向、表情及任务掩码条件；确认性比较使用与两个对照相同 CUDA 环境重新训练的五个模型 | 待评价方法 |
| B-lite | 当前冻结的轻量 UV 补全基线，不进行 FaceScape 配对微调 | 既有直接基线与资源参照 |
| Condition0 | 与 Full 使用相同残差分支结构、参数量、初始化规则、训练数据、训练步数、损失及优化器，但坐标、法向和表情条件固定为零；保留 B-lite 初始纹理、纹理特征以及定义任务域所需的可见性和规范支持掩码 | 同容量无结构条件对照，用于判断结构条件是否提供增量作用 |
| B-lite-FT | 从同一冻结 B-lite 权重开始，在与 Full 相同的 FaceScape 身份划分、配对样本、步数、种子和模型选择规则下进行微调，不增加坐标、法向或表情输入 | 数据和额外训练对照，用于判断相对原始 B-lite 的变化能否由同数据微调解释 |
| FreeUV-native | FreeUV V1.2 同一次前向得到的原始公共 64×64 端点，尚未逐值复制源视图已观测 UV | 仅用于输出协议分析 |
| FreeUV-conserved | 在 FreeUV-native 的规范支持域内逐值复制源视图已观测 UV；与 native 端点的隐藏区域完全相同 | 原生人脸 UV 方法的主要统一协议比较端点 |

Condition0 和 B-lite-FT 的训练实现、损失映射、checkpoint 选择和终态必须由独立运行合同固定并绑定哈希。若实现不满足上表定义，相应比较不得进入确认性统计。

## 3. 输入与哈希绑定

`PENDING_BINDING` 必须在任何新增指标汇总被打开前替换为真实 SHA-256。路径仅用于本地定位，正式绑定以文件内容哈希为准。

| 输入角色 | 预期对象 | SHA-256 |
|---|---|---|
| 本计划冻结副本 | 外部只读副本 | `PENDING_BINDING` |
| 统计程序 | V14 逐样本聚合与检验入口 | `PENDING_BINDING` |
| Full seed 2026080447 | RTX 4090/CUDA 同设备重训 checkpoint 与运行终态 | `PENDING_BINDING` |
| Full seed 2026080448 | RTX 4090/CUDA 同设备重训 checkpoint 与运行终态 | `PENDING_BINDING` |
| Full seed 2026080449 | RTX 4090/CUDA 同设备重训 checkpoint 与运行终态 | `PENDING_BINDING` |
| Full seed 2026080450 | RTX 4090/CUDA 同设备重训 checkpoint 与运行终态 | `PENDING_BINDING` |
| Full seed 2026080451 | RTX 4090/CUDA 同设备重训 checkpoint 与运行终态 | `PENDING_BINDING` |
| B-lite | checkpoint 与状态字典清单 | `PENDING_BINDING` |
| Condition0 seed 2026080447 | checkpoint 与运行终态 | `PENDING_BINDING` |
| Condition0 seed 2026080448 | checkpoint 与运行终态 | `PENDING_BINDING` |
| Condition0 seed 2026080449 | checkpoint 与运行终态 | `PENDING_BINDING` |
| Condition0 seed 2026080450 | checkpoint 与运行终态 | `PENDING_BINDING` |
| Condition0 seed 2026080451 | checkpoint 与运行终态 | `PENDING_BINDING` |
| B-lite-FT seed 2026080447 | checkpoint 与运行终态 | `PENDING_BINDING` |
| B-lite-FT seed 2026080448 | checkpoint 与运行终态 | `PENDING_BINDING` |
| B-lite-FT seed 2026080449 | checkpoint 与运行终态 | `PENDING_BINDING` |
| B-lite-FT seed 2026080450 | checkpoint 与运行终态 | `PENDING_BINDING` |
| B-lite-FT seed 2026080451 | checkpoint 与运行终态 | `PENDING_BINDING` |
| FreeUV V1.2 私有结果包 | `W5B49N_FREEUV_D1D2_20260820V12_PRIVATE_RESULTS_V1_2.tar.gz` | `25e26864d5cf6429171faf76c3575944a7f860315e3835b32adbe7f5710e418c` |
| FreeUV D1 公共 64×64 聚合数组 | 包内 `FREEUV_D1_COMMON64_OUTPUTS.npz` | `3f25312879e395676aeac32c5ee3a1d1b08bb3db3703bb34ac73a28a0ee02ff0` |
| FreeUV D2 公共 64×64 聚合数组 | 包内 `FREEUV_D2_COMMON64_OUTPUTS.npz` | `60d6ad02174cbdae1ea466e5a94e1f7b456fc7537aa11c91d10235f05a67e430` |
| FreeUV 安全样本映射 | V1.2 运行绑定所用映射 | `ba64333dc39daafdfb45a13363705fb4cf8e716d4cc6901c352e744b78dcbeb2` |
| FreeUV 渲染清单 | 包内 `RENDER_MANIFEST.jsonl` | `f701dd3931c995de947e732838ad6acdaf2ae1665aeb633cc746df08b02b4357` |
| FreeUV 目标帧清单 | 包内 `TARGET_FRAME_MANIFEST.jsonl` | `63008d05585994d0ec2e7830e8739a61ce4b38897b2bb3c495a19dd9b2e0c616` |
| D1 评价缓存清单及数组 | 源/目标局部 UV、可见性、规范支持域 | `PENDING_BINDING` |
| D2 评价缓存清单及数组 | 源/目标局部 UV、可见性、规范支持域 | `PENDING_BINDING` |
| D1 有向配对清单 | FaceScape 固定配对 | `PENDING_BINDING` |
| D2 有向配对清单 | REALY 固定配对 | `PENDING_BINDING` |
| 统一目标视角渲染器 | 固定几何渲染入口 | `PENDING_BINDING` |
| Full/Condition0/B-lite-FT 渲染清单 | 与 FreeUV 共享目标帧和裁剪的渲染输出 | `PENDING_BINDING` |
| LPIPS Linux 资格终态 | 新输出根、当前 CPU 环境、评估器导出与固定探针；CUDA 调用数为 0 | `PENDING_BINDING` |
| LPIPS 运行清单 | LPIPS-Alex v0.1、权重与运行环境 | `PENDING_BINDING` |
| SFace Linux 资格终态 | 新输出根、当前 CPU 环境、检测器、识别器与固定探针；CUDA 调用数为 0 | `PENDING_BINDING` |
| SFace 检测器 | YuNet 模型与运行清单 | `PENDING_BINDING` |
| SFace 识别器 | OpenCV Zoo SFace 模型与运行清单 | `PENDING_BINDING` |

SFace 首次资格检查在 OpenCV 4.7.0 中触发 YuNet `getLayerData id=-1` 兼容性错误。该检查发生在任何真实图像或实验指标读取之前，终态记录为 `real_image_reads=0` 和 `metric_rows=0`，原失败终态永久保留且不转换为通过状态。在保持 YuNet/SFace 模型、合成探针、CPU 单线程、关闭 OpenCL、零 CUDA 调用及其他评价条件不变的前提下，正式 SFace 环境预先修正为 `opencv-python-headless==4.10.0.84`。修正后的独立环境已通过相同合成探针并输出预期的 128 维特征，正式资格检查必须写入新的输出根。通过资格检查后不得再次改变 SFace 运行环境。

Full、Condition0 和 B-lite-FT 均须在同一 RTX 4090/CUDA 软件环境、相同训练身份划分和相同训练预算下完成五次独立训练，共 15 个训练单元。每个训练单元固定为 512 步，总计 7,680 步。15 个终态必须共同绑定 CUDA 环境、训练划分和训练预算清单，并分别绑定各自 checkpoint。历史 Full `2026080447`—`2026080449` 的 MPS 训练结果不进入 V14 确认性比较，仅可作为明确标注的历史参照。

## 4. 数据范围与统计单位

### 4.1 固定样本范围

| 数据集 | 身份数 | 有向配对 | 主要 MAE 可评价配对 |
|---|---:|---:|---:|
| D1 FaceScape | 20 | 160 | 148；其余 12 个固定记为共同隐藏区域为空 |
| D2 REALY | 100 | 1,200 | 1,200 |

统计推断单位始终为身份。视角对、RGB 通道、texel 和随机种子均不得作为独立推断样本。

LPIPS 和 SFace 的主要统计沿用 D1 的 148 个共同隐藏区域非空配对及 D2 的 1,200 个配对，以保证三类质量指标使用相同的预定样本范围。D1 其余 12 个配对可以形成完整性记录，但不进入主要比较。

### 4.2 评价区域

共同隐藏区域固定为

\[
H_{v}=M_{canon}\odot(1-V_{v}^{src})\odot V_{v}^{tgt}.
\]

主要 UV 指标只在 \(H_v\) 上计算。全部目标可见区域

\[
A_v=M_{canon}\odot V_{v}^{tgt}
\]

仅用于协议分析，不替代共同隐藏区域主指标。

## 5. 指标定义

### 5.1 主要指标

共同隐藏区域 RGB MAE，取值归一化到 \([0,1]\)，越低越好：

\[
e_v(T)=\frac{\sum |T-T_v^{tgt}|\odot H_v}{3\sum H_v}.
\]

主要方法端点均使用已观测 UV 逐值保留后的输出。FreeUV 的主要端点固定为 `FreeUV-conserved`；`FreeUV-native` 不作为另一独立方法加入排名。

### 5.2 关键次要指标

1. **配对目标视角 LPIPS**：LPIPS-Alex v0.1，越低越好。方法渲染与目标帧使用同一公共人脸掩码；掩码外置为 0.5；双线性缩放到 128×128，`align_corners=False`；输入映射至 \([-1,1]\)。
2. **SFace 身份余弦**：源图与目标视角方法渲染的 SFace 余弦相似度，越高越好。源图检测一次、目标图检测一次，方法渲染共享目标对齐且不重新检测。人脸嵌入不得保存。

### 5.3 协议与实现检查

- FreeUV-native 与 FreeUV-conserved 在共同隐藏区域的最大绝对差必须为零。
- FreeUV-conserved 在源视图已观测区域与输入采样纹理的最大绝对差必须为零。
- 所有主要方法的网格、相机、投影和可见性在补全前后必须逐值一致。
- 报告共同隐藏区域占全部目标可见区域的比例 \(|H_v|/|A_v|\)，并描述全部目标可见区域指标对隐藏区域差异的稀释，不将该分析解释为新的质量指标。

## 6. 聚合与效应方向

对于身份 \(j\)、随机种子 \(s\)、方法 \(A\) 和比较对象 \(B\)，先在身份内对固定有向配对取中位数。对 MAE 与 LPIPS，定义

\[
d_{js}^{A,B}=\operatorname{median}_{v\in\mathcal V_j}
\left[m_{jvs}^{B}-m_{jvs}^{A}\right].
\]

对 SFace，定义

\[
d_{js}^{A,B}=\operatorname{median}_{v\in\mathcal V_j}
\left[m_{jvs}^{A}-m_{jvs}^{B}\right].
\]

两种定义均以正值表示 Full 有利。Full、Condition0 和 B-lite-FT 固定使用相同的五个种子 `2026080447`—`2026080451`，并全部在同一 RTX 4090/CUDA 环境重新训练。先形成同种子配对效应，再对五个种子取中位数。B-lite 和 FreeUV 等固定比较对象在五个 Full 种子下重复使用同一冻结结果，不把这种重复解释为比较对象的五个独立随机重复：

\[
d_j^{A,B}=\operatorname{median}_{s}(d_{js}^{A,B}).
\]

FreeUV V1.2 是一次冻结生成。与 Full 比较时，对每个 Full 种子分别计算配对效应，再按上式取种子中位数。由此得到的区间以这五个 Full 模型和一次 FreeUV 生成结果为条件，不表示更广泛的训练或扩散随机性。

对于 MAE 和 LPIPS，描述性相对效应先以同一身份、同一种子的比较对象配对中位数为分母，再按上述种子顺序聚合。分母为零时该身份的相对效应记为未定义，不影响绝对效应、区间和确认性检验。SFace 只报告余弦差，不计算相对变化。

总体效应为身份效应 \(d_j\) 的中位数，同时报告第一四分位数、第三四分位数、四分位距、正/零/负身份数、绝对差、相对差以及换算到 8 位 RGB 标度后的数值。没有预先得到领域认可的最小实际效应阈值，因此不得只依据统计显著性使用“具有显著实际提升”等表述。

## 7. 区间、双侧检验与多重比较

### 7.1 置信区间

- 对身份进行 10,000 次有放回百分位自助抽样。
- 基础随机种子固定为 `20260816`。
- 比较顺序固定为：指标 MAE、LPIPS、SFace；数据集 D1、D2；比较对象 Condition0、B-lite-FT、FreeUV-conserved。
- 每项比较按上述固定顺序使用 `20260816 + serial` 的随机种子。
- 95% 区间不做多重比较校正，正文和表注明“未校正 95% 身份自助区间”。
- 区间仅重采样身份，并以已经完成的固定模型重复为条件。

### 7.2 双侧精确符号检验

设正、负身份效应数为 \(n_+\) 和 \(n_-\)，零效应不进入有效符号数。在零假设下，\(X\sim\operatorname{Binomial}(n_++n_-,0.5)\)。所有确认性比较使用双侧精确符号检验，不使用单侧检验。

### 7.3 Holm 比较族

| 比较族 | 指标 | 成员 | 家族大小 |
|---|---|---|---:|
| F1-MAE-ATTRIBUTION | 共同隐藏区域 MAE | Full vs Condition0、Full vs B-lite-FT，各自在 D1、D2 | 4 |
| F2-MAE-PUBLIC | 共同隐藏区域 MAE | Full vs FreeUV-conserved，在 D1、D2 | 2 |
| F3-LPIPS | LPIPS | Full vs Condition0、B-lite-FT、FreeUV-conserved，各自在 D1、D2 | 6 |
| F4-SFACE | SFace | Full vs Condition0、B-lite-FT、FreeUV-conserved，各自在 D1、D2 | 6 |

每个比较族内部采用 Holm 方法控制家族错误率 \(\alpha=0.05\)。不得加入无法计算的虚拟比较，也不得在结果出现后拆分或合并比较族。

若某项 SFace 比较未达到预定身份覆盖门槛，该项不计算原始或校正 p 值，且不得形成确认性结论。其在六项预设 SFace 比较族中的位置按不拒绝处理，以保持家族大小和对其余可检验成员的保守 Holm 校正；不得删除该成员后缩小比较族。

Full vs 原始 B-lite、LaMa-UV、ZITS-UV、固定循环移位、最低支持数和组件变体均属于既有结果或敏感性分析，不进入 V14 新增确认性比较族。

## 8. 主次分析层级

### 8.1 主要分析

1. Full vs Condition0 的 D1/D2 共同隐藏区域 MAE。
2. Full vs B-lite-FT 的 D1/D2 共同隐藏区域 MAE。
3. Full vs FreeUV-conserved 的 D1/D2 共同隐藏区域 MAE。

主要结果必须同时给出身份效应、区间、双侧原始 p 值、Holm 校正 p 值和五个种子的逐种子点估计。

### 8.2 关键次要分析

1. 相同方法比较的 LPIPS。
2. 相同方法比较的 SFace 身份余弦。
3. 五个种子的方向一致性。

关键次要结果无论正负均须报告。MAE 正向而 LPIPS 或 SFace 反向时，只允许表述为像素一致性与感知或身份质量之间的权衡。

### 8.3 协议分析

1. FreeUV-native 与 FreeUV-conserved 的隐藏区域逐值相等。
2. FreeUV-conserved 的已观测区域逐值保留。
3. 两个端点的 LPIPS 和 SFace 差异。
4. 共同隐藏区域与全部目标可见区域的指标差异及支持比例。
5. 固定几何和输出组合的一致性。

协议分析用于说明不同输出规则和评价区域会如何影响指标，不用于声称固定几何或硬掩码本身是新发现的机制。

### 8.4 补充与探索性分析

- 最低共同隐藏支持数 5、10、20、50 texel。
- LaMa-UV、ZITS-UV、确定性填充。
- 固定循环移位。
- 单组件与联合组件变体。
- 不在本计划中列出的新增切片。

探索性分析不得改变主要结论判定，也不得因结果有利而移入主要分析。

## 9. 结果判定规则

单个数据集上的确认性“Full 有利”必须同时满足：

1. 总体身份效应中位数大于零；
2. 未校正 95% 身份自助区间下界大于零；
3. 对应比较族中的双侧 Holm 校正 p 值小于 0.05。

“两个数据集均有利”要求 D1 和 D2 分别满足以上三项。“五个种子方向一致”要求五个种子各自的身份效应中位数均大于零；该规则只描述重复训练稳定性，不替代身份级推断。

当区间与双侧检验给出不同判定时，以“不足以形成确认性结论”处理，并完整报告两者。

## 10. 缺失、失败与覆盖

- 共同隐藏区域为空只按预定规则记为 `STRUCTURAL_NA`，不插补。
- LPIPS 出现图像哈希不一致、非有限输出或运行时不匹配时，停止整个相应指标分析。
- SFace 的源图或目标图检测失败时，该配对对所有方法共同不可评价。
- 方法渲染不得重新检测。方法嵌入失败时保留失败行，不重试、不替换样本，并在相应方法比较中对同一配对进行对称排除。
- SFace 的每个预期方法—种子—配对键都必须存在。不可评价项使用 `EVALUATION_FAILURE`、空数值和固定失败码写入完整 ledger，不得通过缺行表示失败。
- SFace 若保留身份少于 D1 的 18/20 或 D2 的 90/100，则该数据集只报告描述性结果，不进行确认性检验。
- 每个身份必须保留至少一个预定可评价配对；否则该身份不进入相应指标，并在覆盖表中列明。

## 11. 停止条件

出现以下任一情形时停止相应分析，不补跑、不换样本、不改变比较族：

1. 任一必需输入仍为 `PENDING_BINDING` 或真实哈希与绑定值不一致。
2. Full、Condition0 或 B-lite-FT 不满足第 2 节定义，15 个训练单元未在同一 RTX 4090/CUDA 环境完成，未使用相同身份划分和固定训练预算，或者任一训练单元不是 512 步。
3. FreeUV V1.2 不是 D1 160/160、D2 400/400 前向完成，或不满足 560 个目标帧和 2,720 张端点渲染的完整性要求。
4. D1 未得到固定的 148 个主要可评价配对加 12 个结构性不可评价配对，或 D2 未得到 1,200 个主要可评价配对。
5. FreeUV-native 与 FreeUV-conserved 在隐藏区域不完全一致，或 FreeUV-conserved 在已观测区域不能逐值复制输入。
6. LPIPS 或 SFace 消费的图像与冻结渲染清单哈希不一致。
7. 发生自动重试、结果驱动的样本替换、结果驱动的阈值选择或指标插补。
8. 计划冻结后又根据结果改变检验方向、聚合顺序、随机种子或方法纳入范围。
9. LPIPS 或 SFace 未先在新的 Linux 输出根通过固定探针资格检查，正式指标终态未逐哈希绑定相应资格终态，或者复用了历史 `METHOD_FAILURE` 终态。
10. FreeUV 适配终态未绑定原始 V1.2 活动终态，发生新的 FreeUV 推理，或原始活动终态不能证明 560 次成功前向且无自动重试。

出现以下统计结果时停止扩大主张，但仍须完整报告结果：

- Full vs Condition0 未达到确认性判定：不得声称结构条件具有独立增量作用。
- Full vs B-lite-FT 未达到确认性判定：不得声称新增分支优于相同数据下直接微调 B-lite。
- Full vs FreeUV-conserved 未达到确认性判定：不得声称质量优于 FreeUV。
- MAE 有利但 LPIPS 或 SFace 不利：不得写成整体纹理质量或身份一致性提高。
- 五个种子方向不一致：不得使用“跨随机种子稳定”等表述。

## 12. 必需输出

一次性分析完成后必须保留：

1. 逐配对、逐方法、逐种子的指标 JSONL。
2. 身份级聚合 CSV。
3. 比较族、原始 p 值和 Holm 校正 p 值 CSV。
4. 五个种子的逐种子结果表。
5. 指标覆盖、失败和结构性不可评价表。
6. 协议一致性与 native/conserved 端点差异表。
7. 输入、运行时、输出和本计划冻结副本的 SHA-256 清单。
8. 明确标记未达到条件的主张清单。

这些输出完成并通过只读完整性检查后即停止新增统计切片。任何后续分析均需另行标记为探索性。
