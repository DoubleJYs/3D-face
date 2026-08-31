# V14 统计闭环使用说明

本文件说明同目录中的统计侧文件。训练与推理入口见 `README.md`，统计侧不调用训练代码，也不修改任何输入或论文文件。

## 冻结文件

- `contract.v1.json` 固定五个种子、同一 RTX 4090/CUDA 环境下的 15 个训练单元、每单元 512 步、三个指标和四个比较族。
- `analyze_v14_matched_controls.py` 实现身份级聚合、10,000 次身份百分位自助法、双侧精确符号检验和族内 Holm 校正。
- `INPUT_BINDING.template.json` 给出生产输入绑定结构。
- `test_statistics_synthetic.py` 使用合成数据检验正确路径和拒绝路径，不包含论文结果。

## 统计顺序

每个方法先在同一身份内对固定有向配对取中位数，再对五个同种子配对效应取中位数。总体点估计为身份效应中位数。自助法只重采样身份，不重采样 texel、视角对、通道或种子。正效应统一表示 Full 有利。

四个预设比较族为：

1. MAE 归因比较：Full 对 Condition0、B-lite-FT，分别在 D1/D2 上比较，共 4 项。
2. MAE 公开方法比较：Full 对 FreeUV-conserved，分别在 D1/D2 上比较，共 2 项。
3. LPIPS：上述三个比较对象分别在 D1/D2 上比较，共 6 项。
4. SFace：上述三个比较对象分别在 D1/D2 上比较，共 6 项。

每个比较族内部独立执行 Holm 校正。确认性有利结论同时要求身份效应中位数大于零、未校正 95% 身份自助区间下界大于零、族内 Holm 校正双侧 p 值小于 0.05。

## 关闭式失败条件

以下任一情况发生时，不创建正式输出目录：

- 任一训练、缓存、渲染、LPIPS 或 SFace 终态不是合同规定的完成状态。
- LPIPS 或 SFace 没有来自新输出根的 Linux CPU 资格终态，资格终态的 CUDA 调用数不为 0，没有绑定当前运行环境、执行脚本、模型或导出文件和固定探针，或者正式指标终态没有逐哈希绑定该资格终态。
- 15 个训练终态没有共同绑定同一环境、训练划分和训练预算哈希。
- 15 个训练终态未绑定 15 个不同 checkpoint。
- 历史 MPS Full 被混入确认性训练终态。
- 配对清单的身份数或配对数不符合 D1 20/160/148、D2 100/1200/1200 的冻结规模。
- 指标 JSONL 有缺行、额外行、重复行、非有限值、越界值或支持数不一致。
- FreeUV 适配终态没有声明 `no_new_inference=true`，没有绑定原始 V1.2 活动终态，或原始活动终态不是 560 次成功前向且无自动重试。
- 任一绑定文件、终态、checkpoint、FreeUV V1.2 输入或统计程序的 SHA-256 不一致。
- 输出目录已经存在。

旧 LPIPS 准备阶段的 `METHOD_FAILURE` 不能绑定为完成输入。必须先在新的输出根生成 `PASS_V14_LPIPS_LINUX_QUALIFIED`，再由新的 `PASS_V14_LPIPS_COMPLETE` 逐哈希引用该资格终态。SFace 使用同样的独立资格与正式完成链。

SFace 检测或嵌入失败不允许用缺行表示。每个预期键仍须写入 JSONL，使用 `terminal_state="EVALUATION_FAILURE"`、`value=null` 和固定 `failure_code`。源图或目标图检测失败必须对该配对的所有方法与种子完全对称。方法嵌入失败在相应成对比较中对称排除。保留身份少于 D1 的 18 个或 D2 的 90 个时，程序仍生成覆盖和描述性结果，但关闭相应确认性判定。

## 检查命令

```bash
python3 analyze_v14_matched_controls.py contract-check --contract contract.v1.json
python3 test_statistics_synthetic.py
```

所有实验和指标完成后，将模板中的 `PENDING` 字段替换为实际绝对路径与 SHA-256，并把绑定状态设为 `FROZEN_COMPLETE`，再运行：

```bash
python3 analyze_v14_matched_controls.py analyze \
  --contract contract.v1.json \
  --binding /absolute/path/INPUT_BINDING.json
```

正式输出包含比较结果、身份效应、逐种子结果、覆盖记录、主张判定条件、输入验证记录及输出终态。任何未列入四个预设比较族的后续切片均应单独标为探索性。
