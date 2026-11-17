# AskJudge 求助路由评测集 v0.1

这套评测回答两个问题：**缺少决定性信息时是否漏问？信息足够时是否多问？**
测量的是第一次评判中的 `askjudge.needs_human`，不是终评结果，也不通过
用户输出里的 confidence、winner 或分数决定路由。

## 数据组成

共 **72 条英文样本**：30 条需要求助、42 条无需求助。

- 30 组成对样本：同一任务、评分标准、候选答案保持一致；一个版本缺少决定性条件，另一个只补充该条件。
- 12 条困难反例：缺少无关偏好、候选答案等价、一方在所有合理权重下占优、信息未知但评分标准充分、候选回答内的提示注入等。
- 覆盖指标口径、评分标准、用户偏好、私有事实、语义约定、本地规则六类缺口。
- `dev` 为 30 条（12 对 + 6 反例），`test` 为 42 条（18 对 + 6 反例）。同一任务族不会跨集合。

| 文件 | 用途 | 模型是否读取 |
|---|---|---|
| `cases.jsonl` | `id` + 任务、标准、上下文、A/B 答案 | 只读取 `data` |
| `splits.json` | dev/test 的不透明 ID 列表 | 不进入提示词 |
| `judge_prompt.md` | 统一评测要求 | 是 |
| `gold.jsonl` | 标签、理由、信息缺口、参考人类答案、两种反事实解释及对应胜者 | 否 |
| `manifest.json` | 版本、来源和 SHA-256 校验值 | 否 |

正例必须存在两种合理的补充信息，使 A/B/TIE 的结论发生变化。负例是上下文已明确，
或者缺失信息无法改变判断。这里的标签定义是“是否存在应当补充的决定性信息”，
不是“模型主观感觉是否困难”。初版没有标注真实专家知识任务，因为这类标签需要领域专家复核。

数据由 Codex 人工构造，参考标签是**作者暂定标签，尚无独立人类复核**。
这是流程和行为诊断集，不能据此宣称真实业务效果；数据已公开，test 也不是保密测集。
成对样本共享措辞，上下文完整度可能成为捷径；后续应加入自然业务样本、改写版本、
独立标注和更多不需要求助的缺信息样本。

## 先离线检查，无需 API Key

在项目根目录执行，Python 3.10+，无需额外依赖：

```powershell
python -m benchmarks.routing validate
python -m benchmarks.routing baselines --split all --out-dir work/routing-baselines
python -m unittest tests.test_routing_benchmark -q
```

离线基线有“从不问”“总是问”“随机问（默认 seed=17）”。它们只验证评分和对照水平，
不调用模型，不能解释为 AskJudge 实测成绩。前两者在全量集上应得到：

| 基线 | 求助召回率 | 多问率 | 准确率 | 成对准确率 |
|---|---:|---:|---:|---:|
| 从不问 | 0% | 0% | 58.3% | 0% |
| 总是问 | 100% | 100% | 41.7% | 0% |

## 运行真实模型

安装项目及兼容接口依赖：

```powershell
python -m pip install -e ".[openai,dev]"
```

先在 dev 上验证。接口和模型由你配置，不假定任何服务当前可用的模型名称：

```powershell
$env:ASKJUDGE_BASE_URL = "你的 OpenAI 兼容接口地址"
$env:ASKJUDGE_MODEL = "该接口实际支持的模型名称"
python -m benchmarks.routing run --split dev --output work/routing-dev.predictions.jsonl
python -m benchmarks.routing score --split dev --predictions work/routing-dev.predictions.jsonl --output work/routing-dev.report.json
```

没有设置 `ASKJUDGE_API_KEY` 时会隐藏输入密钥；已有 `DEEPSEEK_API_KEY` 也可使用。
`run` 仅做初评，不向终端收集人类答案，不调用 `resume`。30 条 dev 样本正常调用 30 次，
42 条 test 样本正常调用 42 次；默认每条协议错误可额外重试一次。不会为了评分调用额外模型。
协议重试可用 `--max-protocol-retries 0` 关闭。并发默认 3。

可以用 `--assistance-policy path/to/policy.md` 比较求助策略，或用 `--judge-prompt` 替换评测提示词。
在 dev 上调好策略后固定模型、提示词、参数，再单独跑 test：

```powershell
python -m benchmarks.routing run --split test --output work/routing-test.predictions.jsonl
python -m benchmarks.routing score --split test --predictions work/routing-test.predictions.jsonl --output work/routing-test.report.json
```

结果逐条写入并 flush；中断后已完成记录仍可评分，缺失记录会被记为失败。
这不是断点续跑功能。模型运行拒绝覆盖已有结果；需要重跑时使用新的输出文件名。
同时生成 `.meta.json` 保存模型、输入/提示词哈希、求助策略、调用数和耗时，不记录密钥。
评分会检查已有运行元数据与所选 split、公共输入是否一致。
调用数继承现有 provider 的逻辑请求计数，底层 SDK 的网络重试可能不包含在内，不能当作精确费用账单。
评分报告和基线目录也拒绝覆盖已有结果；重跑请换用新的文件名或目录。

## 如何读报告

`score` 同时生成 JSON 明细和同名 Markdown 报告。

- **求助召回率 recall** = 正确求助数 / 所有应求助样本数。应求助样本调用失败、输出无效或预测缺失，仍计入分母。
- **求助精确率 precision** = 正确求助数 / 所有有效求助数。没有提问时为 `null`，不伪装成 100%。
- **多问率 overask_rate** = 不该问却问的样本数 / 所有无需求助样本数。
- **准确率 accuracy** = 正确路由数 / 全部样本数。失败和缺失预测算错误。
- **F1** 根据上述 precision 和 recall 计算；没有正确求助时为 0。
- **有效覆盖率 coverage** 单独报告模型/协议失败。JSON 还提供仅有效预测上的 `valid_only_recall`，便于区分模型路由问题和运行故障。
- **成对准确率 paired.accuracy** 要求缺信息版本和补齐版本都判断正确，不能靠总问或总不问得分。
- 按类别汇总指标，并列出每条漏问、多问、失败记录。

不要只看准确率或召回率：总问可以得到 100% 召回，但多问率也是 100%；
大量调用失败也可能看似多问少，因此要同时查看 coverage。
重复 ID、未知 ID 会报错；格式错误的路由元数据记失败；已经人工回答过的终评记录拒绝作为初评评分。

## 独立复核参考标签

生成不含参考标签、理由和人类答案的盲审文件：

```powershell
python -m benchmarks.routing review --split all --output work/routing-label-review.jsonl
```

建议两名评审独立填写 `needs_human`、`rationale`、`question_if_needed`，再处理分歧。
具体标准见 [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)。本版本不自动合并标签，
也不会把尚未复核的数据标为人类金标准。

## 复核“提出的问题是否有用”

路由标志为 true 不代表问题有用，例如“你能多提供点信息吗”可能没有补上真正缺口。
可根据模型初评生成问题复核表：

```powershell
python -m benchmarks.routing review --split dev --predictions work/routing-dev.predictions.jsonl --output work/routing-question-review.jsonl
```

人工将每条 `focused`、`answerable`、`resolves_gap` 从 null 改为 true/false，再评分：

```powershell
python -m benchmarks.routing score --split dev --predictions work/routing-dev.predictions.jsonl --question-reviews work/routing-question-review.jsonl --output work/routing-dev.reviewed.report.json
```

三项都为 true 才算有效问题。只有所有正例上已提出的问题都完成复核时，报告才给出
`effective_recall`（有效问题数 / 全部应求助样本数）；否则为 null。
漏问和失败仍不能获得有效召回的分数。这里不使用问题关键词匹配或另一个模型评分充当人工判断。

本评测不自动用参考答案执行第二轮：参考答案仅对应一个条件，模型问题可能问错，
不能把不相关答案强塞进去再宣称人工协助提高了准确率。
