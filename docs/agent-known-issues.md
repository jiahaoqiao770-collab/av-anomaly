# 诊断助手 Agent：已知问题清单

最后更新：2026-11-12

## 一、当前状态

| 层 | 状态 |
|---|---|
| 指标计算 / 规则检测 / 诊断归并（src/） | 稳定，未改动 |
| RAG 检索 | 完成，Recall@3 86.5%（纯向量）/ 96.2%（规则过滤），见 docs/rag-experiments.md |
| RAG 生成（数字一致率） | 完成，100%（68/68），见 docs/rag-gen-eval-v1.md |
| SFT | 完成对照，槽位准确率 63.5% → 100%，见 docs/sft-eval-*.md |
| Agent 意图路由 + 端到端问答 | 进行中，30 题探针约 10~12 条可用，下列 8 个问题待修 |

## 二、环境（重要）

- WSL 3.0.5 + Ubuntu 24.04，内核 6.18.58.1-microsoft-standard-WSL2
- NVIDIA 驱动 621.14（2026-11-12 从 560.94 升级）

  升级原因：旧驱动下 `wsl --shutdown` 后虚拟机僵死、CUDA 报 `unknown error`、
  显存充足却分配失败（`memory mapping failed`），一天崩 4 次。
  升级后：连续跑 30 题不再出现 CUDA 错误。

- 统一用 `sft` 环境跑 RAG / SFT / Agent 演示；`av` 只跑 src 那一套。

## 三、已完成并验证的修复

| 编号 | 修复 | 验证方式 |
|---|---|---|
| F1 | plan 必须输出完整 JSON（带 unsupported 示例） | 30 题里非 JSON 从 13 条降到 1 条 |
| F2 | intent 别名表（explain_anomaly / explain_anomalies / diagnose） | Q14/15 不再报「不支持的查询类型」 |
| F3 | business_type 中英文归一化（parking → 泊车） | Q19 不再因英文业务名查不到 |
| F4 | 指标白名单校验（不在 METRICS 里直接报错） | Q26 返回「系统没有这些指标」 |
| F5 | 时间锚点（最新月 2026-11，「上个月」= 2026-10） | Q6 从 2026-01 修正为 2026-10 |
| F6 | 回答层禁止复述问题、变化率强制百分比 | Q14 不再复读问题 |
| F7 | 线上提示词格式对齐 SFT 训练格式（事实数据：） | Q14 从复述变为正常回答 |

## 四、待修问题（按严重度）

### A1（最严重）9 月有 4 条异常，却答「没有异常」

- 现象：`9月有什么异常吗？` → `2026-09 无异常`
- 实际：2026-09 行车 + 泊车各触发 A03、A04，共 4 条事件
- 已确认：query 层返回了这 4 条（get_anomalies 三业务合并逻辑正确）
- 根因：answer 层拿到数据但无视它，仍然套模板
- 建议修法：在 answer 的 user content 里把查询结果显式转成中文列表
  （不要直接丢 JSON 数组），例如「查询到 4 条异常：1) 行车 单Clip成本超过绝对上限 2) ...」，
  让模型难以忽略
- 验证：重跑 Q13，必须列出 A03、A04

### A2 大小比较反向

- 现象：行车 74.65 vs 泊车 13.37，却回答「泊车单Clip成本更贵」
- 根因：模型在做数值比较（这是它不该做的事，违反「数值由程序算」原则）
- 建议修法：query 层对 compare_metric 直接返回「哪个更大」字段，answer 只负责复述
- 验证：重跑 Q9

### A3 业务 / 指标读错

- Q29：问「5月泊车、6月行车哪个成本更高」，回答成「5月泊车 vs 6月泊车」
- Q19：问 Clip 数下降，回答单Clip成本上涨
- 根因：plan 的 business_type / metric_ids 与问句不完全对齐
- 建议修法：query 返回时把 metric_id + 中文名 + 月份 + 业务 拼成一句话给模型，
  减少模型自行组合的空间

### A4 过度拒答（最近一次修复引入）

- 现象：Q5（是不是试采搞多了）、Q16（7月泊车那几条规则）、Q20（是不是车少了）、
  Q21（是不是口径变了）全部被判 unsupported
- 根因：EXTRA_PLAN_RULES 第 15 条「清单外指标一律 unsupported」管得太宽
- 建议修法：把拒答限定为「指标层面」（问的是不存在的指标），而非「问句层面」；
  能映射到已有指标且语义在业务范围内的，一律不拒
- 验证：重跑 Q5、Q16、Q20、Q21 应正常回答，Q23、Q24 仍应拒答

### A5 plan 偶发非 JSON

- 现象：Q17 → `{}` + 「意图识别没有返回JSON：unsupported」
- 根因：模型在边界问题上仍偶尔只吐一个词
- 建议修法：parse_json 失败时重试一次，重试时追加「只输出JSON，不要解释」
- 验证：Q17 能正常出 plan

### A6 内存泄漏（跑到第 24 题几乎必 OOM）

- 现象：30 题连跑，第 24~25 题显存耗尽（free: 0）
- 根因：graph.invoke 每次调用有对象未释放（疑为 chromadb client 每次新建、
  或 langgraph state 引用累积）
- 临时对策：分批跑，每批 <= 10 题（--start/--end），进程退出即释放
- 建议修法：chromadb client 与 embedding model 提到模块级复用；每轮 empty_cache
- 验证：单进程连跑 30 题不 OOM

### A7 多月份 / 季度 / 累计 不支持（schema 限制）

- 现象：问「6、7、8 月分别多少」「Q3 vs Q2」「今年累计」→ 只能给两个月或拒答
- 根因：plan 的 schema 只有 period + baseline_period（两个月），无法表达 N 个月或区间聚合
- 建议修法：扩展 schema 为 periods: []，query 层支持多月份聚合；属结构性改动

### A8 浮点泄漏

- 现象：`总成本下降了8.699999999999989万元，变化率为-0.06350364963503641`
- 根因：模型直接复述 JSON 里的原始 float
- 建议修法：query 层返回前统一 round(x, 2) 并转成字符串
- 验证：全 30 题不再出现 5 位以上小数

## 五、复现与验证命令

```bash
# 单批（10 题以内，避免 OOM）
/home/qiaojiahao/miniforge3/envs/sft/bin/python -m scripts.probe_plan_30 --start 0 --end 10

# 全量（不推荐，会 OOM）
/home/qiaojiahao/miniforge3/envs/sft/bin/python -m scripts.probe_plan_30

# 交互演示
/home/qiaojiahao/miniforge3/envs/sft/bin/python -m scripts.run_monthly_cost_langgraph_assistant --adapter data/sft/output/sft_v4
```

## 六、30 题探针当前表现（2026-11-12 21:57）

| 类别 | 条数 | 题号 |
|---|---|---|
| 正确 | 10~12 | 2、3、7、8、14、15、18、22、26、27、30 |
| 过度拒答 | 6 | 5、16、20、21、23、24 |
| 答错内容 | 4 | 9、13、19、29 |
| plan 崩溃 | 1 | 17 |
| OOM 跳过 | 1 | 25 |
| 指标选错 | 2 | 1、2 |

## 七、面试口径

> RAG 层是完整的：18 张卡、168 条人工 GT、Recall@3 86.5%（纯向量）/ 96.2%（规则过滤），
> 数字一致率 100%，拒答 75%，线上已接同一套索引。
> SFT 做了 base vs 微调对照：槽位准确率 63.5% → 100%，但发现它主要提升的是格式规范，
> 端到端 30 题正确率仍只有约四成。
> 拆解后定位到瓶颈在意图路由和查询层，不在生成层，所以没有继续做 DPO——
> 先把 plan 的指标映射和上下文传递修好，比继续训练更有效。
> 已知的 8 个问题、根因和修法都记在 docs/agent-known-issues.md 里。