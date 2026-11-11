# RAG 实验台账

最后更新：2026-11-11

## 环境约定

| 用途 | conda 环境 |
|---|---|
| 指标 / 检测 / 诊断（`src/`） | `av` |
| **RAG 全部**（建索引、检索评测、生成评测、端到端） | **`sft`** |

`sft` 环境含 chromadb 1.5.9、sentence-transformers 6.2.0、duckdb 2.0.0、
torch 2.14.1+cu126、transformers 5.22.1、peft 0.21.2、trl 1.17.1。
GPU：RTX 4060 Laptop 8GB，`torch.cuda.is_available() = True`。

## 数据版本

| 资产 | 说明 |
|---|---|
| 知识库 | `monthly_cost_troubleshooting_knowledge.jsonl`，18 张卡（14 条规则全覆盖 + 7 干扰 + 2 数据质量） |
| GT | `rag_goldenset_v1.jsonl`，168 条（28 母题 × 6 变体）；v2 修订过陷阱组问法 |
| 索引 | `vector_store/monthly_cost_troubleshooting_chroma_v2`，collection `monthly_cost_troubleshooting_v2` |
| 别名表 | `config/rag_aliases.json`，18 张卡 |

## 检索层实验

| 编号 | 变更 | raw R@3 | filtered R@3 | 干扰进Top3 | 结论 |
|---|---|---|---|---|---|
| E1 | 基线（GT v1） | 80.8% | 96.2% | 72.9% | 采用 |
| E2 | + BGE 查询指令前缀 | 80.1% | 96.2% | 70.8% | **否决**：单卡 +2.1、陷阱 -4.7，净负 |
| E3 | 卡片补英文别名 | 86.5% | 96.2% | 54.2% | 保留（见下） |

> E3 之所以"持平"，是因为别名被均值池化稀释：给所有卡都加英文词等于给全员加噪。
> 别名实际贡献在 R@5（91.0% → 92.3%）。

### 当前基线（GT v2）

- raw：Recall@1 71.7% / Recall@3 **86.5%** / Recall@5 92.3% / MRR 0.822
- filtered：Recall@1 96.4% / Recall@3 **96.2%** / MRR 0.983
- 干扰卡进入 Top-3 的比例：54.2%

> filtered 路径下干扰卡与大部分辅卡被硬过滤排除，其干扰率恒为 0，
> **不能与 raw 直接对比检索能力**，只能说明过滤层是有效的。

## 生成层实验

| 编号 | 变更 | 数值存在率 | 一致性缺陷 | 知识卡引用率 |
|---|---|---|---|---|
| E1-gen | 基线 prompt | 99.0%（103 数） | 4/17 条 | **0/17** |
| E2-gen | 禁自算 + 禁区间句式 + 强制引用 | 100%（68 数） | 2/17 条 | **17/17** |

副作用：输出变短，每条平均数值数 6.06 → 4.00（信息密度降 34%）。
**因此报告一致率时必须同时报"每条平均数值数"，否则指标会因"说得少"而虚高。**

残留缺陷：浮点泄漏 4 条次未解决。根因不是"精度"，是模型把 `change_rate`
当成本期值并自行加 `%`（字段误用 + 标度错），prompt 约束无效，需生成后校验（guard）。

## 拒答评测

| 编号 | 模式 | 结果 |
|---|---|---|
| R1 | 仅生成层（给无关知识卡、不给业务数据） | 0/12 = **0%**，且会凭空生成财务表格 |
| R2 | 端到端（四节点） | 6/12 = **50%** |
| R3 | 端到端 + 意图路由收紧（2a） | 9/12 = **75%** |

R3 残留 3 条：Q163（讲笑话，判 general_chat，可接受）、
Q164（翻译，明确误判）、Q157（写周报，业务灰色地带，单独标记不计入拒答率）。

**未做**：把 Q163/Q164 直接写进反例清单可冲到 11/12，但那是对着考题改提示词，
指标虚高，不做。

## 验收线汇总

| 验收线 | 结果 | 说明 |
|---|---|---|
| 检索 Recall@3 | 86.5%（纯向量）/ 96.2%（规则过滤） | 168 条人工 GT，28 母题分层报告 |
| 数字一致率 | 100%（68/68），但缺陷率 2/17 | 口径 B：归一化数值 + 百分比互换 |
| 拒答准确率 | 75%（端到端）；生成层单独 0% | 单点防御，plan 误判即穿透 |

## 已知局限（面试要主动说）

1. **K003 分摊成本卡最弱**：R@3 62.5%，多卡组 61.1% 的失败几乎都因它
2. **中英混用查询弱**：V5 失败率 23.1%。bge-small-zh 是中文专用模型，
   补别名已验证无效，要根治需换多语言模型（bge-m3）
3. **生成层浮点泄漏未解决**，需 guard 重试
4. **拒答是单点防御**：plan 判错一次，answer 层无兜底（R1 的 0% 即此证的极端情况）
5. **GT 由本人标注**，28 母题覆盖 14 条规则 + 7 干扰 + 2 数据质量

## 复现命令

```bash
# 建索引（18 卡）
/home/qiaojiahao/miniforge3/envs/sft/bin/python -m scripts.build_monthly_cost_rag_index_v2

# 生成 GT（168 条）
python -m scripts.build_rag_goldenset

# 检索评测
/home/qiaojiahao/miniforge3/envs/sft/bin/python -m scripts.evaluate_rag_retrieval --save

# 生成评测（数字一致率）
/home/qiaojiahao/miniforge3/envs/sft/bin/python -m scripts.evaluate_rag_generation --save

# 拒答评测（端到端）
/home/qiaojiahao/miniforge3/envs/sft/bin/python -m scripts.evaluate_rag_refusal --end-to-end