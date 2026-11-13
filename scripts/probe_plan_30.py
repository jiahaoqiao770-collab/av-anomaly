#!/usr/bin/env python3
"""30 道题批量问生产链路，逐条打印 问题 / plan / 回答，供人工判断。

不打分、不判定，只输出原料。
结果同时写入 docs/plan-probe-30.md
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

import scripts.run_monthly_cost_langgraph_assistant as prod

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "docs/plan-probe-30.md"

QUESTIONS = [
    # —— 指标歧义 / 单位 ——
    "8月泊车的clips比7月少了多少？",
    "6月的泊车有效单Clip成本是多少？",
    "8月泊车分摊成本是 103.8257 万元对吧？",
    "我们8月一共采了多少天？",
    # —— 口语 / 含糊 / 时间表达 ——
    "我们7月成本有点高，帮我看看是不是试采搞多了",
    "上个月泊车怎么样？",
    "第三季度整体成本比第二季度高吗？",
    "2025年12月到2026年1月，总成本降了多少？",
    # —— 对比 / 多口径 ——
    "行车和泊车的单Clip成本差多少，哪个更贵？",
    "8月泊车每小时的产出比7月高还是低？",
    "12月到2月整体是什么趋势？",
    "试采占比最近几个月是什么趋势？",
    # —— 异常 / 诊断 ——
    "9月有什么异常吗？",
    "11月泊车单Clip成本为什么连续涨？",
    "2月泊车为什么单clip成本涨这么多？",
    "7月泊车那几条规则分别是什么？",
    "为什么11月行车只有一条规则触发？",
    "6月应该没异常吧？",
    # —— 错误前提 ——
    "7月泊车clip数下降，是数据没入库还是真采少了？",
    "4月行车成本涨是不是因为车少了？",
    "10月成本跳了一下，是不是口径变了？",
    "3月总成本涨了，钱花哪了？",
    # —— 数据不存在 / 边界 ——
    "我们硬盘邮寄费这个月花了多少？",
    "今年累计总成本多少？",
    "把去年12月到今年11月的总成本都列出来",
    "泊车试采占比超过多少算异常？",
    # —— 应拒答 ——
    "帮我写个SQL查一下clip数",
    "哪个业务更烧钱？",
    "5月泊车6月行车哪个成本更高？",
    "环比和同比有什么区别？",
]


def main() -> None:
    tokenizer, model = prod.load_qwen()
    adapter = PROJECT_ROOT / "data/sft/output/sft_v4"
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter))
    print(f"已加载 LoRA：{adapter}")
    graph = prod.build_graph(tokenizer, model)

    lines = ["# 30 题探针（plan + answer 原始输出）", ""]
    for index, question in enumerate(QUESTIONS, start=1):
        torch.cuda.empty_cache()
        try:
            state = graph.invoke({"question": question})
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n[{index}/30] {question}\n!! OOM，跳过")
            lines += [f"## {index}. {question}", "", "**OOM 跳过**", ""]
            continue
        plan = state.get("plan") or {}
        answer = (state.get("answer") or "").strip()

        print(f"\n{'=' * 60}")
        print(f"[{index}/30] {question}")
        print(f"plan: {json.dumps(plan, ensure_ascii=False)}")
        print(f"回答: {answer}")

        lines.append(f"## {index}. {question}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(plan, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append(answer)
        lines.append("")

        if index % 5 == 0:
            OUT.write_text("\n".join(lines), encoding="utf-8")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写入：{OUT}")


if __name__ == "__main__":
    main()
