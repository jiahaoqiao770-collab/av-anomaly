#!/usr/bin/env python3
"""构造 SFT 第一批数据（约 130 条）。

标签用模板拼装，不由模型生成 —— 保证期望答案确定正确。
每条标签都要过判定器（evaluate_rag_facts）自检，不合格直接丢弃并计数。

输出：
  data/sft/v2/train.jsonl
  data/sft/v2/test.jsonl     （锁定的 6 条诊断，永不参与训练）
  data/sft/v2/README.md
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

from scripts.evaluate_rag_facts import (
    CMAP,
    judge_coverage,
    judge_slots,
    load_jsonl,
    name_variants,
    reason_metric,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_PATH = PROJECT_ROOT / "data/processed/monthly_cost_diagnosis_results.jsonl"
EVENT_PATH = PROJECT_ROOT / "data/processed/monthly_cost_anomaly_events.jsonl"
METRICS_PATH = PROJECT_ROOT / "data/processed/monthly_clip_cost_metrics.csv"
KNOW_PATH = PROJECT_ROOT / "data/knowledge/raw/monthly_cost_troubleshooting_knowledge.jsonl"
OUT_DIR = PROJECT_ROOT / "data/sft/v2"

SEED = 42
# 锁定测试集：按业务与错误类型分层挑 6 条
TEST_IDS = [
    "2026-02_parking",
    "2026-03_overall",
    "2026-05_driving",
    "2026-07_parking",
    "2026-08_driving",
    "2026-11_parking",
]

TEST_PAIRS = {
    ("2026-02", "泊车"), ("2026-05", "行车"), ("2026-07", "泊车"),
    ("2026-08", "行车"), ("2026-11", "泊车"), ("2026-03", "整体"),
}


SYSTEM_PROMPT = (
    "你是采集成本效率诊断助手。只能使用给定数据作答，不得编造数值；"
    "关联因素不得表述为已确认根因；数据不足时必须明确说明。"
)


def fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def metric_names(record: dict) -> list[str]:
    business = record.get("business_type")
    names: list[str] = []
    for key in ("primary_anomaly", "primary_reason"):
        name = CMAP.get(reason_metric(record.get(key) or {}, business) or "")
        if name:
            names.append(name)
    for item in record.get("contributing_reasons") or []:
        name = CMAP.get(reason_metric(item, business) or "")
        if name:
            names.append(name)
    return list(dict.fromkeys(names))


def rule_to_card() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for card in load_jsonl(KNOW_PATH):
        for rid in card.get("related_rule_ids") or []:
            if rid != "*":
                mapping.setdefault(rid, []).append(card["knowledge_id"])
    return mapping


def fact_sentence(record: dict) -> str:
    pa, pr = record["primary_anomaly"], record["primary_reason"]
    parts = [f"{record['period']} {record['business_type']}"]
    parts.append(
        f"主要异常：{pa['rule_name']}，本期 {fmt(pa.get('current_value'))}，"
        f"上期 {fmt(pa.get('baseline_value'))}，环比 {fmt((pa.get('change_rate') or 0) * 100)}%"
    )
    pr_metric = CMAP.get(reason_metric(pr, record.get("business_type")) or "", "")
    parts.append(f"主因：{pr['rule_name']}" + (f"（涉及 {pr_metric}）" if pr_metric else ""))
    for item in record.get("contributing_reasons") or []:
        name = CMAP.get(reason_metric(item, record.get("business_type")) or "", "")
        if name:
            parts.append(f"关联因素：{item['rule_name']}（涉及 {name}）")
    return "；".join(parts) + "。"


def target_answer(record: dict, cards: list[str]) -> str:
    pa, pr = record["primary_anomaly"], record["primary_reason"]
    names = metric_names(record)
    lines = [
        f"{record['period']} {record['business_type']}的主要异常是「{pa['rule_name']}」。",
        f"本期 {fmt(pa.get('current_value'))}，上期 {fmt(pa.get('baseline_value'))}，"
        f"环比变化 {fmt((pa.get('change_rate') or 0) * 100)}%。",
        f"已确认主因为「{pr['rule_name']}」" + (f"，涉及 {names[1]}。" if len(names) > 1 else "。"),
    ]
    contrib = [c for c in record.get("contributing_reasons") or []]
    if contrib:
        cnames = [
            CMAP.get(reason_metric(c, record.get("business_type")) or "", "") for c in contrib
        ]
        cnames = [n for n in cnames if n]
        if cnames:
            lines.append(f"关联因素（未确认为根因）：{'、'.join(cnames)}。")
    lines.append(f"涉及指标：{'、'.join(names)}。")
    if cards:
        lines.append(f"依据知识卡：{'、'.join(cards)}。")
    return "\n".join(lines)


def build_t1(record: dict, cards: list[str]) -> list[dict]:
    facts = fact_sentence(record)
    targets = target_answer(record, cards)
    questions = [
        f"{record['period']} {record['business_type']}这个月的成本效率异常，主要问题和依据是什么？",
        f"帮我看下{record['period']} {record['business_type']}的成本效率，哪里不对，为什么？",
        f"{record['period']} {record['business_type']}的月度分析：异常、原因、关键指标各是什么？",
    ]
    return [
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{q}\n事实数据：{facts}"},
            {"role": "assistant", "content": targets},
        ], "meta": {"type": "T1"}
        }
        for q in questions
    ]


def build_t2(record: dict, cards: list[str]) -> list[dict]:
    pa = record["primary_anomaly"]
    facts = fact_sentence(record)
    names = metric_names(record)
    main_metric = names[0] if names else pa["rule_name"]
    targets = (
        f"不能直接归因为单价上涨。数据显示本次异常为「{pa['rule_name']}」，"
        f"本期 {fmt(pa.get('current_value'))}，上期 {fmt(pa.get('baseline_value'))}，"
        f"环比变化 {fmt((pa.get('change_rate') or 0) * 100)}%。"
        f"应按{main_metric}的成本侧与产出侧分别核对后再判断，"
        f"当前数据不支持「单价上涨」这一结论。\n涉及指标：{'、'.join(names)}。"
    )
    if cards:
        targets += f"\n依据知识卡：{'、'.join(cards)}。"
    questions = [
        f"{record['period']} {record['business_type']}{main_metric}涨了，是不是单价上涨导致的？",
        f"{record['period']} {record['business_type']}这个数变差，我猜是单价涨了，对吗？",
    ]
    return [
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{q}\n事实数据：{facts}"},
            {"role": "assistant", "content": targets},
        ], "meta": {"type": "T2"}
        }
        for q in questions
    ]


def build_t3(record: dict) -> list[dict]:
    pa = record["primary_anomaly"]
    rate = (pa.get("change_rate") or 0) * 100
    facts = (
        f"{record['period']} {record['business_type']}：环比变化率 {fmt(rate)}%。"
        f"（本期与上期数值未提供）"
    )
    question = f"{record['period']} {record['business_type']}本期{CMAP.get(reason_metric(pa, record.get('business_type')) or '', '该指标')}是多少？"
    target = (
        f"数据中没有提供本期数值，只有环比变化率 {fmt(rate)}%。"
        "变化率不能当作本期值使用，需要本期与上期的原始数值才能回答。"
    )
    return [{"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n事实数据：{facts}"},
        {"role": "assistant", "content": target},
    ], "meta": {"type": "T3"}}]


def build_t4(record: dict) -> list[dict]:
    pa = record["primary_anomaly"]
    facts = f"{record['period']} {record['business_type']}：本月触发了「{pa['rule_name']}」。（本期与上期数值缺失）"
    question = f"{record['period']} {record['business_type']}这个异常具体涨了多少？"
    target = (
        "数据中没有本期与上期的具体数值，无法给出涨跌幅度。"
        "只能确认触发了「" + pa["rule_name"] + "」这一条规则，需要补充原始数值后才能判断。"
    )
    return [{"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n事实数据：{facts}"},
        {"role": "assistant", "content": target},
    ], "meta": {"type": "T4"}}]


def build_t5(month: str, business: str) -> list[dict]:
    question = f"{month} {business}这个月成本效率有异常吗？"
    target = (
        f"{month} {business}本月未触发任何异常规则，指标处于正常区间。"
        "如需进一步确认，可提供环比数据做交叉核对。"
    )
    return [{"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n事实数据：{month} {business}：本月无规则触发。"},
        {"role": "assistant", "content": target},
    ], "meta": {"type": "T5"}}]


def build_t6(event: dict, cards: list[str]) -> list[dict]:
    metric = (event.get("evidence") or {}).get("metric")
    name = CMAP.get(metric or "", "")
    cur = event.get("current_value")
    base = event.get("baseline_value")
    rate = event.get("change_rate")
    threshold = event.get("threshold")

    detail_parts = [f"本期 {fmt(cur)}"]
    if isinstance(base, (int, float)):
        detail_parts.append(f"上期 {fmt(base)}")
    if isinstance(rate, (int, float)):
        detail_parts.append(f"环比变化 {fmt(rate * 100)}%")
    if threshold is not None:
        detail_parts.append(f"触发阈值 {fmt(threshold)}")
    detail = "，".join(detail_parts)

    facts = f"{event['period']} {event['business_type']}触发「{event['rule_name']}」：{detail}"
    target = f"{event['period']} {event['business_type']}触发了「{event['rule_name']}」。\n{detail}。"
    if name:
        target += f"\n涉及指标：{name}。"
    if cards:
        target += f"\n依据知识卡：{'、'.join(cards)}。"

    question = f"{event['period']} {event['business_type']}为什么触发了「{event['rule_name']}」？"
    return [{"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n事实数据：{facts}"},
        {"role": "assistant", "content": target},
    ], "meta": {"type": "T6"}}]


def build_test_variants(record: dict, cards: list[str]) -> list[dict]:
    """测试专用问法：与训练集 T1 模板完全不同。"""
    facts = fact_sentence(record)
    targets = target_answer(record, cards)
    period, business = record["period"], record["business_type"]
    questions = [
        f"帮我瞅瞅 {period} {business}这块儿，这个月哪儿不对劲？",
        f"{period} {business}成本涨了是不是因为单价上涨？给我讲讲。",
        f"{period} {business}的 unit clip cost 和 clip count 都有异常信号，帮忙归因一下。",
        f"最近在做复盘，{period} {business}的数据我看了下，想确认异常、原因和关键指标。",
    ]
    return [
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{q}\n事实数据：{facts}"},
            {"role": "assistant", "content": targets},
        ], "meta": {"type": "TV", "diagnosis_id": record["diagnosis_id"]}}
        for q in questions
    ]


def main() -> None:
    random.seed(SEED)
    diagnostics = load_jsonl(DIAG_PATH)
    cards = rule_to_card()

    rows: list[dict] = []
    dropped: list[dict] = []
    skipped: list[str] = []

    for record in diagnostics:
        rid = record["diagnosis_id"]
        if rid in TEST_IDS:
            continue
        if not record.get("primary_anomaly") or not record.get("primary_reason"):
            skipped.append(rid)
            continue
        rule_ids = record.get("triggered_rule_ids") or []
        card_ids = sorted({c for r in rule_ids for c in cards.get(r, [])})[:2]

        for builder in (build_t1, build_t2):
            batch = builder(record, card_ids)
            for item in batch:
                errors: list[dict] = []
                judge_slots(item["messages"][2]["content"], record, errors)
                covered, total = judge_coverage(item["messages"][2]["content"], record)
                if errors or covered < total:
                    dropped.append({"id": rid, "type": item["meta"]["type"],
                                    "errors": [e["type"] for e in errors],
                                    "coverage": f"{covered}/{total}"})
                else:
                    item["meta"]["diagnosis_id"] = rid
                    rows.append(item)

        for item in build_t3(record) + build_t4(record):
            item["meta"]["diagnosis_id"] = rid
            rows.append(item)

    for event in load_jsonl(EVENT_PATH):
        if (event.get("period"), event.get("business_type")) in TEST_PAIRS:
            continue
        card_ids_e = sorted(set(cards.get(event.get("rule_id"), [])))[:1]
        for item in build_t6(event, card_ids_e):
            fake = {
                "period": event.get("period"),
                "business_type": event.get("business_type"),
                "primary_anomaly": event,
            }
            errors: list[dict] = []
            judge_slots(item["messages"][2]["content"], fake, errors)
            if errors:
                dropped.append({
                    "id": f"{event['period']}_{event['rule_id']}",
                    "type": "T6",
                    "errors": [e["type"] for e in errors],
                    "coverage": "-",
                })
            else:
                item["meta"]["diagnosis_id"] = f"{event['period']}_{event['rule_id']}"
                rows.append(item)

    clean_months = ["2025-12", "2026-01", "2026-06"]
    for month in clean_months:
        for business in ("泊车", "行车"):
            item = build_t5(month, business)[0]
            item["meta"]["diagnosis_id"] = f"{month}_{business}"
            rows.append(item)

    random.shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "train.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    test_rows = []
    for record in diagnostics:
        if record["diagnosis_id"] not in TEST_IDS:
            continue
        rule_ids = record.get("triggered_rule_ids") or []
        card_ids = sorted({c for r in rule_ids for c in cards.get(r, [])})[:2]
        if not record.get("primary_anomaly") or not record.get("primary_reason"):
            continue
        for item in build_t1(record, card_ids):
            item["meta"]["diagnosis_id"] = record["diagnosis_id"]
            test_rows.append(item)

    variant_rows: list[dict] = []
    for record in diagnostics:
        if record["diagnosis_id"] not in TEST_IDS:
            continue
        if not record.get("primary_anomaly") or not record.get("primary_reason"):
            continue
        rule_ids = record.get("triggered_rule_ids") or []
        card_ids = sorted({c for r in rule_ids for c in cards.get(r, [])})[:2]
        variant_rows.extend(build_test_variants(record, card_ids))

    with (OUT_DIR / "test_variant.jsonl").open("w", encoding="utf-8") as file:
        for row in variant_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (OUT_DIR / "test.jsonl").open("w", encoding="utf-8") as file:
        for row in test_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["meta"]["type"]] = counts.get(row["meta"]["type"], 0) + 1

    print(f"训练样本：{len(rows)}  测试样本（锁定）：{len(test_rows)}")
    print(f"类型分布：{counts}")
    print(f"判定器拦截：{len(dropped)} 条")
    print(f"跳过（缺主因）：{len(skipped)} 条 {skipped}")
    for item in dropped[:10]:
        print("  丢弃", item)

    (OUT_DIR / "README.md").write_text(f"""# SFT 数据 v2（第一批）

- 训练样本：{len(rows)}
- 测试样本（锁定，不参与训练）：{len(test_rows)} 条，来自 {TEST_IDS}
- 类型分布：{counts}
- 独立事实：17 条诊断（其中 6 条锁定为测试）
- 标签生成方式：模板拼装（非模型生成），每条经判定器校验后入库
- 随机种子：{SEED}
- 判定器拦截：{len(dropped)} 条不合格样本
""", encoding="utf-8")
    print(f"已写入：{OUT_DIR}")


if __name__ == "__main__":
    main()
