#!/usr/bin/env python3
"""评估故障注入的召回与副作用。

运行：
    python scripts/evaluate_faults.py --scenario config/scenarios/faulty_v1.json

读：
    config/scenarios/faulty_v1.json                        真值（权威来源）
    data/results/baseline/monthly_cost_anomaly_events.csv  对照组事件
    data/results/faulty_v1/anomaly_events.csv              故障组事件

写：
    data/results/evaluation.csv
    data/results/evaluation.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data/results"

DEFAULT_SCENARIO = PROJECT_ROOT / "config/scenarios/faulty_v1.json"

BASELINE_EVENTS = RESULTS_DIR / "baseline/monthly_cost_anomaly_events.csv"
FAULTY_EVENTS = RESULTS_DIR / "faulty_v1/anomaly_events.csv"
GROUND_TRUTH = PROJECT_ROOT / "data/injected/fault_ground_truth.csv"

# 注入月之后还要看一个月：抬高基数会让下个月的环比失真。
BUFFER_MONTHS = 2

def load_scenario(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到剧本：{path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_events(path: Path) -> dict[str, Counter]:
    """读取事件 CSV，返回 {月份: Counter(规则ID -> 次数)}。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到事件文件：{path}")

    events: dict[str, Counter] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            month = row["月份"]
            events.setdefault(month, Counter())[row["规则ID"]] += 1
    return events

def load_ground_truth(path: Path) -> list[dict]:
    """读注入器产出的真值表，保留 supported 标记。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到真值表：{path}")

    faults: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            faults.append(
                {
                    "fault_id": row["fault_id"],
                    "month": row["month"],
                    "type": row["type"],
                    "supported": row["supported"].strip().lower() == "true",
                    "expected_rules": [
                        item
                        for item in row["expected_rules"].split("|")
                        if item
                    ],
                    "tolerated_rules": [
                        item
                        for item in row["tolerated_rules"].split("|")
                        if item
                    ],
                }
            )
    return faults


def build_affected(months: list[str], faults: list[dict]) -> set[str]:
    """注入月 + 后续 BUFFER_MONTHS 个月，都算受影响区间。"""
    order = {month: index for index, month in enumerate(months)}
    affected: set[str] = set()
    for fault in faults:
        if not fault.get("supported", True):
            continue
        month = fault["month"]
        affected.add(month)
        start = order.get(month)
        if start is None:
            continue
        for offset in range(1, BUFFER_MONTHS + 1):
            if start + offset < len(months):
                affected.add(months[start + offset])
    return affected


def evaluate(scenario: dict) -> tuple[list[dict], list[dict], dict]:
    months = sorted(scenario["months"])
    faults = load_ground_truth(GROUND_TRUTH)

    baseline = load_events(BASELINE_EVENTS)
    faulty = load_events(FAULTY_EVENTS)
    affected = build_affected(months, faults)

    rows: list[dict] = []
    for fault in faults:
        month = fault["month"]
        expected = set(fault.get("expected_rules", []))
        tolerated = set(fault.get("tolerated_rules", []))
        fired = set(faulty.get(month, Counter()))

        hit = expected & fired
        missed = expected - fired
        tolerated_fired = tolerated & fired
        unexpected = fired - expected - tolerated

        if not fault.get("supported", True):
            status = "not_injected"
        elif not expected:
            status = "no_rule_coverage" 
        elif not missed:
            status = "detected"
        else:
            status = "missed"

        rows.append(
            {
                "fault_id": fault["fault_id"],
                "month": month,
                "type": fault["type"],
                "expected": "|".join(sorted(expected)),
                "hit": "|".join(sorted(hit)),
                "missed": "|".join(sorted(missed)),
                "tolerated_fired": "|".join(sorted(tolerated_fired)),
                "unexpected": "|".join(sorted(unexpected)),
                "status": status,
            }
        )

    side_effects: list[dict] = []
    for month in months:
        if month in affected:
            continue
        base_counter = baseline.get(month, Counter())
        fault_counter = faulty.get(month, Counter())
        if base_counter == fault_counter:
            continue
        diff = fault_counter - base_counter
        extra = base_counter - fault_counter
        side_effects.append(
            {
                "month": month,
                "extra_rules": "|".join(sorted(diff)),
                "missing_rules": "|".join(sorted(extra)),
            }
        )

    summary = {
        "months": len(months),
        "faults_total": len(faults),
        "detected": sum(1 for row in rows if row["status"] == "detected"),
        "missed": sum(1 for row in rows if row["status"] == "missed"),
        "no_coverage": sum(
            1 for row in rows if row["status"] == "no_rule_coverage"
        ),
        "not_injected": sum(
            1 for row in rows if row["status"] == "not_injected"
        ),
        "affected_months": sorted(affected),
        "clean_month_side_effects": len(side_effects),
    }
    return rows, side_effects, summary


def write_csv_report(rows: list[dict], output_path: Path) -> None:
    columns = [
        "fault_id",
        "month",
        "type",
        "expected",
        "hit",
        "missed",
        "tolerated_fired",
        "unexpected",
        "status",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    rows: list[dict],
    side_effects: list[dict],
    summary: dict,
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# 故障注入评估报告")
    lines.append("")
    lines.append(f"- 评估月份数：{summary['months']}")
    lines.append(f"- 注入故障数：{summary['faults_total']}")
    lines.append(f"- 检出：{summary['detected']}")
    lines.append(f"- 漏检：{summary['missed']}")
    lines.append(f"- 无规则覆盖：{summary['no_coverage']}")
    lines.append(f"- 未注入（不参与评估）：{summary['not_injected']}")
    lines.append(
        f"- 受影响月份（不计误报）：{'、'.join(summary['affected_months'])}"
    )
    lines.append(
        f"- 干净月份副作用数：{summary['clean_month_side_effects']}"
    )
    lines.append("")
    lines.append("## 逐条故障")
    lines.append("")
    lines.append("| 故障 | 月份 | 类型 | 应报 | 实报 | 漏 | 连带 | 意外 | 结论 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| {fault_id} | {month} | {type} | {expected} | {hit} | "
            "{missed} | {tolerated_fired} | {unexpected} | {status} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## 干净月份副作用")
    lines.append("")
    if side_effects:
        lines.append("| 月份 | 多出的规则 | 消失的规则 |")
        lines.append("|---|---|---|")
        for item in side_effects:
            lines.append(
                f"| {item['month']} | {item['extra_rules']} | "
                f"{item['missing_rules']} |"
            )
    else:
        lines.append("无。干净月份的事件与对照组完全一致。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="剧本 JSON 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="报告输出目录",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    rows, side_effects, summary = evaluate(scenario)

    csv_path = args.output_dir / "evaluation.csv"
    md_path = args.output_dir / "evaluation.md"
    write_csv_report(rows, csv_path)
    write_markdown_report(rows, side_effects, summary, md_path)

    print(f"剧本：{scenario['scenario_id']}")
    print(f"评估月份：{summary['months']}，注入故障：{summary['faults_total']}")
    print(
        f"检出 {summary['detected']}，漏检 {summary['missed']}，"
        f"无规则覆盖 {summary['no_coverage']}"
    )
    print(f"受影响月份：{'、'.join(summary['affected_months'])}")
    print(f"干净月份副作用：{summary['clean_month_side_effects']}")
    print(f"报告：{md_path}")


if __name__ == "__main__":
    main()