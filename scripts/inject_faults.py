#!/usr/bin/env python3
"""按剧本把故障注入到现有合成数据，产出对照数据集与真值表。

运行：
    python scripts/inject_faults.py --scenario config/scenarios/faulty_v1.json

读 data/synthetic 下现有三张表，写 data/injected 下：
    财务.csv / main_clips.csv / 日报表.csv / fault_ground_truth.csv

本版只支持两类注入：
    1. cost_multiplier != 1.0 的月份：财务四项费用同乘倍率
    2. trial_ratio 偏离现状的月份：把一部分正式采集改成试采（行数不变）
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = PROJECT_ROOT / "data/synthetic"
OUTPUT_DIR = PROJECT_ROOT / "data/injected"

DEFAULT_SCENARIO = PROJECT_ROOT / "config/scenarios/faulty_v1.json"

FINANCE_SOURCE = SYNTHETIC_DIR / "财务.csv"
CLIPS_SOURCE = SYNTHETIC_DIR / "main_clips_202606_202611.csv"
DAILY_SOURCE = SYNTHETIC_DIR / "日报表.csv"

FINANCE_COST_COLUMNS = [
    "外包服务费(万)",
    "车辆运维费(万)",
    "硬盘邮寄(万)",
    "场地测试(新增)(万)",
]

SUPPORTED_FAULT_TYPES = {"cost_spike", "trial_spike"}
BASELINE_TRIAL_RATIO = 0.15
SEED = 42


def load_scenario(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到剧本：{path}")
    with path.open("r", encoding="utf-8") as file:
        scenario = json.load(file)
    months = scenario.get("months")
    if not isinstance(months, dict) or not months:
        raise ValueError(f"剧本缺少 months：{path}")
    return scenario


def month_of(value: str) -> str:
    """'2026/1/1' 或 '2026/1/1 8:00' -> '2026-01'。"""
    date_part = value.split(" ")[0]
    year, month, _ = date_part.split("/")
    return f"{year}-{int(month):02d}"


def inject_finance(scenario: dict, output_path: Path) -> dict[str, float]:
    """按 cost_multiplier 调整财务表；只改倍率不为 1 的月份。"""
    multipliers = {
        month: float(config.get("cost_multiplier", 1.0))
        for month, config in scenario["months"].items()
    }

    with FINANCE_SOURCE.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        expected = ["时间", *FINANCE_COST_COLUMNS]
        if fieldnames != expected:
            raise ValueError(f"财务表字段与预期不符：{fieldnames}")
        rows = list(reader)

    changed: dict[str, float] = {}
    for row in rows:
        month = month_of(row["时间"])
        factor = multipliers.get(month, 1.0)
        if factor == 1.0:
            continue
        for column in FINANCE_COST_COLUMNS:
            row[column] = round(float(row[column]) * factor, 1)
        changed[month] = factor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=expected, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)

    return changed


def flush_month(
    writer: csv.DictWriter,
    rows: list[dict],
    month: str,
    targets: dict[str, float],
    rng: random.Random,
) -> dict[str, int] | None:
    """处理一个月的数据再写出；返回改动摘要（未改动则返回 None）。"""
    if month not in targets:
        writer.writerows(rows)
        return None
    total = len(rows)
    target_ratio = targets.get(month, BASELINE_TRIAL_RATIO)
    target_trial = round(total * target_ratio)

    official_index = [
        index
        for index, row in enumerate(rows)
        if row["collectionStatus"] == "正式采集"
    ]
    trial_before = total - len(official_index)
    need = target_trial - trial_before

    if need > 0 and official_index:
        picked = rng.sample(official_index, min(need, len(official_index)))
        for index in picked:
            rows[index]["collectionStatus"] = "试采"

    writer.writerows(rows)

    if need <= 0:
        return None
    return {
        "total": total,
        "trial_before": trial_before,
        "trial_after": target_trial,
    }


def inject_trial_ratio(
    scenario: dict,
    output_path: Path,
) -> dict[str, dict[str, int]]:
    """按 trial_ratio 调整试采占比；不增删行，只改 collectionStatus。"""
    ratios = {
        month: float(config.get("trial_ratio", BASELINE_TRIAL_RATIO))
        for month, config in scenario["months"].items()
    }
    # 只处理相对基线有变化的月份，其余月份原样写出、一条不改。
    targets = {
        month: ratio
        for month, ratio in ratios.items()
        if abs(ratio - BASELINE_TRIAL_RATIO) > 1e-9
    }
    rng = random.Random(SEED)
    applied: dict[str, dict[str, int]] = {}

    with CLIPS_SOURCE.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as destination:
            writer = csv.DictWriter(
                destination, fieldnames=fieldnames, extrasaction="raise"
            )
            writer.writeheader()

            buffer: list[dict] = []
            current: str | None = None

            for row in reader:
                month = month_of(row["eventTime"])
                if current is None:
                    current = month
                if month != current:
                    summary = flush_month(
                        writer, buffer, current, targets, rng
                    )
                    if summary:
                        applied[current] = summary
                    buffer = []
                    current = month
                buffer.append(row)

            if buffer and current is not None:
                summary = flush_month(writer, buffer, current, targets, rng)
                if summary:
                    applied[current] = summary

    return applied


def write_ground_truth(scenario: dict, output_path: Path) -> list[dict]:
    """把剧本里的 injected_faults 抄成一张表，标注本版是否支持。"""
    columns = [
        "fault_id",
        "month",
        "type",
        "detail",
        "supported",
        "expected_rules",
        "tolerated_rules",
        "note",
    ]
    faults = scenario.get("injected_faults", [])
    rows = []
    for fault in faults:
        rows.append(
            {
                "fault_id": fault["fault_id"],
                "month": fault["month"],
                "type": fault["type"],
                "detail": fault.get("detail", ""),
                "supported": fault["type"] in SUPPORTED_FAULT_TYPES,
                "expected_rules": "|".join(fault.get("expected_rules", [])),
                "tolerated_rules": "|".join(
                    fault.get("tolerated_rules", [])
                ),
                "note": fault.get("note", ""),
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="剧本 JSON 路径，默认 config/scenarios/faulty_v1.json",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"剧本：{scenario['scenario_id']}（{args.scenario}）")

    finance_changed = inject_finance(scenario, OUTPUT_DIR / "财务.csv")
    print(f"财务成本注入：{finance_changed or '无'}")

    trial_applied = inject_trial_ratio(
        scenario, OUTPUT_DIR / "main_clips.csv"
    )
    print(f"试采占比注入：{trial_applied or '无'}")

    shutil.copyfile(DAILY_SOURCE, OUTPUT_DIR / "日报表.csv")
    print("日报表：原样复制（行数与工时不受注入影响）")

    ground_truth = write_ground_truth(
        scenario, OUTPUT_DIR / "fault_ground_truth.csv"
    )
    supported = [row for row in ground_truth if row["supported"]]
    skipped = [row for row in ground_truth if not row["supported"]]
    print(f"真值：共 {len(ground_truth)} 条，本版支持 {len(supported)} 条")
    for row in skipped:
        print(f"  跳过（未支持）：{row['fault_id']} {row['type']}")

    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()