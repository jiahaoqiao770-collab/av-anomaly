#!/usr/bin/env python3
"""生成 2026 年 6 月至 11 月的合成财务数据。

运行：
    cd ~/projects/av-anomaly
    conda activate av
    python scripts/generate_finance.py

输出：
    data/synthetic/财务.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "财务.csv"

COLUMNS = [
    "时间",
    "外包服务费(万)",
    "车辆运维费(万)",
    "硬盘邮寄(万)",
    "场地测试(新增)(万)",
]

# 每月总费用区间。8~11 月整体高于 6~7 月。
# 每月总费用区间（万元）。整体上升，年末最高。
TOTAL_COST_RANGES = {
    "2025-12": (128.0, 142.0),
    "2026-01": (118.0, 132.0),
    "2026-02": (110.0, 124.0),
    "2026-03": (134.0, 148.0),
    "2026-04": (140.0, 154.0),
    "2026-05": (150.0, 166.0),
    "2026-06": (150.0, 168.0),
    "2026-07": (170.0, 192.0),
    "2026-08": (195.0, 218.0),
    "2026-09": (205.0, 232.0),
    "2026-10": (218.0, 246.0),
    "2026-11": (225.0, 248.0),
}

def split_total_cost(total: float, rng: random.Random) -> tuple[float, ...]:
    """把月度总费用拆为四项，并保证四项之和等于总费用。"""
    outsourced = round(total * rng.uniform(0.82, 0.86), 1)
    vehicle = round(total * rng.uniform(0.09, 0.12), 1)
    shipping = round(total * rng.uniform(0.010, 0.018), 1)
    site = round(total - outsourced - vehicle - shipping, 1)

    if min(outsourced, vehicle, shipping, site) < 0:
        raise AssertionError("费用拆分出现负数")

    return outsourced, vehicle, shipping, site


def build_rows() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []

    for month_key, (low, high) in TOTAL_COST_RANGES.items():
        year, month = map(int, month_key.split("-"))
        total = round(rng.uniform(low, high), 1)
        outsourced, vehicle, shipping, site = split_total_cost(total, rng)
        rows.append({
            "时间": f"{year}/{month}/1",
            "外包服务费(万)": outsourced,
            "车辆运维费(万)": vehicle,
            "硬盘邮寄(万)": shipping,
            "场地测试(新增)(万)": site,
        })

    return rows
    for month in range(7, 12):
        low, high = TOTAL_COST_RANGES[month]
        total = round(rng.uniform(low, high), 1)
        outsourced, vehicle, shipping, site = split_total_cost(total, rng)
        rows.append(
            {
                "时间": f"2026/{month}/1",
                "外包服务费(万)": outsourced,
                "车辆运维费(万)": vehicle,
                "硬盘邮寄(万)": shipping,
                "场地测试(新增)(万)": site,
            }
        )

    return rows


def row_total(row: dict[str, object]) -> float:
    return round(
        sum(float(row[column]) for column in COLUMNS[1:]),
        1,
    )


def validate(rows: list[dict[str, object]]) -> None:
    expected_count = len(TOTAL_COST_RANGES)
    if len(rows) != expected_count:
        raise AssertionError(f"财务表必须恰好包含 {expected_count} 个月")
    if list(rows[0]) != COLUMNS:
        raise AssertionError("财务表字段顺序错误")

    expected_dates = [
        f"{int(key.split('-')[0])}/{int(key.split('-')[1])}/1"
        for key in TOTAL_COST_RANGES
    ]
    if [str(row["时间"]) for row in rows] != expected_dates:
        raise AssertionError("月份范围或顺序错误")

    for row in rows:
        if any(float(row[column]) < 0 for column in COLUMNS[1:]):
            raise AssertionError(f"费用不能为负数：{row['时间']}")
        total = row_total(row)
        if not 100.0 <= total <= 250.0:
            raise AssertionError(f"月度总费用不在 100~250 万：{row['时间']}")

    half = expected_count // 2
    early_average = sum(row_total(row) for row in rows[:half]) / half
    later_average = sum(row_total(row) for row in rows[half:]) / half
    if later_average <= early_average:
        raise AssertionError("下半年平均成本必须高于上半年")

def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    validate(rows)
    write_csv(rows)

    print(f"输出文件：{OUTPUT_PATH}")
    print("月份    月度总费用(万)")
    for row in rows:
        print(f"{row['时间']:<10} {row_total(row):>6.1f}")


if __name__ == "__main__":
    main()
