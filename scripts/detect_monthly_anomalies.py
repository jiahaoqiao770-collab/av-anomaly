#!/usr/bin/env python3
"""Run monthly Clip cost-efficiency anomaly rules."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.monthly_cost_anomaly_detector import (  # noqa: E402
    DEFAULT_ANOMALY_CSV_PATH,
    DEFAULT_ANOMALY_JSONL_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_RULES_PATH,
    detect_anomalies,
    write_anomaly_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_ANOMALY_CSV_PATH)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_ANOMALY_JSONL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = detect_anomalies(args.metrics, args.rules)
    csv_path, jsonl_path = write_anomaly_outputs(
        events, args.csv_output, args.jsonl_output
    )
    severity_counts = Counter(event["severity"] for event in events)
    month_counts = Counter(event["period"] for event in events)

    print(f"异常事件数：{len(events)}")
    print(
        f"严重={severity_counts['critical']}，"
        f"警告={severity_counts['warning']}"
    )
    print("按月统计：")
    for month in sorted(month_counts):
        print(f"  {month}: {month_counts[month]}")
    print(f"CSV输出：{csv_path}")
    print(f"JSONL输出：{jsonl_path}")


if __name__ == "__main__":
    main()
