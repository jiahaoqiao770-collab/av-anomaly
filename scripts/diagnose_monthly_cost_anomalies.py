#!/usr/bin/env python3
"""Merge monthly cost anomaly events into preliminary diagnoses."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnosis.monthly_cost_diagnoser import (  # noqa: E402
    DEFAULT_ANOMALY_JSONL_PATH,
    DEFAULT_DIAGNOSIS_CSV_PATH,
    DEFAULT_DIAGNOSIS_JSONL_PATH,
    DEFAULT_DIAGNOSIS_RULES_PATH,
    DEFAULT_METRICS_PATH,
    diagnose_monthly_cost_anomalies,
    write_diagnosis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_ANOMALY_JSONL_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_DIAGNOSIS_RULES_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_DIAGNOSIS_CSV_PATH)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_DIAGNOSIS_JSONL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnoses = diagnose_monthly_cost_anomalies(
        events_path=args.events,
        metrics_path=args.metrics,
        diagnosis_rules_path=args.rules,
    )
    csv_path, jsonl_path = write_diagnosis_outputs(
        diagnoses, args.csv_output, args.jsonl_output
    )
    severity_counts = Counter(item["severity"] for item in diagnoses)

    print(f"诊断结果数：{len(diagnoses)}")
    print(
        f"严重={severity_counts['critical']}，"
        f"警告={severity_counts['warning']}"
    )
    print(f"CSV输出：{csv_path}")
    print(f"JSONL输出：{jsonl_path}")


if __name__ == "__main__":
    main()
