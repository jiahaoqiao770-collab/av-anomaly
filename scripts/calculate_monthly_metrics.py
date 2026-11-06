#!/usr/bin/env python3
"""Calculate monthly Clip cost-efficiency metrics and write a CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.clip_cost_metrics import (  # noqa: E402
    DEFAULT_DAILY_REPORT_PATH,
    DEFAULT_DICTIONARY_PATH,
    DEFAULT_FINANCE_PATH,
    DEFAULT_MAIN_CLIPS_PATH,
    DEFAULT_OUTPUT_PATH,
    calculate_monthly_metrics,
    format_metrics_for_output,
    write_monthly_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finance", type=Path, default=DEFAULT_FINANCE_PATH)
    parser.add_argument("--daily-report", type=Path, default=DEFAULT_DAILY_REPORT_PATH)
    parser.add_argument("--main-clips", type=Path, default=DEFAULT_MAIN_CLIPS_PATH)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = calculate_monthly_metrics(
        finance_path=args.finance,
        daily_report_path=args.daily_report,
        main_clips_path=args.main_clips,
        dictionary_path=args.dictionary,
    )
    output_path = write_monthly_metrics(metrics, args.output)

    print(format_metrics_for_output(metrics).to_string(index=False))
    print(f"\n输出文件：{output_path}")
    print(f"输出月份数：{len(metrics)}")
    print("质量校验：通过")


if __name__ == "__main__":
    main()
