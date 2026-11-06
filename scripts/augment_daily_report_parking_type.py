#!/usr/bin/env python3
"""Append parking-lot type to an existing daily report without changing its source.

The type is read from main_clips.tagsMap entries whose tag2Cn is
"停车场类型". Matching uses the report date and vehicle ID against each
clip's eventTime date and carId. Multiple distinct types are joined with "、".

Run from the project root:
    python scripts/augment_daily_report_parking_type.py

The default output is a new file. Use --overwrite only when intentionally
replacing that output file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS = PROJECT_ROOT / "data/synthetic/main_clips_202606_202611.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data/synthetic/日报表.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/synthetic/日报表_补充停车场类型.csv"
NEW_COLUMN = "停车场类型"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV没有表头：{path}")
        return reader.fieldnames, list(reader)


def parse_date(value: str, *, source: str, line_number: int) -> str:
    value = (value or "").strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"无法解析日期 {value!r}（{source} 第{line_number}行）")


def collect_parking_types(clips_path: Path) -> tuple[dict[tuple[str, str], list[str]], int]:
    fields, clips = read_csv(clips_path)
    required = {"eventTime", "carId", "tripType", "tagsMap"}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"main_clips缺少字段：{sorted(missing)}")

    found: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    tagged_clip_count = 0

    for line_number, clip in enumerate(clips, start=2):
        if (clip.get("tripType") or "").strip().lower() != "parking":
            continue
        date_key = parse_date(clip.get("eventTime", ""), source=str(clips_path), line_number=line_number)
        car_id = (clip.get("carId") or "").strip()
        if not car_id:
            continue

        raw_tags = (clip.get("tagsMap") or "").strip()
        if not raw_tags:
            continue
        try:
            tags_map = json.loads(raw_tags)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tagsMap JSON解析失败（{clips_path} 第{line_number}行）：{exc}") from exc
        if not isinstance(tags_map, dict):
            continue

        key = (date_key, car_id)
        clip_has_type = False
        for tag in tags_map.values():
            if not isinstance(tag, dict) or str(tag.get("tag2Cn") or "").strip() != NEW_COLUMN:
                continue
            value = str(tag.get("tag3Cn") or "").strip()
            if value:
                clip_has_type = True
                if value not in seen[key]:
                    found[key].append(value)
                    seen[key].add(value)
        tagged_clip_count += int(clip_has_type)

    return dict(found), tagged_clip_count


def is_parking_demand(value: str) -> bool:
    return "泊车" in (value or "").strip()


def augment_report(report_path: Path, clips_path: Path, output_path: Path, overwrite: bool) -> None:
    report_fields, rows = read_csv(report_path)
    required = {"日期", "车辆编号（CYS）", "需求类型"}
    missing = required.difference(report_fields)
    if missing:
        raise ValueError(f"日报表缺少字段：{sorted(missing)}")
    if NEW_COLUMN in report_fields:
        raise ValueError(f"日报表已经包含“{NEW_COLUMN}”列，为避免重复追加已停止。")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在：{output_path}。确认要替换时请添加 --overwrite。")

    types_by_vehicle_day, tagged_clip_count = collect_parking_types(clips_path)
    output_fields = [*report_fields, NEW_COLUMN]
    parking_rows = 0
    filled_rows = 0

    for line_number, row in enumerate(rows, start=2):
        value = ""
        if is_parking_demand(row.get("需求类型", "")):
            parking_rows += 1
            date_key = parse_date(row.get("日期", ""), source=str(report_path), line_number=line_number)
            vehicle_id = (row.get("车辆编号（CYS）") or "").strip()
            types = types_by_vehicle_day.get((date_key, vehicle_id), [])
            value = "、".join(types)
            if value:
                filled_rows += 1
        row[NEW_COLUMN] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=output_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    print(f"原日报表：{report_path}")
    print(f"main_clips：{clips_path}")
    print(f"新输出文件：{output_path}")
    print(f"日报原始列数：{len(report_fields)}；输出列数：{len(output_fields)}")
    print(f"日报原始行数：{len(rows)}；输出行数：{len(rows)}")
    print(f"泊车需求行：{parking_rows}；成功填入停车场类型：{filled_rows}")
    print(f"含停车场类型标签的Clip数：{tagged_clip_count}")
    print("行车需求行的停车场类型保持空白。")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS, help="main_clips CSV路径")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="原日报表CSV路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="新增列后的输出路径")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    augment_report(args.report, args.clips, args.output, args.overwrite)


if __name__ == "__main__":
    main()
