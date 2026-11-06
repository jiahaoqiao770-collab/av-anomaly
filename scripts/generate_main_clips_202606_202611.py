#!/usr/bin/env python3
"""生成 2026-06 至 2026-11 的 main_clips 全量合成数据。

运行：
    python scripts/generate_main_clips_202606_202611.py

默认不会覆盖同名输出。确需重新生成时：
    python scripts/generate_main_clips_202606_202611.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import scripts.generate_data_clips_sample100 as base


SEED = 42
PARKING_RATIO = 0.85
TRIAL_RATIO = 0.15

# 固定月度行数，总计 36,220 行，可复现且落在 35,000~40,000 范围内。
MONTHLY_COUNTS = {
    "2026-06": 7619,
    "2026-07": 5456,
    "2026-08": 5102,
    "2026-09": 6126,
    "2026-10": 6003,
    "2026-11": 5914,
}

OUTPUT_PATH = base.OUTPUT_DIR / "main_clips_202606_202611.csv"

# 16681 来自真实 main_clips 样例；16682 是本项目专用的合成扩展标签。
TRIAL_TAG = base.Tag(
    tag_index=16682,
    tag1_cn="采集状态",
    tag1_en="collection_status",
    tag2_cn="试采",
    tag2_en="Trial_collect",
    tag3_cn=None,
    tag3_en=None,
    tag4_cn=None,
    tag4_en=None,
    tag5_cn=None,
    tag5_en=None,
    tag_type=4,
    source="本项目合成扩展标签",
)


def replace_official_with_trial(row: dict[str, object]) -> None:
    """把一行中的正式采集 16681 替换成试采 16682。"""
    tags_map = json.loads(str(row["tagsMap"]))
    replaced: dict[str, dict] = {}

    for key, value in tags_map.items():
        if key == "16681":
            replaced["16682"] = TRIAL_TAG.as_tags_map_value()
        else:
            replaced[key] = value

    if "16682" not in replaced:
        raise AssertionError("该行没有找到可替换的正式采集标签 16681")

    row["tagsMap"] = json.dumps(
        replaced,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row["tags"] = "'" + ",".join(replaced.keys())


def make_exact_flags(
    total: int,
    positive_count: int,
    rng: random.Random,
) -> list[bool]:
    """生成数量精确、位置随机的布尔标记。"""
    flags = [True] * positive_count
    flags.extend([False] * (total - positive_count))
    rng.shuffle(flags)
    return flags


def generate_rows() -> tuple[
    list[dict[str, object]],
    dict[int, base.Tag],
]:
    rng = random.Random(SEED)

    dictionary, tags = base.load_dictionary()
    if TRIAL_TAG.tag_index in dictionary:
        raise ValueError(f"标签 ID {TRIAL_TAG.tag_index} 已在映射表中存在")

    dictionary[TRIAL_TAG.tag_index] = TRIAL_TAG
    tags.append(TRIAL_TAG)
    groups = base.prepare_tag_groups(tags)

    # 先只生成时间和车辆，tripType 与采集状态再按精确比例分配。
    basic_specs: list[tuple[object, str]] = []
    for month_key, count in MONTHLY_COUNTS.items():
        year, month = map(int, month_key.split("-"))
        for _ in range(count):
            local_time = base.random_local_datetime(year, month, rng)
            car_id = rng.choice(base.CAR_IDS)
            basic_specs.append((local_time, car_id))

    total = len(basic_specs)
    parking_count = round(total * PARKING_RATIO)
    trial_count = round(total * TRIAL_RATIO)

    parking_flags = make_exact_flags(total, parking_count, rng)
    trial_flags = make_exact_flags(total, trial_count, rng)

    specifications = []
    for index, (local_time, car_id) in enumerate(basic_specs):
        trip_type = "parking" if parking_flags[index] else "driving"
        specifications.append(
            (local_time, car_id, trip_type, trial_flags[index])
        )

    specifications.sort(
        key=lambda item: (item[0], item[1], item[2], item[3])
    )

    rows: list[dict[str, object]] = []
    for index, (local_time, car_id, trip_type, is_trial) in enumerate(
        specifications
    ):
        row = base.build_row(
            row_id=base.START_ID + index,
            local_time=local_time,
            car_id=car_id,
            trip_type=trip_type,
            groups=groups,
            rng=rng,
        )
        if is_trial:
            replace_official_with_trial(row)
        rows.append(row)

    return rows, dictionary


def validate_exact_distribution(
    rows: list[dict[str, object]],
) -> dict[str, int]:
    """补充检查总量、tripType 和采集状态的精确数量。"""
    total = sum(MONTHLY_COUNTS.values())
    expected_parking = round(total * PARKING_RATIO)
    expected_trial = round(total * TRIAL_RATIO)

    trip_counts = Counter(str(row["tripType"]) for row in rows)
    status_counts: Counter[str] = Counter()

    for row in rows:
        tags_map = json.loads(str(row["tagsMap"]))
        if "16681" in tags_map:
            status_counts["official"] += 1
        if "16682" in tags_map:
            status_counts["trial"] += 1

    assert len(rows) == total
    assert trip_counts["parking"] == expected_parking
    assert trip_counts["driving"] == total - expected_parking
    assert status_counts["trial"] == expected_trial
    assert status_counts["official"] == total - expected_trial

    return {
        "total": total,
        "parking": trip_counts["parking"],
        "driving": trip_counts["driving"],
        "official": status_counts["official"],
        "trial": status_counts["trial"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的同名全量 CSV",
    )
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出文件已存在：{OUTPUT_PATH}\n"
            "如确认覆盖，请加参数 --overwrite"
        )

    rows, dictionary = generate_rows()

    # 复用原脚本的字段、时间、路径、标签和月份校验。
    base.validate_rows(rows, MONTHLY_COUNTS, dictionary)
    summary = validate_exact_distribution(rows)

    # 原脚本使用 utf-8-sig，生成的 CSV 可直接用 Excel 打开中文。
    base.write_csv(rows, OUTPUT_PATH)

    print(f"输出文件：{OUTPUT_PATH}")
    print(f"总行数：{summary['total']}")
    print(
        f"tripType：parking={summary['parking']}，"
        f"driving={summary['driving']}"
    )
    print(
        f"采集状态：正式采集={summary['official']}，"
        f"试采={summary['trial']}"
    )
    print("各月行数：")
    for month, count in MONTHLY_COUNTS.items():
        print(f"  {month}: {count}")


if __name__ == "__main__":
    main()
