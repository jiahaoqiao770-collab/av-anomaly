#!/usr/bin/env python3
"""生成 2025-12 至 2026-11 的 main_clips 全量合成数据（12 个月）。

运行：
    python -m scripts.generate_main_clips_202606_202611

默认不会覆盖同名输出。确需重新生成时：
    python -m scripts.generate_main_clips_202606_202611 --overwrite

逐行产出、按批写入，内存占用与总行数无关；写成功后才替换目标文件。
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter

import scripts.generate_data_clips_sample100 as base


SEED = 42
PARKING_RATIO = 0.85
TRIAL_RATIO = 0.15

# 每攒够这么多行写一次盘，用来控制内存峰值。
BATCH_SIZE = 20_000

# 固定月度行数，总计 71,870 行（12 个月），可复现。
# 固定月度行数，总计 1,200,000 行（12 个月），可复现。
MONTHLY_COUNTS = {
    "2025-12": 98000,
    "2026-01": 102000,
    "2026-02": 76000,
    "2026-03": 105000,
    "2026-04": 101000,
    "2026-05": 104000,
    "2026-06": 111000,
    "2026-07": 99000,
    "2026-08": 95000,
    "2026-09": 103000,
    "2026-10": 106000,
    "2026-11": 100000,
}

OUTPUT_PATH = base.OUTPUT_DIR / "main_clips_202606_202611.csv"
PART_PATH = OUTPUT_PATH.parent / (OUTPUT_PATH.name + ".part")


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


def generate_rows():
    """逐行产出 Clip 记录；调用方负责写入，函数本身不缓存。"""
    rng = random.Random(SEED)

    dictionary, tags = base.load_dictionary()
    groups = base.prepare_tag_groups(tags)

    # 先只生成时间和车辆，tripType 与采集状态再按精确比例分配。
    specifications: list[tuple[object, str, str, bool]] = []
    for month_key, count in MONTHLY_COUNTS.items():
        year, month = map(int, month_key.split("-"))
        for _ in range(count):
            local_time = base.random_local_datetime(year, month, rng)
            car_id = rng.choice(base.CAR_IDS)
            specifications.append((local_time, car_id, "", False))

    total = len(specifications)
    parking_count = round(total * PARKING_RATIO)
    trial_count = round(total * TRIAL_RATIO)

    parking_flags = make_exact_flags(total, parking_count, rng)
    trial_flags = make_exact_flags(total, trial_count, rng)

    # 原地补上 tripType 和采集状态，避免再复制一份大列表。
    for index in range(total):
        local_time, car_id, _, _ = specifications[index]
        trip_type = "parking" if parking_flags[index] else "driving"
        specifications[index] = (
            local_time,
            car_id,
            trip_type,
            trial_flags[index],
        )

    specifications.sort(
        key=lambda item: (item[0], item[1], item[2], item[3])
    )

    for index, (local_time, car_id, trip_type, is_trial) in enumerate(
        specifications
    ):
        yield base.build_row(
            row_id=base.START_ID + index,
            local_time=local_time,
            car_id=car_id,
            trip_type=trip_type,
            groups=groups,
            rng=rng,
            collection_status="试采" if is_trial else "正式采集",
        )


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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    trip_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    month_counts: Counter[str] = Counter()
    expected_id = base.START_ID
    batch: list[dict[str, object]] = []

    # 先写 .part，全部校验通过后再原子替换，避免失败时留下半个文件。
    with PART_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=base.CSV_COLUMNS,
            extrasaction="raise",
        )
        writer.writeheader()

        for row in generate_rows():
            base.check_row(row)

            assert int(row["id"]) == expected_id, "id 不是连续自增"
            expected_id += 1

            trip_counts[str(row["tripType"])] += 1
            status_counts[
                "trial" if row["collectionStatus"] == "试采" else "official"
            ] += 1
            year, month, _ = str(row["eventTime"]).split(" ")[0].split("/")
            month_counts[f"{year}-{int(month):02d}"] += 1

            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                writer.writerows(batch)
                batch.clear()

        if batch:
            writer.writerows(batch)
            batch.clear()

    total = sum(month_counts.values())
    expected_total = sum(MONTHLY_COUNTS.values())
    expected_parking = round(expected_total * PARKING_RATIO)
    expected_trial = round(expected_total * TRIAL_RATIO)

    assert total == expected_total, f"总行数不符：{total} != {expected_total}"
    assert dict(month_counts) == MONTHLY_COUNTS, (
        f"月度行数不符：{dict(month_counts)}"
    )
    assert trip_counts["parking"] == expected_parking, "parking 数量不符"
    assert trip_counts["driving"] == expected_total - expected_parking, (
        "driving 数量不符"
    )
    assert status_counts["trial"] == expected_trial, "试采数量不符"
    assert status_counts["official"] == expected_total - expected_trial, (
        "正式采集数量不符"
    )

    os.replace(PART_PATH, OUTPUT_PATH)

    print(f"输出文件：{OUTPUT_PATH}")
    print(f"总行数：{total}")
    print(
        f"tripType：parking={trip_counts['parking']}，"
        f"driving={trip_counts['driving']}"
    )
    print(
        f"采集状态：正式采集={status_counts['official']}，"
        f"试采={status_counts['trial']}"
    )
    print("各月行数：")
    for month, count in MONTHLY_COUNTS.items():
        print(f"  {month}: {count}")


if __name__ == "__main__":
    main()
