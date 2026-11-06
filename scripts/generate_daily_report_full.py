#!/usr/bin/env python3
"""根据全量 main_clips 生成全量日报表。

运行：
    cd ~/projects/av-anomaly
    conda activate av
    python scripts/generate_daily_report_full.py

日报粒度为：日期 + 车辆 + tripType。
因此同一天同一辆车同时有 parking 和 driving 时，会拆成两行，
确保泊车次数和行车数量分别可以与 main_clips 严格对账。
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


SEED = 42
PROJECT_ROOT = Path("/home/qiaojiahao/projects/av-anomaly")
INPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "main_clips_202606_202611.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "日报表.csv"

COLUMNS = [
    "日期", "数采工程师", "省/市/区", "车辆编号（CYS）", "班次",
    "任务类型", "任务状态", "需求类型", "实际开始时间", "实际结束时间",
    "工作时长", "是否有异常情况影响", "异常分类", "开始里程", "结束里程",
    "有效采集时长", "数据量(G)", "每日有效里程", "地理位置", "城市",
    "所属公司", "泊车次数（泊车）", "当日标签总数（行车）", "已发出班次数",
]

ENGINEERS = [f"采集员{i:03d}" for i in range(1, 31)]
COMPANIES = ["四维", "全道"]
TASK_TYPES = ["泊车", "专项采集", "导航组", "常规采集"]
LOCATIONS = [
    ("广东省/深圳市/宝安区", "深圳市", "宝安示例停车场"),
    ("广东省/深圳市/南山区", "深圳市", "南山示例园区"),
    ("广东省/广州市/黄埔区", "广州市", "黄埔示例停车场"),
    ("上海市/上海市/嘉定区", "上海市", "嘉定示例园区"),
    ("江苏省/苏州市/工业园区", "苏州市", "苏州示例停车场"),
    ("四川省/成都市/龙泉驿区", "成都市", "龙泉示例园区"),
]


@dataclass
class Group:
    date_text: str
    date: datetime
    vehicle_id: str
    trip_type: str
    clip_count: int = 0
    first_event: datetime | None = None
    last_event: datetime | None = None
    anomaly_names: set[str] = field(default_factory=set)


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y/%m/%d %H:%M")


def leaf_name(value: dict) -> str:
    for level in ("tag5Cn", "tag4Cn", "tag3Cn", "tag2Cn"):
        if value.get(level):
            return str(value[level])
    return "质检异常"


def load_groups() -> list[Group]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到全量 Clip：{INPUT_PATH}")

    groups: dict[tuple[str, str, str], Group] = {}
    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"eventTime", "carId", "tripType", "tagsMap"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Clip 缺少字段：{sorted(missing)}")

        for row in reader:
            if row["tripType"] not in {"parking", "driving"}:
                raise ValueError(f"未知 tripType：{row['tripType']}")

            event = parse_time(row["eventTime"])
            date_text = f"{event.year}/{event.month}/{event.day}"
            key = (date_text, row["carId"], row["tripType"])
            if key not in groups:
                groups[key] = Group(
                    date_text=date_text,
                    date=event.replace(hour=0, minute=0),
                    vehicle_id=row["carId"],
                    trip_type=row["tripType"],
                )

            group = groups[key]
            group.clip_count += 1
            group.first_event = min(group.first_event or event, event)
            group.last_event = max(group.last_event or event, event)

            tags_map = json.loads(row["tagsMap"])
            for tag in tags_map.values():
                if tag.get("tag1Cn") == "质检标签":
                    group.anomaly_names.add(leaf_name(tag))

    return sorted(
        groups.values(),
        key=lambda item: (item.date, item.vehicle_id, item.trip_type),
    )


def make_profiles(groups: list[Group], rng: random.Random) -> dict[str, dict]:
    profiles = {}
    for index, vehicle_id in enumerate(sorted({g.vehicle_id for g in groups})):
        profiles[vehicle_id] = {
            "engineer": ENGINEERS[index % len(ENGINEERS)],
            "company": COMPANIES[index % len(COMPANIES)],
            "location": LOCATIONS[index % len(LOCATIONS)],
            "odometer": rng.randint(25_000, 85_000),
        }
    return profiles


def make_row(group: Group, profile: dict, rng: random.Random) -> dict:
    assert group.first_event is not None
    assert group.last_event is not None

    if group.first_event.hour < 11:
        shift, start_hour = "早班", 8
    elif group.first_event.hour < 17:
        shift, start_hour = "中班", 12
    else:
        shift, start_hour = "晚班", 16

    start = group.date.replace(hour=start_hour, minute=rng.randint(0, 30))
    work_hours = round(rng.uniform(7.0, 10.0), 1)
    end = start + timedelta(hours=work_hours)

    # 用户指定有效采集时长约 4~6 小时，且不超过工作时长。
    effective_hours = round(rng.uniform(4.0, min(6.0, work_hours)), 1)

    start_mileage = int(profile["odometer"])
    distance = rng.randint(45, 180)
    end_mileage = start_mileage + distance
    profile["odometer"] = end_mileage
    effective_distance = round(distance * rng.uniform(0.72, 0.95), 1)

    province_city_district, city, location = profile["location"]
    has_anomaly = bool(group.anomaly_names)
    task_type = "泊车" if group.trip_type == "parking" else rng.choice(
        ["导航组", "专项采集", "常规采集"]
    )
    demand_type = "泊车" if group.trip_type == "parking" else "行车"
    dispatched = f"{group.date.month:02d}-{group.date.day:02d}{group.vehicle_id}"

    return {
        "日期": group.date_text,
        "数采工程师": profile["engineer"],
        "省/市/区": province_city_district,
        "车辆编号（CYS）": group.vehicle_id,
        "班次": shift,
        "任务类型": task_type,
        "任务状态": "已发车采集",
        "需求类型": demand_type,
        "实际开始时间": f"{start.hour}:{start.minute:02d}",
        "实际结束时间": f"{end.hour % 24}:{end.minute:02d}",
        "工作时长": work_hours,
        "是否有异常情况影响": "是" if has_anomaly else "否",
        "异常分类": "设备/数据异常" if has_anomaly else "",
        "开始里程": start_mileage,
        "结束里程": end_mileage,
        "有效采集时长": effective_hours,
        "数据量(G)": "",
        "每日有效里程": effective_distance,
        "地理位置": location,
        "城市": city,
        "所属公司": profile["company"],
        "泊车次数（泊车）": group.clip_count if group.trip_type == "parking" else 0,
        "当日标签总数（行车）": group.clip_count if group.trip_type == "driving" else 0,
        "已发出班次数": dispatched,
    }


def build_rows() -> tuple[list[dict], list[Group]]:
    rng = random.Random(SEED)
    groups = load_groups()
    profiles = make_profiles(groups, rng)
    rows = [make_row(group, profiles[group.vehicle_id], rng) for group in groups]
    return rows, groups


def validate(rows: list[dict], groups: list[Group]) -> None:
    if len(rows) != len(groups):
        raise AssertionError("输出行数与 Clip 分组数不一致")

    expected = {
        (g.date_text, g.vehicle_id, g.trip_type): g.clip_count
        for g in groups
    }
    actual_keys = set()
    for row in rows:
        if list(row) != COLUMNS:
            raise AssertionError("字段顺序发生变化")
        demand = row["需求类型"]
        if demand not in {"泊车", "行车"}:
            raise AssertionError(f"需求类型非法：{demand}")
        if not 4.0 <= float(row["有效采集时长"]) <= 6.0:
            raise AssertionError("有效采集时长不在 4~6 小时")

        trip_type = "parking" if demand == "泊车" else "driving"
        key = (row["日期"], row["车辆编号（CYS）"], trip_type)
        if key in actual_keys:
            raise AssertionError(f"日报主键重复：{key}")
        actual_keys.add(key)

        parking_count = int(row["泊车次数（泊车）"])
        driving_count = int(row["当日标签总数（行车）"])
        if parking_count != (expected[key] if trip_type == "parking" else 0):
            raise AssertionError(f"泊车次数对账失败：{key}")
        if driving_count != (expected[key] if trip_type == "driving" else 0):
            raise AssertionError(f"行车数量对账失败：{key}")

    if sum(int(r["泊车次数（泊车）"]) for r in rows) != 30_787:
        raise AssertionError("泊车总数未与 Clip 总数一致")
    if sum(int(r["当日标签总数（行车）"]) for r in rows) != 5_433:
        raise AssertionError("行车总数未与 Clip 总数一致")


def write_csv(rows: list[dict]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, groups = build_rows()
    validate(rows, groups)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    print(f"输入 Clip：{INPUT_PATH}")
    print(f"输出日报：{OUTPUT_PATH}")
    print(f"日报行数：{len(rows)}")
    print("全量对账：parking=30787，driving=5433")
    print("需求类型：仅使用 泊车 / 行车")
    print("有效采集时长：每行 4.0~6.0 小时")


if __name__ == "__main__":
    main()
