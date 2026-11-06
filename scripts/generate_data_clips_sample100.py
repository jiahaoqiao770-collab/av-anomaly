#!/usr/bin/env python3
"""生成 main_clips 合成数据。

默认只生成 100 行确认样本：
    python scripts/generate_main_clips.py

确认后生成全量：
    python scripts/generate_main_clips.py --full
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import random
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET


SEED = 42
START_ID = 157_398_990

PROJECT_ROOT = Path("/home/qiaojiahao/projects/av-anomaly")
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"
MAPPING_DIR = Path("/mnt/d/分析用/映射")

MAPPING_FILES = {
    MAPPING_DIR / "映射2.xlsx": 2,
    MAPPING_DIR / "映射4.xlsx": 4,
    MAPPING_DIR / "映射6.xlsx": 6,
}

CSV_COLUMNS = [
    "id", "fileName", "eventTime", "tripType",
    "collectionStatus", "parkingLotType",
    "status", "vin", "carId", "carName",
    "createTime", "updateTime",
    "triggerName", "tripName", "fileSize", "srcCtime", "srcEtime",
    "duration", "platform", "srcPath", "srcBucket", "unpackPath",
    "unpackBucket", "imgUrl",
]

PARKING_LOT_TYPES = [
    "住宅小区停车场",
    "商场停车场",
    "公司园区停车场",
    "公共设施停车场",
    "高速服务区停车场",
    "路边临时停车",
    "破烂路面临时停车场",
]

BUCKET = "private-gwm-dp-inner-1377196348"
PLATFORM = "ADC3.0L_YR研采"

# 完全合成的车辆编号，不对应真实车辆。
CAR_IDS = [f"YR-S{i:02d}-{10000 + i:05d}" for i in range(1, 19)]

TAG_TYPE_BY_TAG1 = {
    "质检标签": 2,
    "采集状态": 4,
    "本车信息": 0,
    "主车信息": 6,
    # 根据三个映射文件所属标签域补充。
    "泊车采集": 4,
    "环境信息": 6,
    "道路信息": 6,
}

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = (
    "http://schemas.openxmlformats.org/"
    "officeDocument/2006/relationships"
)


@dataclass(frozen=True)
class Tag:
    tag_index: int
    tag1_cn: str | None
    tag1_en: str | None
    tag2_cn: str | None
    tag2_en: str | None
    tag3_cn: str | None
    tag3_en: str | None
    tag4_cn: str | None
    tag4_en: str | None
    tag5_cn: str | None
    tag5_en: str | None
    tag_type: int
    source: str

    @property
    def leaf_cn(self) -> str:
        for value in (
            self.tag5_cn,
            self.tag4_cn,
            self.tag3_cn,
            self.tag2_cn,
            self.tag1_cn,
        ):
            if value:
                return value
        raise ValueError(f"标签 {self.tag_index} 没有中文名称")

    def as_tags_map_value(self) -> dict:
        return {
            "id": None,
            "tagIndex": self.tag_index,
            "tag1En": self.tag1_en,
            "tag1Cn": self.tag1_cn,
            "tag2En": self.tag2_en,
            "tag2Cn": self.tag2_cn,
            "tag3En": self.tag3_en,
            "tag3Cn": self.tag3_cn,
            "tag4En": self.tag4_en,
            "tag4Cn": self.tag4_cn,
            "tag5En": self.tag5_en,
            "tag5Cn": self.tag5_cn,
            "tagType": self.tag_type,
        }


def cell_column_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise ValueError(f"无法解析单元格坐标：{cell_ref}")

    number = 0
    for char in letters.group(0):
        number = number * 26 + ord(char) - ord("A") + 1
    return number


def read_first_sheet(path: Path) -> list[dict[str, str]]:
    """使用标准库读取 xlsx 第一张工作表，不依赖 openpyxl。"""
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []

        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root:
                text = "".join(
                    node.text or ""
                    for node in item.iter(f"{{{NS}}}t")
                )
                shared_strings.append(text)

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_map = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships
        }

        sheets = workbook.find(f"{{{NS}}}sheets")
        if sheets is None or not list(sheets):
            raise ValueError(f"{path} 中没有工作表")

        first_sheet = list(sheets)[0]
        relationship_id = first_sheet.attrib[f"{{{REL_NS}}}id"]
        target = relationship_map[relationship_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        sheet_root = ET.fromstring(archive.read(target))
        raw_rows: list[list[str]] = []

        for row_node in sheet_root.findall(
            f".//{{{NS}}}sheetData/{{{NS}}}row"
        ):
            row = [""] * 13

            for cell in row_node.findall(f"{{{NS}}}c"):
                column = cell_column_number(cell.attrib["r"])
                if column > 13:
                    continue

                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{NS}}}v")

                if cell_type == "inlineStr":
                    inline = cell.find(f"{{{NS}}}is")
                    value = (
                        "".join(
                            node.text or ""
                            for node in inline.iter(f"{{{NS}}}t")
                        )
                        if inline is not None
                        else ""
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text or ""

                row[column - 1] = value.strip()

            raw_rows.append(row)

    if not raw_rows:
        raise ValueError(f"{path} 是空工作簿")

    headers = raw_rows[0]
    return [
        dict(zip(headers, row))
        for row in raw_rows[1:]
        if any(row)
    ]


def blank_to_none(value: str) -> str | None:
    value = value.strip()
    return value if value else None


def load_dictionary() -> tuple[dict[int, Tag], list[Tag]]:
    tags_by_id: dict[int, Tag] = {}
    active_tags: list[Tag] = []

    for path, default_tag_type in MAPPING_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"映射文件不存在：{path}")

        for row in read_first_sheet(path):
            if row["标签状态"] != "使用中":
                continue
            if row["是否删除"] != "未删除":
                continue

            tag1_cn = blank_to_none(row["一级标签中文名"])
            tag_type = TAG_TYPE_BY_TAG1.get(tag1_cn, default_tag_type)

            tag = Tag(
                tag_index=int(row["标签ID"]),
                tag1_cn=tag1_cn,
                tag1_en=blank_to_none(row["一级标签英文名"]),
                tag2_cn=blank_to_none(row["二级标签中文名"]),
                tag2_en=blank_to_none(row["二级标签英文名"]),
                tag3_cn=blank_to_none(row["三级标签中文名"]),
                tag3_en=blank_to_none(row["三级标签英文名"]),
                tag4_cn=blank_to_none(row["四级标签中文名"]),
                tag4_en=blank_to_none(row["四级标签英文名"]),
                tag5_cn=blank_to_none(row["五级标签中文名"]),
                tag5_en=blank_to_none(row["五级标签英文名"]),
                tag_type=tag_type,
                source=path.name,
            )

            if tag.tag_index in tags_by_id:
                raise ValueError(f"活动标签 ID 重复：{tag.tag_index}")

            tags_by_id[tag.tag_index] = tag
            active_tags.append(tag)

    # 三个 Excel 中没有该记录，只能从用户提供的真实样例补充。
    official_tag = Tag(
        tag_index=16681,
        tag1_cn="采集状态",
        tag1_en="collection_status",
        tag2_cn="正式采集",
        tag2_en="Normal_collect",
        tag3_cn=None,
        tag3_en=None,
        tag4_cn=None,
        tag4_en=None,
        tag5_cn=None,
        tag5_en=None,
        tag_type=4,
        source="用户提供的main_clips样例",
    )

    if official_tag.tag_index not in tags_by_id:
        tags_by_id[official_tag.tag_index] = official_tag
        active_tags.append(official_tag)

    return tags_by_id, active_tags


def find_tag(
    tags: list[Tag],
    leaf_cn: str,
    tag1_cn: str | None = None,
    tag2_cn: str | None = None,
) -> Tag:
    matches = [
        tag
        for tag in tags
        if tag.leaf_cn == leaf_cn
        and (tag1_cn is None or tag.tag1_cn == tag1_cn)
        and (tag2_cn is None or tag.tag2_cn == tag2_cn)
    ]

    if len(matches) != 1:
        ids = [tag.tag_index for tag in matches]
        raise ValueError(
            f"标签查找结果不唯一：leaf={leaf_cn!r}, "
            f"tag1={tag1_cn!r}, tag2={tag2_cn!r}, ids={ids}"
        )

    return matches[0]


def format_local_time(value: datetime) -> str:
    return (
        f"{value.year}/{value.month}/{value.day} "
        f"{value.hour}:{value.minute:02d}"
    )


def choose_weather(month: int, rng: random.Random) -> str:
    if month in (6, 7, 8):
        return rng.choices(
            ["晴天", "雨天", "雾霾天"],
            weights=[0.65, 0.30, 0.05],
            k=1,
        )[0]

    if month in (9, 10):
        return rng.choices(
            ["晴天", "雨天", "雾霾天"],
            weights=[0.78, 0.15, 0.07],
            k=1,
        )[0]

    return rng.choices(
        ["晴天", "雨天", "雪天", "雾霾天"],
        weights=[0.76, 0.12, 0.04, 0.08],
        k=1,
    )[0]


def random_local_datetime(
    year: int,
    month: int,
    rng: random.Random,
) -> datetime:
    day = rng.randint(1, calendar.monthrange(year, month)[1])

    # 约 75% 白天，25% 夜晚。均限制在本地 08:00 后，
    # 这样转换成 UTC 后仍然处于同一个自然日。
    if rng.random() < 0.75:
        hour = rng.randint(8, 17)
    else:
        hour = rng.randint(18, 23)

    return datetime(
        year,
        month,
        day,
        hour,
        rng.randint(0, 59),
        rng.randint(0, 59),
    )


def prepare_tag_groups(tags: list[Tag]) -> dict[str, object]:
    parking_types = [
        tag for tag in tags
        if tag.tag1_cn == "泊车采集"
        and tag.tag2_cn == "停车场类型"
    ]
    if not parking_types:
        raise ValueError("字典中没有可用的停车场类型标签")

    quality_anomalies = [
        tag for tag in tags
        if tag.tag1_cn == "质检标签"
        and ("缺失" in tag.leaf_cn or "异常" in tag.leaf_cn)
    ]
    if not quality_anomalies:
        raise ValueError("字典中没有可用的缺失/异常质检标签")

    return {
        "official": find_tag(tags, "正式采集", "采集状态"),
        "day": find_tag(tags, "白天", "环境信息", "当日时段"),
        "night": find_tag(tags, "黑夜", "环境信息", "当日时段"),
        "indoor": find_tag(tags, "室内停车场"),
        "outdoor": find_tag(tags, "室外停车场"),
        "stationary": find_tag(tags, "存在静止", "主车信息", "本车行为"),
        "weather": {
            name: find_tag(tags, name, "泊车采集", "天气")
            for name in ("晴天", "雨天", "雪天", "雾霾天")
        },
        "parking_types": parking_types,
        "quality_anomalies": quality_anomalies,
    }


def build_tags(
    trip_type: str,
    local_time: datetime,
    groups: dict[str, object],
    rng: random.Random,
) -> list[Tag]:
    selected: list[Tag] = []

    # 时段是由时间决定的，不随机抽标签 ID。
    if 8 <= local_time.hour < 18:
        selected.append(groups["day"])
    else:
        selected.append(groups["night"])

    if trip_type == "parking":
        parking_type = rng.choice(groups["parking_types"])
        selected.append(parking_type)

        indoor = rng.random() < 0.25
        selected.append(groups["indoor"] if indoor else groups["outdoor"])

        weather_name = choose_weather(local_time.month, rng)
        selected.append(groups["weather"][weather_name])

    # 当前只有正式采集的完整记录，不能编造试采标签。
    selected.append(groups["official"])

    # 先决定是否发生设备异常，再查异常名称对应的标签。
    if rng.random() < 0.04:
        count = rng.randint(1, 3)
        selected.extend(
            rng.sample(
                groups["quality_anomalies"],
                k=min(count, len(groups["quality_anomalies"])),
            )
        )

    # 可用字典里只有“存在静止”满足给定本车行为示例。
    if rng.random() < 0.25:
        selected.append(groups["stationary"])

    seen: set[int] = set()
    deduplicated: list[Tag] = []
    for tag in selected:
        if tag.tag_index not in seen:
            seen.add(tag.tag_index)
            deduplicated.append(tag)

    return deduplicated


def tags_to_json(tags: list[Tag]) -> str:
    value = {
        str(tag.tag_index): tag.as_tags_map_value()
        for tag in tags
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_row(
    row_id: int,
    local_time: datetime,
    car_id: str,
    trip_type: str,
    groups: dict[str, object],  # 暂时保留：tag 体系下线后删除
    rng: random.Random,
    collection_status: str = "正式采集",
) -> dict[str, object]:

    utc_time = local_time - timedelta(hours=8)
    file_name = (
        f"{car_id}_{utc_time:%Y%m%d_%H%M%S}.All.bag.des"
    )

    trigger_utc = utc_time - timedelta(
        seconds=rng.randint(10 * 60, 60 * 60)
    )
    trip_name = f"{car_id}_{trigger_utc:%Y%m%d_%H%M%S}"

    create_time = local_time + timedelta(
        minutes=rng.randint(4 * 24 * 60, 6 * 24 * 60)
    )
    update_time = create_time + timedelta(
        minutes=rng.randint(0, 2 * 24 * 60)
    )

    
    utc_date = utc_time.strftime("%Y%m%d")
    utc_hour = utc_time.strftime("%H")
    utc_minute = utc_time.strftime("%M")
    parking_lot_type = (
    rng.choice(PARKING_LOT_TYPES) if trip_type == "parking" else ""
    )

    return {
        "id": row_id,
        "fileName": file_name,
        "eventTime": format_local_time(local_time),
        "tripType": trip_type,
        "collectionStatus": collection_status,
        "parkingLotType": (
            rng.choice(PARKING_LOT_TYPES) if trip_type == "parking" else ""
        ),
        "status": "",
        "vin": "",
        "carId": car_id,
        "carName": "",
        "createTime": format_local_time(create_time),
        "updateTime": format_local_time(update_time),
        "triggerName": trip_name,
        "tripName": trip_name,
        "fileSize": rng.randint(6_400_000_000, 7_200_000_000),
        "srcCtime": format_local_time(local_time),
        "srcEtime": format_local_time(local_time + timedelta(seconds=60)),
        "duration": 60,
        "platform": PLATFORM,
        "srcPath": (
            f"source/2/{utc_date}/{utc_hour}/{utc_minute}/"
            f"{car_id}/21/{file_name}"
        ),
        "srcBucket": BUCKET,
        "unpackPath": (
            f"unpack/2/{utc_date}/{utc_hour}/{utc_minute}/"
            f"{car_id}/21/{car_id}_{utc_time:%Y%m%d_%H%M%S}"
        ),
        "unpackBucket": BUCKET,
        "imgUrl": "",
    }


def generate_rows() -> tuple[list[dict[str, object]], dict[str, int], dict[int, Tag]]:
    rng = random.Random(SEED)
    dictionary, tags = load_dictionary()
    groups = prepare_tag_groups(tags)

    monthly_counts = {
        f"2026-{month:02d}": rng.randint(5000, 8000)
        for month in range(6, 12)
    }

    specifications: list[tuple[datetime, str, str]] = []

    for month in range(6, 12):
        count = monthly_counts[f"2026-{month:02d}"]

        for _ in range(count):
            local_time = random_local_datetime(2026, month, rng)
            car_id = rng.choice(CAR_IDS)
            trip_type = (
                "parking" if rng.random() < 0.85 else "driving"
            )
            specifications.append((local_time, car_id, trip_type))

    specifications.sort(key=lambda item: (item[0], item[1], item[2]))

    rows = [
        build_row(
            row_id=START_ID + index,
            local_time=local_time,
            car_id=car_id,
            trip_type=trip_type,
            groups=groups,
            rng=rng,
        )
        for index, (local_time, car_id, trip_type)
        in enumerate(specifications)
    ]

    return rows, monthly_counts, dictionary


def parse_event_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y/%m/%d %H:%M")


def validate_rows(
    rows: list[dict[str, object]],
    monthly_counts: dict[str, int],
    dictionary: dict[int, Tag],
) -> None:
    ids = [int(row["id"]) for row in rows]
    expected_ids = list(range(START_ID, START_ID + len(rows)))
    assert ids == expected_ids, "id 不是连续自增"

    month_actual: dict[str, int] = {}
    parking_count = 0

    for row in rows:
        assert list(row.keys()) == CSV_COLUMNS
        assert row["status"] == ""
        assert row["vin"] == ""
        assert row["carName"] == ""
        assert row["duration"] == 60
        assert row["platform"] == PLATFORM
        assert row["srcBucket"] == BUCKET
        assert row["unpackBucket"] == BUCKET
        assert 6_400_000_000 <= int(row["fileSize"]) <= 7_200_000_000

        file_match = re.fullmatch(
            r"(YR-[A-Z0-9]{3}-\d{5})_"
            r"(\d{8})_(\d{6})\.All\.bag\.des",
            str(row["fileName"]),
        )
        assert file_match, f"fileName 格式错误：{row['fileName']}"
        assert file_match.group(1) == row["carId"]

        utc_time = datetime.strptime(
            file_match.group(2) + file_match.group(3),
            "%Y%m%d%H%M%S",
        )
        local_time = parse_event_time(str(row["eventTime"]))

        # eventTime 不保存秒，因此比较到分钟。
        assert (
            utc_time + timedelta(hours=8)
        ).replace(second=0) == local_time

        expected_path_part = (
            f"/{utc_time:%Y%m%d}/{utc_time:%H}/{utc_time:%M}/"
        )
        assert expected_path_part in str(row["srcPath"])
        assert str(row["srcPath"]).endswith(str(row["fileName"]))

        assert row["collectionStatus"] in {"正式采集", "试采"}, (
            f"collectionStatus 取值非法：{row['collectionStatus']}"
        )
        if row["tripType"] == "parking":
            assert row["parkingLotType"] in PARKING_LOT_TYPES, (
                f"parkingLotType 取值非法：{row['parkingLotType']}"
            )
        else:
            assert row["parkingLotType"] == "", "行车行不应有停车场类型"

        month_key = f"{local_time.year}-{local_time.month:02d}"
        month_actual[month_key] = month_actual.get(month_key, 0) + 1

        if row["tripType"] == "parking":
            parking_count += 1

    assert month_actual == monthly_counts

    parking_ratio = parking_count / len(rows)
    assert 0.82 <= parking_ratio <= 0.88, (
        f"parking 比例偏离 85%：{parking_ratio:.2%}"
    )


def write_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="生成全部 6 个月数据；默认只输出前 100 行",
    )
    args = parser.parse_args()

    rows, monthly_counts, dictionary = generate_rows()
    validate_rows(rows, monthly_counts, dictionary)

    if args.full:
        output_path = OUTPUT_DIR / "main_clips.csv"
        output_rows = rows
    else:
        output_path = OUTPUT_DIR / "main_clips_sample_100.csv"
        output_rows = rows[:100]

    write_csv(output_rows, output_path)

    print(f"随机种子：{SEED}")
    print(f"输出文件：{output_path}")
    print(f"输出行数：{len(output_rows)}")
    print(f"全量行数：{len(rows)}")
    print(f"字段数：{len(CSV_COLUMNS)}")
    print("各月全量：")
    for month, count in monthly_counts.items():
        print(f"  {month}: {count}")


if __name__ == "__main__":
    main()