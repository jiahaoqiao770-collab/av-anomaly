"""Calculate monthly driving and parking Clip cost-efficiency metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINANCE_PATH = PROJECT_ROOT / "data/synthetic/财务.csv"
DEFAULT_DAILY_REPORT_PATH = PROJECT_ROOT / "data/synthetic/日报表.csv"
DEFAULT_MAIN_CLIPS_PATH = PROJECT_ROOT / "data/synthetic/main_clips_202606_202611.csv"
DEFAULT_DICTIONARY_PATH = PROJECT_ROOT / "config/metrics/clip_metric_definitions.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/monthly_clip_cost_metrics.csv"

FINANCE_COST_COLUMNS = [
    "外包服务费(万)",
    "车辆运维费(万)",
    "硬盘邮寄(万)",
    "场地测试(新增)(万)",
]

METRIC_COLUMNS = [
    "monthly_total_cost",
    "total_work_hours",
    "driving_work_hours",
    "parking_work_hours",
    "driving_allocated_cost",
    "parking_allocated_cost",
    "driving_clip_count",
    "parking_clip_count",
    "driving_clips_per_hour",
    "parking_clips_per_hour",
    "driving_unit_clip_cost",
    "parking_unit_clip_cost",
    "parking_trial_clip_count",
    "parking_trial_clip_ratio",
    "parking_effective_unit_cost",
]

OUTPUT_COLUMN_NAMES = {
    "month": "月份",
    "monthly_total_cost": "月度总成本_万元",
    "total_work_hours": "总工时_小时",
    "driving_work_hours": "行车工时_小时",
    "parking_work_hours": "泊车工时_小时",
    "driving_allocated_cost": "行车分摊成本_万元",
    "parking_allocated_cost": "泊车分摊成本_万元",
    "driving_clip_count": "行车Clip数",
    "parking_clip_count": "泊车Clip数",
    "driving_clips_per_hour": "行车每小时Clip数",
    "parking_clips_per_hour": "泊车每小时Clip数",
    "driving_unit_clip_cost": "行车单Clip成本_元",
    "parking_unit_clip_cost": "泊车单Clip成本_元",
    "parking_trial_clip_count": "泊车试采Clip数",
    "parking_trial_clip_ratio": "泊车试采占比",
    "parking_effective_unit_cost": "泊车有效单Clip成本_元",
}


class MetricDataError(ValueError):
    """Raised when source data cannot support the confirmed metric rules."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MetricDataError(f"{source} 缺少字段：{missing}")


def _parse_month(series: pd.Series, field: str, source: Path) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.isna().any():
        rows = (parsed[parsed.isna()].index + 2).tolist()[:10]
        raise MetricDataError(f"{source} 的 {field} 存在无法解析的日期，CSV行号：{rows}")
    return parsed.dt.to_period("M").astype(str)


def _parse_nonnegative_number(
    series: pd.Series,
    field: str,
    source: Path,
) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    invalid = parsed.isna() | (parsed < 0)
    if invalid.any():
        rows = (parsed[invalid].index + 2).tolist()[:10]
        raise MetricDataError(
            f"{source} 的 {field} 存在空值、非数字或负数，CSV行号：{rows}"
        )
    return parsed


def load_metric_dictionary(path: Path = DEFAULT_DICTIONARY_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到指标字典：{path}")
    with path.open("r", encoding="utf-8") as file:
        dictionary = json.load(file)

    definitions = dictionary.get("metrics")
    if not isinstance(definitions, dict):
        raise MetricDataError(f"指标字典缺少 metrics 对象：{path}")

    missing = sorted(set(METRIC_COLUMNS).difference(definitions))
    if missing:
        raise MetricDataError(f"指标字典缺少已实现指标：{missing}")
    return dictionary


def calculate_monthly_total_cost(
    finance: pd.DataFrame,
    source: Path = DEFAULT_FINANCE_PATH,
) -> pd.DataFrame:
    required = {"时间", *FINANCE_COST_COLUMNS}
    _require_columns(finance, required, source)

    prepared = finance.loc[:, ["时间", *FINANCE_COST_COLUMNS]].copy()
    prepared["month"] = _parse_month(prepared["时间"], "时间", source)
    if prepared["month"].duplicated().any():
        months = sorted(prepared.loc[prepared["month"].duplicated(False), "month"].unique())
        raise MetricDataError(f"{source} 每月应只有一行，重复月份：{months}")

    for field in FINANCE_COST_COLUMNS:
        prepared[field] = _parse_nonnegative_number(prepared[field], field, source)

    prepared["monthly_total_cost"] = prepared[FINANCE_COST_COLUMNS].sum(axis=1)
    return prepared[["month", "monthly_total_cost"]].sort_values("month").reset_index(drop=True)


def calculate_monthly_work_hours(
    daily_report: pd.DataFrame,
    source: Path = DEFAULT_DAILY_REPORT_PATH,
) -> pd.DataFrame:
    required = {"日期", "需求类型", "工作时长"}
    _require_columns(daily_report, required, source)

    prepared = daily_report.loc[:, ["日期", "需求类型", "工作时长"]].copy()
    prepared["month"] = _parse_month(prepared["日期"], "日期", source)
    prepared["工作时长"] = _parse_nonnegative_number(
        prepared["工作时长"], "工作时长", source
    )

    allowed = {"行车", "泊车"}
    actual = set(prepared["需求类型"].dropna().astype(str).str.strip())
    invalid = sorted(actual.difference(allowed))
    if invalid or prepared["需求类型"].isna().any():
        raise MetricDataError(f"{source} 存在未定义的需求类型：{invalid}")
    prepared["需求类型"] = prepared["需求类型"].astype(str).str.strip()

    pivot = prepared.pivot_table(
        index="month",
        columns="需求类型",
        values="工作时长",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=["行车", "泊车"], fill_value=0.0)

    result = pivot.rename(
        columns={"行车": "driving_work_hours", "泊车": "parking_work_hours"}
    ).reset_index()
    result["total_work_hours"] = (
        result["driving_work_hours"] + result["parking_work_hours"]
    )
    return result[
        ["month", "total_work_hours", "driving_work_hours", "parking_work_hours"]
    ].sort_values("month").reset_index(drop=True)


def calculate_monthly_clip_counts(
    main_clips_path: Path = DEFAULT_MAIN_CLIPS_PATH,
) -> pd.DataFrame:
    if not main_clips_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{main_clips_path}")

    connection = duckdb.connect()
    try:
        frame = connection.execute(
            """
            SELECT
                substr(eventTime, 1, 4)
                    || '-'
                    || lpad(split_part(eventTime, '/', 2), 2, '0') AS month,
                lower(trim(tripType)) AS trip_type,
                trim(collectionStatus) AS collection_status,
                count(*) AS clip_count
            FROM read_csv(?, header = true, all_varchar = true)
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
            """,
            [str(main_clips_path)],
        ).df()
    finally:
        connection.close()

    if frame.empty:
        raise MetricDataError(f"{main_clips_path} 没有可用的 Clip 记录")

    allowed_trip = {"driving", "parking"}
    actual_trip = set(frame["trip_type"].dropna())
    invalid_trip = sorted(actual_trip.difference(allowed_trip))
    if invalid_trip or frame["trip_type"].isna().any():
        raise MetricDataError(
            f"{main_clips_path} 存在未定义的 tripType：{invalid_trip}"
        )

    allowed_status = {"正式采集", "试采"}
    actual_status = set(frame["collection_status"].dropna())
    invalid_status = sorted(actual_status.difference(allowed_status))
    if invalid_status or frame["collection_status"].isna().any():
        raise MetricDataError(
            f"{main_clips_path} 存在未定义的 collectionStatus：{invalid_status}"
        )

    frame["bucket"] = frame["trip_type"] + "|" + frame["collection_status"]
    pivot = frame.pivot(
        index="month", columns="bucket", values="clip_count"
    )
    pivot = pivot.reindex(
        columns=[
            "driving|正式采集",
            "driving|试采",
            "parking|正式采集",
            "parking|试采",
        ],
        fill_value=0,
    ).fillna(0).astype("int64")
    pivot.columns.name = None
    pivot.index.name = "month"

    result = pd.DataFrame(
        {
            "driving_clip_count": pivot["driving|正式采集"]
            + pivot["driving|试采"],
            "parking_clip_count": pivot["parking|正式采集"]
            + pivot["parking|试采"],
            "parking_trial_clip_count": pivot["parking|试采"],
        }
    ).reset_index()

    return result.sort_values("month").reset_index(drop=True)


def combine_monthly_base_metrics(
    costs: pd.DataFrame,
    work_hours: pd.DataFrame,
    clip_counts: pd.DataFrame,
) -> pd.DataFrame:
    month_sets = {
        "财务": set(costs["month"]),
        "日报": set(work_hours["month"]),
        "Clip": set(clip_counts["month"]),
    }
    if len({frozenset(months) for months in month_sets.values()}) != 1:
        details = {name: sorted(months) for name, months in month_sets.items()}
        raise MetricDataError(f"三张输入表的月份范围不一致：{details}")

    result = costs.merge(work_hours, on="month", validate="one_to_one")
    result = result.merge(clip_counts, on="month", validate="one_to_one")
    return result.sort_values("month").reset_index(drop=True)


def calculate_derived_metrics(base_metrics: pd.DataFrame) -> pd.DataFrame:
    result = base_metrics.copy()
    if (result["total_work_hours"] <= 0).any():
        raise MetricDataError("总工时必须大于0，无法进行成本分摊")
    if (result[["driving_work_hours", "parking_work_hours"]] <= 0).any().any():
        raise MetricDataError("行车和泊车工时必须大于0，无法计算每小时Clip数")
    if (result[["driving_clip_count", "parking_clip_count"]] <= 0).any().any():
        raise MetricDataError("行车和泊车Clip数必须大于0，无法计算单Clip成本")

    result["driving_allocated_cost"] = (
        result["monthly_total_cost"]
        * result["driving_work_hours"]
        / result["total_work_hours"]
    )
    result["parking_allocated_cost"] = (
        result["monthly_total_cost"]
        * result["parking_work_hours"]
        / result["total_work_hours"]
    )
    result["driving_clips_per_hour"] = (
        result["driving_clip_count"] / result["driving_work_hours"]
    )
    result["parking_clips_per_hour"] = (
        result["parking_clip_count"] / result["parking_work_hours"]
    )
    result["driving_unit_clip_cost"] = (
        result["driving_allocated_cost"] * 10_000 / result["driving_clip_count"]
    )
    result["parking_unit_clip_cost"] = (
        result["parking_allocated_cost"] * 10_000 / result["parking_clip_count"]
    )
    official_parking_clips = (
        result["parking_clip_count"] - result["parking_trial_clip_count"]
    )
    if (official_parking_clips <= 0).any():
        raise MetricDataError("正式采集泊车Clip数必须大于0，无法计算有效成本")

    result["parking_trial_clip_ratio"] = (
        result["parking_trial_clip_count"] / result["parking_clip_count"]
    )
    result["parking_effective_unit_cost"] = (
        result["parking_allocated_cost"] * 10_000 / official_parking_clips
    )
    return result[["month", *METRIC_COLUMNS]]


def validate_monthly_metrics(metrics: pd.DataFrame) -> None:
    required = {"month", *METRIC_COLUMNS}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise MetricDataError(f"月度指标结果缺少字段：{missing}")
    if metrics.empty or metrics["month"].duplicated().any():
        raise MetricDataError("月度指标结果为空或月份重复")
    if metrics[METRIC_COLUMNS].isna().any().any():
        raise MetricDataError("月度指标结果存在空值")
    if (metrics[METRIC_COLUMNS] < 0).any().any():
        raise MetricDataError("月度指标结果存在负数")

    hour_difference = (
        metrics["total_work_hours"]
        - metrics["driving_work_hours"]
        - metrics["parking_work_hours"]
    ).abs()
    if (hour_difference > 1e-9).any():
        raise MetricDataError("总工时不等于行车工时与泊车工时之和")

    cost_difference = (
        metrics["monthly_total_cost"]
        - metrics["driving_allocated_cost"]
        - metrics["parking_allocated_cost"]
    ).abs()
    if (cost_difference > 1e-9).any():
        raise MetricDataError("行车与泊车分摊成本之和不等于月度总成本")

    driving_efficiency_difference = (
        metrics["driving_clips_per_hour"]
        - metrics["driving_clip_count"] / metrics["driving_work_hours"]
    ).abs()
    parking_efficiency_difference = (
        metrics["parking_clips_per_hour"]
        - metrics["parking_clip_count"] / metrics["parking_work_hours"]
    ).abs()
    if (
        (driving_efficiency_difference > 1e-12).any()
        or (parking_efficiency_difference > 1e-12).any()
    ):
        raise MetricDataError("每小时Clip数与Clip数、工时无法对账")


def calculate_monthly_metrics(
    finance_path: Path = DEFAULT_FINANCE_PATH,
    daily_report_path: Path = DEFAULT_DAILY_REPORT_PATH,
    main_clips_path: Path = DEFAULT_MAIN_CLIPS_PATH,
    dictionary_path: Path = DEFAULT_DICTIONARY_PATH,
) -> pd.DataFrame:
    """Load source CSVs and return one validated row per month."""
    load_metric_dictionary(dictionary_path)
    finance = _read_csv(finance_path)
    daily_report = _read_csv(daily_report_path)

    costs = calculate_monthly_total_cost(finance, finance_path)
    work_hours = calculate_monthly_work_hours(daily_report, daily_report_path)
    clip_counts = calculate_monthly_clip_counts(main_clips_path)
    base_metrics = combine_monthly_base_metrics(costs, work_hours, clip_counts)
    metrics = calculate_derived_metrics(base_metrics)
    validate_monthly_metrics(metrics)
    return metrics


def get_metric(
    metric_id: str,
    month: str,
    metrics: pd.DataFrame | None = None,
) -> int | float:
    """Return one confirmed metric value for a YYYY-MM month."""
    if metric_id not in METRIC_COLUMNS:
        raise KeyError(f"未实现的指标：{metric_id}")
    available = calculate_monthly_metrics() if metrics is None else metrics
    selected = available.loc[available["month"] == month, metric_id]
    if selected.empty:
        raise KeyError(f"没有月份 {month} 的指标数据")
    value = selected.iloc[0]
    return int(value) if metric_id.endswith("_count") else float(value)


def compare_metric(
    metric_id: str,
    current_month: str,
    baseline_month: str,
    metrics: pd.DataFrame | None = None,
) -> dict[str, int | float | str | None]:
    """Compare one metric between two months for the future query layer."""
    available = calculate_monthly_metrics() if metrics is None else metrics
    current = get_metric(metric_id, current_month, available)
    baseline = get_metric(metric_id, baseline_month, available)
    change = float(current) - float(baseline)
    change_rate = None if float(baseline) == 0 else change / float(baseline)
    return {
        "metric_id": metric_id,
        "current_month": current_month,
        "current_value": current,
        "baseline_month": baseline_month,
        "baseline_value": baseline,
        "change": change,
        "change_rate": change_rate,
    }


def format_metrics_for_output(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a readable CSV view without changing calculation precision."""
    output = metrics.copy()
    hour_columns = ["total_work_hours", "driving_work_hours", "parking_work_hours"]
    cost_columns = [
        "monthly_total_cost",
        "driving_allocated_cost",
        "parking_allocated_cost",
    ]
    unit_cost_columns = [
        "driving_unit_clip_cost",
        "parking_unit_clip_cost",
        "parking_effective_unit_cost",
    ]
    efficiency_columns = ["driving_clips_per_hour", "parking_clips_per_hour"]
    count_columns = [
        "driving_clip_count",
        "parking_clip_count",
        "parking_trial_clip_count",
    ]

    output[hour_columns] = output[hour_columns].round(1)
    output[cost_columns] = output[cost_columns].round(4)
    output[efficiency_columns] = output[efficiency_columns].round(4)
    output["parking_trial_clip_ratio"] = output["parking_trial_clip_ratio"].round(4)
    output[unit_cost_columns] = output[unit_cost_columns].round(2)
    output[count_columns] = output[count_columns].astype("int64")
    return output.rename(columns=OUTPUT_COLUMN_NAMES)


def write_monthly_metrics(
    metrics: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write a validated, Excel-readable monthly metrics CSV."""
    validate_monthly_metrics(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_metrics_for_output(metrics).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return output_path

