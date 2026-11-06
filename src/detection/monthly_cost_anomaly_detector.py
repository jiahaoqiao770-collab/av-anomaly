"""Rule-based anomaly detection for monthly Clip cost-efficiency metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.clip_cost_metrics import (
    DEFAULT_OUTPUT_PATH as DEFAULT_METRICS_PATH,
    OUTPUT_COLUMN_NAMES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "config/anomaly_rules/monthly_cost_anomaly_rules.json"
DEFAULT_ANOMALY_CSV_PATH = PROJECT_ROOT / "data/processed/monthly_cost_anomaly_events.csv"
DEFAULT_ANOMALY_JSONL_PATH = PROJECT_ROOT / "data/processed/monthly_cost_anomaly_events.jsonl"

SEVERITY_ORDER = {"warning": 1, "critical": 2}
CSV_COLUMNS = [
    "月份", "对比月份", "规则ID", "规则名称", "业务类型", "严重程度",
    "当前值", "上月值", "变化值", "变化率", "触发阈值", "单位", "说明",
]


class AnomalyRuleError(ValueError):
    """Raised when rule configuration or metric input is invalid."""


def load_rules(path: Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到异常规则：{path}")
    with path.open("r", encoding="utf-8") as file:
        rules = json.load(file)
    if not isinstance(rules.get("rules"), list):
        raise AnomalyRuleError(f"规则文件缺少 rules 数组：{path}")
    ids = [rule.get("rule_id") for rule in rules["rules"]]
    if len(ids) != len(set(ids)) or any(not rule_id for rule_id in ids):
        raise AnomalyRuleError("异常规则ID为空或重复")
    return rules


def load_metrics(path: Path = DEFAULT_METRICS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到月度指标结果：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    reverse_names = {chinese: english for english, chinese in OUTPUT_COLUMN_NAMES.items()}
    frame = frame.rename(columns=reverse_names)
    required = {"month", *[name for name in OUTPUT_COLUMN_NAMES if name != "month"]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AnomalyRuleError(f"月度指标结果缺少字段：{missing}")
    if frame.empty or frame["month"].duplicated().any():
        raise AnomalyRuleError("月度指标结果为空或月份重复")
    frame = frame.sort_values("month").reset_index(drop=True)
    numeric_columns = sorted(required.difference({"month"}))
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    return frame


def _change(current: float, baseline: float) -> float | None:
    return None if baseline == 0 else current / baseline - 1


def _severity(value: float, warning: float, critical: float) -> str | None:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return None


def _event(
    *,
    rule: dict[str, Any],
    month: str,
    baseline_month: str | None,
    business_type: str,
    severity: str,
    current_value: float,
    baseline_value: float | None,
    change_value: float | None,
    change_rate: float | None,
    threshold: Any,
    unit: str,
    summary: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "period": month,
        "baseline_period": baseline_month,
        "rule_id": rule["rule_id"],
        "rule_name": rule["name"],
        "business_type": business_type,
        "severity": severity,
        "current_value": current_value,
        "baseline_value": baseline_value,
        "change_value": change_value,
        "change_rate": change_rate,
        "threshold": threshold,
        "unit": unit,
        "summary": summary,
        "evidence": evidence,
    }


def _threshold_for(rule: dict[str, Any], severity: str) -> float:
    key = "critical_threshold" if severity == "critical" else "warning_threshold"
    return float(rule[key])


def _detect_mom(
    frame: pd.DataFrame,
    rule: dict[str, Any],
    direction: str,
    target: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    target = target or rule
    metric = target["metric"]
    events = []
    for index in range(1, len(frame)):
        current = float(frame.loc[index, metric])
        baseline = float(frame.loc[index - 1, metric])
        raw_change = _change(current, baseline)
        if raw_change is None:
            continue
        adverse_change = raw_change if direction == "increase" else -raw_change
        severity = _severity(
            adverse_change,
            float(rule["warning_threshold"]),
            float(rule["critical_threshold"]),
        )
        if not severity:
            continue
        verb = "上涨" if direction == "increase" else "下降"
        events.append(_event(
            rule=rule,
            month=str(frame.loc[index, "month"]),
            baseline_month=str(frame.loc[index - 1, "month"]),
            business_type=target.get("business_type", rule.get("business_type", "整体")),
            severity=severity,
            current_value=current,
            baseline_value=baseline,
            change_value=current - baseline,
            change_rate=raw_change,
            threshold=_threshold_for(rule, severity),
            unit=rule["unit"],
            summary=f"{rule['name']}，环比{verb}{abs(raw_change):.2%}。",
            evidence={"metric": metric, "adverse_change_rate": adverse_change},
        ))
    return events


def _detect_absolute(frame: pd.DataFrame, rule: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for target in rule["targets"]:
        metric = target["metric"]
        for _, row in frame.iterrows():
            value = float(row[metric])
            severity = _severity(
                value,
                float(target["warning_threshold"]),
                float(target["critical_threshold"]),
            )
            if not severity:
                continue
            threshold_key = "critical_threshold" if severity == "critical" else "warning_threshold"
            events.append(_event(
                rule=rule,
                month=str(row["month"]),
                baseline_month=None,
                business_type=target["business_type"],
                severity=severity,
                current_value=value,
                baseline_value=None,
                change_value=None,
                change_rate=None,
                threshold=float(target[threshold_key]),
                unit=rule["unit"],
                summary=f"{target['business_type']}单Clip成本为{value:.2f}元，超过{severity}阈值。",
                evidence={"metric": metric},
            ))
    return events


def _detect_consecutive(frame: pd.DataFrame, rule: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for target in rule["targets"]:
        metric = target["metric"]
        streak = 0
        for index in range(1, len(frame)):
            current = float(frame.loc[index, metric])
            baseline = float(frame.loc[index - 1, metric])
            streak = streak + 1 if current > baseline else 0
            severity = None
            if streak >= int(rule["critical_periods"]):
                severity = "critical"
            elif streak >= int(rule["warning_periods"]):
                severity = "warning"
            if not severity:
                continue
            threshold = int(
                rule["critical_periods"] if severity == "critical" else rule["warning_periods"]
            )
            events.append(_event(
                rule=rule,
                month=str(frame.loc[index, "month"]),
                baseline_month=str(frame.loc[index - 1, "month"]),
                business_type=target["business_type"],
                severity=severity,
                current_value=current,
                baseline_value=baseline,
                change_value=current - baseline,
                change_rate=_change(current, baseline),
                threshold=threshold,
                unit=rule["unit"],
                summary=f"{target['business_type']}单Clip成本已连续上涨{streak}个月。",
                evidence={"metric": metric, "consecutive_periods": streak},
            ))
    return events


def _detect_compound(frame: pd.DataFrame, rule: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    efficiency_rule = rule["kind"] == "hours_up_efficiency_down"
    decline_key = "efficiency_decrease" if efficiency_rule else "clip_decrease"
    metric_key = "efficiency_metric" if efficiency_rule else "clip_metric"
    for target in rule["targets"]:
        for index in range(1, len(frame)):
            hours_current = float(frame.loc[index, target["hours_metric"]])
            hours_baseline = float(frame.loc[index - 1, target["hours_metric"]])
            metric_current = float(frame.loc[index, target[metric_key]])
            metric_baseline = float(frame.loc[index - 1, target[metric_key]])
            hours_change = _change(hours_current, hours_baseline)
            metric_change = _change(metric_current, metric_baseline)
            if hours_change is None or metric_change is None:
                continue
            decline = -metric_change
            severity = None
            if (
                hours_change >= float(rule["critical"]["hours_increase"])
                and decline >= float(rule["critical"][decline_key])
            ):
                severity = "critical"
            elif (
                hours_change >= float(rule["warning"]["hours_increase"])
                and decline >= float(rule["warning"][decline_key])
            ):
                severity = "warning"
            if not severity:
                continue
            threshold = rule[severity]
            subject = "每小时Clip数" if efficiency_rule else "Clip数"
            events.append(_event(
                rule=rule,
                month=str(frame.loc[index, "month"]),
                baseline_month=str(frame.loc[index - 1, "month"]),
                business_type=target["business_type"],
                severity=severity,
                current_value=metric_current,
                baseline_value=metric_baseline,
                change_value=metric_current - metric_baseline,
                change_rate=metric_change,
                threshold=threshold,
                unit=rule["unit"],
                summary=(
                    f"{target['business_type']}工时环比上涨{hours_change:.2%}，"
                    f"但{subject}下降{decline:.2%}。"
                ),
                evidence={
                    "hours_metric": target["hours_metric"],
                    "hours_change_rate": hours_change,
                    "output_metric": target[metric_key],
                    "output_change_rate": metric_change,
                },
            ))
    return events


def _detect_growth_spread(frame: pd.DataFrame, rule: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for target in rule["targets"]:
        for index in range(1, len(frame)):
            cost_change = _change(
                float(frame.loc[index, target["cost_metric"]]),
                float(frame.loc[index - 1, target["cost_metric"]]),
            )
            clip_change = _change(
                float(frame.loc[index, target["clip_metric"]]),
                float(frame.loc[index - 1, target["clip_metric"]]),
            )
            if cost_change is None or clip_change is None:
                continue
            spread = cost_change - clip_change
            severity = _severity(
                spread,
                float(rule["warning_threshold"]),
                float(rule["critical_threshold"]),
            )
            if not severity:
                continue
            events.append(_event(
                rule=rule,
                month=str(frame.loc[index, "month"]),
                baseline_month=str(frame.loc[index - 1, "month"]),
                business_type=target["business_type"],
                severity=severity,
                current_value=cost_change,
                baseline_value=clip_change,
                change_value=spread,
                change_rate=spread,
                threshold=_threshold_for(rule, severity),
                unit=rule["unit"],
                summary=(
                    f"{target['business_type']}分摊成本增幅比Clip数增幅高{spread:.2%}。"
                ),
                evidence={
                    "allocated_cost_change_rate": cost_change,
                    "clip_count_change_rate": clip_change,
                },
            ))
    return events


def _detect_share_change(frame: pd.DataFrame, rule: dict[str, Any]) -> list[dict[str, Any]]:
    share = frame[rule["numerator_metric"]] / frame[rule["denominator_metric"]]
    events = []
    for index in range(1, len(frame)):
        change = float(share.iloc[index] - share.iloc[index - 1])
        severity = _severity(
            abs(change),
            float(rule["warning_threshold"]),
            float(rule["critical_threshold"]),
        )
        if not severity:
            continue
        direction = "上升" if change > 0 else "下降"
        events.append(_event(
            rule=rule,
            month=str(frame.loc[index, "month"]),
            baseline_month=str(frame.loc[index - 1, "month"]),
            business_type=rule["business_type"],
            severity=severity,
            current_value=float(share.iloc[index]),
            baseline_value=float(share.iloc[index - 1]),
            change_value=change,
            change_rate=None,
            threshold=_threshold_for(rule, severity),
            unit=rule["unit"],
            summary=f"泊车工时占比环比{direction}{abs(change):.2%}。",
            evidence={"share_change_percentage_point": change},
        ))
    return events


def _detect_ratio_increase(frame: pd.DataFrame, rule: dict[str, Any]) -> list[dict[str, Any]]:
    ratio = frame[rule["numerator_metric"]] / frame[rule["denominator_metric"]]
    events = []
    for index in range(1, len(frame)):
        change = _change(float(ratio.iloc[index]), float(ratio.iloc[index - 1]))
        if change is None:
            continue
        severity = _severity(
            change,
            float(rule["warning_threshold"]),
            float(rule["critical_threshold"]),
        )
        if not severity:
            continue
        events.append(_event(
            rule=rule,
            month=str(frame.loc[index, "month"]),
            baseline_month=str(frame.loc[index - 1, "month"]),
            business_type=rule["business_type"],
            severity=severity,
            current_value=float(ratio.iloc[index]),
            baseline_value=float(ratio.iloc[index - 1]),
            change_value=float(ratio.iloc[index] - ratio.iloc[index - 1]),
            change_rate=change,
            threshold=_threshold_for(rule, severity),
            unit=rule["unit"],
            summary=f"行车与泊车单Clip成本比值环比扩大{change:.2%}。",
            evidence={"cost_ratio": float(ratio.iloc[index])},
        ))
    return events


def detect_anomalies(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    rules_path: Path = DEFAULT_RULES_PATH,
) -> list[dict[str, Any]]:
    """Apply all enabled rules and return sorted anomaly events."""
    frame = load_metrics(metrics_path)
    rules = load_rules(rules_path)
    events: list[dict[str, Any]] = []
    for rule in rules["rules"]:
        if not rule.get("enabled", True):
            continue
        kind = rule["kind"]
        if kind in {"mom_increase", "mom_decrease"}:
            direction = "increase" if kind == "mom_increase" else "decrease"
            targets = rule.get("targets", [rule])
            for target in targets:
                events.extend(_detect_mom(frame, rule, direction, target))
        elif kind == "absolute_high":
            events.extend(_detect_absolute(frame, rule))
        elif kind == "consecutive_increase":
            events.extend(_detect_consecutive(frame, rule))
        elif kind in {"hours_up_clips_down", "hours_up_efficiency_down"}:
            events.extend(_detect_compound(frame, rule))
        elif kind == "growth_spread":
            events.extend(_detect_growth_spread(frame, rule))
        elif kind == "share_change":
            events.extend(_detect_share_change(frame, rule))
        elif kind == "ratio_mom_increase":
            events.extend(_detect_ratio_increase(frame, rule))
        else:
            raise AnomalyRuleError(f"不支持的规则类型：{kind}")
    return sorted(
        events,
        key=lambda item: (
            item["period"],
            -SEVERITY_ORDER[item["severity"]],
            item["rule_id"],
            item["business_type"],
        ),
    )


def write_anomaly_outputs(
    events: list[dict[str, Any]],
    csv_path: Path = DEFAULT_ANOMALY_CSV_PATH,
    jsonl_path: Path = DEFAULT_ANOMALY_JSONL_PATH,
) -> tuple[Path, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for event in events:
        threshold = event["threshold"]
        rows.append({
            "月份": event["period"],
            "对比月份": event["baseline_period"] or "",
            "规则ID": event["rule_id"],
            "规则名称": event["rule_name"],
            "业务类型": event["business_type"],
            "严重程度": "严重" if event["severity"] == "critical" else "警告",
            "当前值": event["current_value"],
            "上月值": event["baseline_value"],
            "变化值": event["change_value"],
            "变化率": event["change_rate"],
            "触发阈值": json.dumps(threshold, ensure_ascii=False) if isinstance(threshold, dict) else threshold,
            "单位": event["unit"],
            "说明": event["summary"],
        })
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    with jsonl_path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def load_anomaly_events(path: Path = DEFAULT_ANOMALY_JSONL_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到异常检测结果：{path}")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_anomalies(
    *,
    month: str | None = None,
    severity: str | None = None,
    rule_id: str | None = None,
    business_type: str | None = None,
    path: Path = DEFAULT_ANOMALY_JSONL_PATH,
) -> list[dict[str, Any]]:
    """Query persisted anomaly events for the future model/tool layer."""
    events = load_anomaly_events(path)
    if severity not in {None, "warning", "critical"}:
        raise ValueError("severity 只能是 warning、critical 或 None")
    return [
        event for event in events
        if (month is None or event["period"] == month)
        and (severity is None or event["severity"] == severity)
        and (rule_id is None or event["rule_id"] == rule_id)
        and (business_type is None or event["business_type"] == business_type)
    ]

