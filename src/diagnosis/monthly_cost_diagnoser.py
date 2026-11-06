"""Merge monthly anomaly events into preliminary business diagnoses."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.detection.monthly_cost_anomaly_detector import (
    DEFAULT_ANOMALY_JSONL_PATH,
    DEFAULT_METRICS_PATH,
    load_anomaly_events,
    load_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSIS_RULES_PATH = (
    PROJECT_ROOT / "config/diagnosis_rules/monthly_cost_diagnosis_rules.json"
)
DEFAULT_DIAGNOSIS_CSV_PATH = (
    PROJECT_ROOT / "data/processed/monthly_cost_diagnosis_results.csv"
)
DEFAULT_DIAGNOSIS_JSONL_PATH = (
    PROJECT_ROOT / "data/processed/monthly_cost_diagnosis_results.jsonl"
)

CSV_COLUMNS = [
    "诊断ID",
    "月份",
    "对比月份",
    "业务类型",
    "严重程度",
    "主要异常",
    "主要关联因素",
    "其他关联因素",
    "辅助证据",
    "背景信息",
    "触发规则ID",
    "触发规则数",
    "诊断说明",
]


class DiagnosisRuleError(ValueError):
    """Raised when diagnosis configuration or input events are invalid."""


def load_diagnosis_rules(
    path: Path = DEFAULT_DIAGNOSIS_RULES_PATH,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到诊断规则：{path}")
    with path.open("r", encoding="utf-8") as file:
        rules = json.load(file)

    roles = rules.get("rule_roles")
    if not isinstance(roles, dict) or not roles:
        raise DiagnosisRuleError(f"诊断规则缺少 rule_roles 对象：{path}")
    allowed_roles = {"outcome", "driver", "evidence", "context"}
    invalid = sorted(
        rule_id
        for rule_id, definition in roles.items()
        if definition.get("role") not in allowed_roles
    )
    if invalid:
        raise DiagnosisRuleError(f"以下规则的诊断角色无效：{invalid}")
    return rules


def _severity_value(severity: str, config: dict[str, Any]) -> int:
    try:
        return int(config["severity_order"][severity])
    except KeyError as error:
        raise DiagnosisRuleError(f"未定义严重程度：{severity}") from error


def _plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _metric_snapshots(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        month = str(row["month"])
        snapshots[month] = {
            key: _plain_value(value)
            for key, value in row.items()
            if key != "month"
        }
    return snapshots


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": event["rule_id"],
        "rule_name": event["rule_name"],
        "severity": event["severity"],
        "summary": event["summary"],
        "current_value": event.get("current_value"),
        "baseline_value": event.get("baseline_value"),
        "change_value": event.get("change_value"),
        "change_rate": event.get("change_rate"),
        "unit": event.get("unit"),
        "evidence": event.get("evidence", {}),
    }


def _ranked(
    events: list[dict[str, Any]],
    role_config: dict[str, dict[str, Any]],
    severity_order: dict[str, int],
    priority_field: str,
) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda event: (
            int(role_config[event["rule_id"]].get(priority_field, 0)),
            int(severity_order[event["severity"]]),
            event["rule_id"],
        ),
        reverse=True,
    )


def _deduplicate(
    events: list[dict[str, Any]],
    role_config: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        definition = role_config[event["rule_id"]]
        key = str(definition.get("deduplication_key", event["rule_id"]))
        if key not in seen:
            selected.append(event)
            seen.add(key)
    return selected


def _clean_summary(summary: str) -> str:
    return summary.rstrip("。")


def _join_summaries(events: list[dict[str, Any]]) -> str:
    return "；".join(_clean_summary(event["summary"]) for event in events)


def _build_summary(
    primary: dict[str, Any],
    primary_reason: dict[str, Any] | None,
    contributing_reasons: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    context: list[dict[str, Any]],
) -> str:
    parts = [_clean_summary(primary["summary"])]
    reasons = ([primary_reason] if primary_reason else []) + contributing_reasons
    if reasons:
        parts.append(f"主要关联因素：{_join_summaries(reasons)}")
    if evidence:
        parts.append(f"辅助证据：{_join_summaries(evidence)}")
    if context:
        parts.append(f"背景信息：{_join_summaries(context)}")
    return "。".join(parts) + "。"


def _diagnose_group(
    period: str,
    business_type: str,
    events: list[dict[str, Any]],
    config: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_config = config["rule_roles"]
    unknown = sorted({event["rule_id"] for event in events}.difference(role_config))
    if unknown:
        raise DiagnosisRuleError(f"诊断配置缺少异常规则：{unknown}")

    outcomes = [
        event for event in events
        if role_config[event["rule_id"]]["role"] == "outcome"
    ]
    candidates = outcomes or [
        event for event in events
        if role_config[event["rule_id"]].get("fallback_primary", False)
    ]
    if not candidates:
        candidates = list(events)
    primary = _ranked(
        candidates, role_config, config["severity_order"], "primary_priority"
    )[0]

    drivers = [
        event for event in events
        if role_config[event["rule_id"]]["role"] == "driver" and event is not primary
    ]
    drivers = _deduplicate(
        _ranked(drivers, role_config, config["severity_order"], "reason_priority"),
        role_config,
    )
    primary_reason = drivers[0] if drivers else None
    contributing_reasons = drivers[1:]

    evidence_events = [
        event for event in events
        if role_config[event["rule_id"]]["role"] == "evidence"
        or (
            role_config[event["rule_id"]]["role"] == "outcome"
            and event is not primary
        )
    ]
    evidence_events = _deduplicate(
        _ranked(
            evidence_events,
            role_config,
            config["severity_order"],
            "evidence_priority",
        ),
        role_config,
    )
    context_events = _deduplicate(
        _ranked(
            [
                event for event in events
                if role_config[event["rule_id"]]["role"] == "context"
                and event is not primary
            ],
            role_config,
            config["severity_order"],
            "context_priority",
        ),
        role_config,
    )

    severity = max(
        (event["severity"] for event in events),
        key=lambda value: _severity_value(value, config),
    )
    baseline_periods = sorted(
        {event["baseline_period"] for event in events if event.get("baseline_period")}
    )
    rule_ids = sorted({event["rule_id"] for event in events})
    business_code = config.get("business_codes", {}).get(business_type, business_type)

    return {
        "diagnosis_id": f"{period}_{business_code}",
        "period": period,
        "baseline_periods": baseline_periods,
        "business_type": business_type,
        "severity": severity,
        "diagnosis_status": "rule_based_preliminary",
        "primary_anomaly": _compact_event(primary),
        "primary_reason": _compact_event(primary_reason) if primary_reason else None,
        "contributing_reasons": [_compact_event(event) for event in contributing_reasons],
        "supporting_evidence": [_compact_event(event) for event in evidence_events],
        "context": [_compact_event(event) for event in context_events],
        "triggered_rule_ids": rule_ids,
        "triggered_rule_count": len(rule_ids),
        "summary": _build_summary(
            primary,
            primary_reason,
            contributing_reasons,
            evidence_events,
            context_events,
        ),
        "metric_snapshot": {
            "current": snapshots.get(period),
            "baselines": {
                month: snapshots.get(month) for month in baseline_periods
            },
        },
        "source_events": events,
    }


def diagnose_monthly_cost_anomalies(
    events_path: Path = DEFAULT_ANOMALY_JSONL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    diagnosis_rules_path: Path = DEFAULT_DIAGNOSIS_RULES_PATH,
) -> list[dict[str, Any]]:
    """Group triggered rules dynamically and return preliminary diagnoses."""
    events = load_anomaly_events(events_path)
    metrics = load_metrics(metrics_path)
    config = load_diagnosis_rules(diagnosis_rules_path)
    snapshots = _metric_snapshots(metrics)

    by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_period[str(event["period"])].append(event)

    diagnoses: list[dict[str, Any]] = []
    for period in sorted(by_period):
        period_events = by_period[period]
        overall_events = [
            event for event in period_events if event["business_type"] == "整体"
        ]
        business_types = sorted(
            {event["business_type"] for event in period_events}
            .difference({"整体"})
        )

        if not business_types:
            diagnoses.append(
                _diagnose_group(period, "整体", overall_events, config, snapshots)
            )
            continue

        attachable_overall = [
            event for event in overall_events
            if config["rule_roles"].get(event["rule_id"], {}).get(
                "attach_to_business", False
            )
        ]
        for business_type in business_types:
            business_events = [
                event for event in period_events
                if event["business_type"] == business_type
            ]
            diagnoses.append(
                _diagnose_group(
                    period,
                    business_type,
                    business_events + attachable_overall,
                    config,
                    snapshots,
                )
            )
    return diagnoses


def _event_summary(event: dict[str, Any] | None) -> str:
    return "" if event is None else event["summary"]


def write_diagnosis_outputs(
    diagnoses: list[dict[str, Any]],
    csv_path: Path = DEFAULT_DIAGNOSIS_CSV_PATH,
    jsonl_path: Path = DEFAULT_DIAGNOSIS_JSONL_PATH,
) -> tuple[Path, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for diagnosis in diagnoses:
        rows.append({
            "诊断ID": diagnosis["diagnosis_id"],
            "月份": diagnosis["period"],
            "对比月份": "、".join(diagnosis["baseline_periods"]),
            "业务类型": diagnosis["business_type"],
            "严重程度": "严重" if diagnosis["severity"] == "critical" else "警告",
            "主要异常": _event_summary(diagnosis["primary_anomaly"]),
            "主要关联因素": _event_summary(diagnosis["primary_reason"]),
            "其他关联因素": "；".join(
                event["summary"] for event in diagnosis["contributing_reasons"]
            ),
            "辅助证据": "；".join(
                event["summary"] for event in diagnosis["supporting_evidence"]
            ),
            "背景信息": "；".join(
                event["summary"] for event in diagnosis["context"]
            ),
            "触发规则ID": "、".join(diagnosis["triggered_rule_ids"]),
            "触发规则数": diagnosis["triggered_rule_count"],
            "诊断说明": diagnosis["summary"],
        })
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    with jsonl_path.open("w", encoding="utf-8") as file:
        for diagnosis in diagnoses:
            file.write(json.dumps(diagnosis, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def load_diagnosis_results(
    path: Path = DEFAULT_DIAGNOSIS_JSONL_PATH,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到诊断结果：{path}")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_diagnoses(
    *,
    month: str | None = None,
    severity: str | None = None,
    business_type: str | None = None,
    path: Path = DEFAULT_DIAGNOSIS_JSONL_PATH,
) -> list[dict[str, Any]]:
    """Query persisted diagnoses for a future model or LangGraph tool."""
    if severity not in {None, "warning", "critical"}:
        raise ValueError("severity 只能是 warning、critical 或 None")
    return [
        diagnosis
        for diagnosis in load_diagnosis_results(path)
        if (month is None or diagnosis["period"] == month)
        and (severity is None or diagnosis["severity"] == severity)
        and (
            business_type is None
            or diagnosis["business_type"] == business_type
        )
    ]
