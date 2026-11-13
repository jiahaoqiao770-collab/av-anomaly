#!/usr/bin/env python3
"""事实质量评测：事实槽位准确率 / 证据支持率 / 关键事实覆盖率。

两种模式：
  --mode diagnosis  17 条诊断，三指标全测（端到端）
  --mode events     54 条 rule 级事件，只测槽位准确率与覆盖率（细粒度定位）

错误分类（产出错误清单，供 SFT 造数据）：
  E1 期间错 / E2 业务错 / E4 数值角色错 / E5 标度错
  E6 越权归因 / E7 无证据结论 / E8 关键事实遗漏

运行：
  python -m scripts.evaluate_rag_facts --mode diagnosis --save
  python -m scripts.evaluate_rag_facts --mode events --save
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from scripts.evaluate_rag_generation import (
    COLLECTION_NAME,
    EMBED_PATH,
    INDEX_PATH,
    MAX_NEW_TOKENS,
    QWEN_PATH,
    SYSTEM_PROMPT,
    generate,
    load_generator,
    load_jsonl,
)
from src.metrics.clip_cost_metrics import OUTPUT_COLUMN_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_PATH = PROJECT_ROOT / "data/processed/monthly_cost_diagnosis_results.jsonl"
EVENT_PATH = PROJECT_ROOT / "data/processed/monthly_cost_anomaly_events.jsonl"
REPORT_DIAG = PROJECT_ROOT / "docs/rag-facts-eval-v1.md"
REPORT_EVENT = PROJECT_ROOT / "docs/rag-facts-eval-events-v1.md"
DETAIL_OUT = PROJECT_ROOT / "data/results/facts_eval_v1.jsonl"

def _clean_metric_name(name: str) -> str:
    """去掉 "_元" "_万元" "_小时" 之类的单位后缀，便于在正文里匹配。"""
    return re.sub(r"_[^_]+$", "", name)


CMAP = {k: _clean_metric_name(v) for k, v in OUTPUT_COLUMN_NAMES.items()}

RULES_PATH = PROJECT_ROOT / "config/anomaly_rules/monthly_cost_anomaly_rules.json"
BUSINESS_PREFIX = {"泊车": "parking", "行车": "driving"}


def _load_rule_metrics() -> dict[str, str]:
    if not RULES_PATH.exists():
        return {}
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return {
        rule["rule_id"]: rule["metric"]
        for rule in data.get("rules", [])
        if rule.get("metric")
    }


RULE_METRIC = _load_rule_metrics()


def reason_metric(item: dict, business: str | None) -> str | None:
    """按三级回退解析一个 reason 对应的指标：evidence.metric -> evidence 键推导 -> 规则配置。"""
    item = item or {}
    evidence = item.get("evidence") or {}
    metric = evidence.get("metric")
    if metric and metric in CMAP:
        return metric

    prefix = BUSINESS_PREFIX.get(business or "", "")
    for key, value in evidence.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        base = key[: -len("_change_rate")] if key.endswith("_change_rate") else key
        for candidate in (f"{prefix}_{base}", base):
            if candidate in CMAP:
                return candidate

    rule_id = item.get("rule_id")
    if rule_id and RULE_METRIC.get(rule_id) in CMAP:
        return RULE_METRIC[rule_id]
    return None

PERIOD_RE = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})")
MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月")
NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|％)?")

RATE_WORDS = ("环比", "变化率", "增幅", "降幅", "涨幅", "同比")
BASE_WORDS = ("上期", "上月", "基期", "基准", "去年同期")
CUR_WORDS = ("本期", "当前", "现在", "本月")
CHANGE_WORDS = ("上涨", "下降", "增加", "减少", "涨了", "降了", "上升", "下滑")
CLAIM_WORDS = ("主要原因是", "根本原因是", "主因", "导致", "引起", "因为", "由于", "建议", "主要关联因素", "表明", "说明", "属于", "主要问题")
ATTRIB_WORDS = ("主要原因是", "根本原因", "主因", "确认是")


def sha8(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。；\n]", text) if s.strip()]


def value_pool(record: dict) -> list[tuple[str, float]]:
    """收集 record 里所有 (角色, 数值)。"""
    pool: list[tuple[str, float]] = []
    keys = ("primary_anomaly", "primary_reason")
    reasons = [record.get(k) for k in keys]
    reasons += record.get("contributing_reasons") or []
    for item in reasons:
        if not isinstance(item, dict):
            continue
        for role in ("current_value", "baseline_value", "change_value", "change_rate"):
            if isinstance(item.get(role), (int, float)):
                pool.append((role, float(item[role])))
        for value in (item.get("evidence") or {}).values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                pool.append(("evidence", float(value)))
        if isinstance(item.get("threshold"), (int, float)):
            pool.append(("threshold", float(item["threshold"])))
    def _collect(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for value in node:
                _collect(value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            pool.append(("snapshot", float(node)))

    _collect(record.get("metric_snapshot") or {})
    return pool


def expected_role(sentence: str) -> str | None:
    if any(w in sentence for w in RATE_WORDS):
        return "change_rate"
    if any(w in sentence for w in BASE_WORDS):
        return "baseline_value"
    if any(w in sentence for w in CUR_WORDS):
        return "current_value"
    if any(w in sentence for w in CHANGE_WORDS):
        return "change_value"
    return None


def name_variants(name: str) -> set[str]:
    """泊车Clip数 -> {泊车Clip数, Clip数}；行车单Clip成本 -> {行车单Clip成本, 单Clip成本}"""
    if not name:
        return set()
    variants = {name}
    for prefix in ("泊车", "行车"):
        if name.startswith(prefix):
            variants.add(name[len(prefix):])
    return variants


def match_number(
    value: float,
    pool: list[tuple[str, float]],
    allow_scaled: bool = True,
) -> list[tuple[str, float, bool]]:
    """返回 [(角色, 候选值, 是否经过换算)]。正负号不敏感（"下降25.30%" vs -0.2530）。"""
    hits = []
    probes = (value,) if value == 0 else (value, -value)
    for role, candidate in pool:
        for probe in probes:
            if candidate == 0:
                if abs(probe) < 1e-9:
                    hits.append((role, candidate, False))
                continue
            tol = max(0.01, abs(candidate) * 0.001)
            if abs(probe - candidate) <= tol:
                hits.append((role, candidate, False))
                continue
            if not allow_scaled:
                continue
            for scaled in (candidate * 100, candidate / 100):
                if scaled == 0:
                    continue
                if abs(probe - scaled) <= max(0.01, abs(scaled) * 0.001):
                    hits.append((role, candidate, True))
    return hits


def role_for_number(sentence: str, start: int, has_pct: bool) -> str | None:
    """只看该数字所在的「逗号分句」内、且紧邻的前 12 个字符。"""
    prefix = sentence[:start]
    cut = max(
        prefix.rfind("，"), prefix.rfind(","), prefix.rfind("、"),
        prefix.rfind("；"), prefix.rfind(";"),
    )
    window = (prefix[cut + 1:] if cut != -1 else prefix)[-12:]

    if has_pct:
        return "change_rate"
    if any(w in window for w in BASE_WORDS):
        return "baseline_value"
    if any(w in window for w in ("上涨", "下降", "增加", "减少", "涨了", "降了", "上升", "下滑")):
        return "change_value"
    if any(w in window for w in CUR_WORDS) or "为" in window or "是" in window:
        return "current_value"
    return None


def judge_slots(answer: str, record: dict, errors: list[dict]) -> list[dict]:
    answer = re.sub(r"[ABK]\d{2,3}", " ", answer)
    slots: list[dict] = []
    pool = value_pool(record)

    # S1 期间
    years = PERIOD_RE.findall(answer)
    months = MONTH_RE.findall(answer)
    allowed = {record.get("period")} | set(record.get("baseline_periods") or [])
    allowed_months = {str(a).split("-")[-1] for a in allowed if a}
    for month in months:
        ok = month.zfill(2) in allowed_months
        slots.append({"slot": "S1期间", "value": f"{month}月", "ok": ok})
        if not ok:
            errors.append({"type": "E1期间错", "detail": f"提到{month}月，数据是{record.get('period')}"})
    for year, month in years:
        ok = f"{year}-{month.zfill(2)}" in allowed
        slots.append({"slot": "S1期间", "value": f"{year}-{month.zfill(2)}", "ok": ok})
        if not ok:
            errors.append({"type": "E1期间错", "detail": f"提到{year}-{month}，数据是{record.get('period')}"})

    # 剥离日期，避免 2026-02 被当成数字 2026 / -02（S1 已在上面判完）
    answer = PERIOD_RE.sub(" ", answer)
    answer = re.sub(r"(?<!\d)(20\d{2})(?=\s*年)", " ", answer)

    # S2 业务
    business = record.get("business_type")
    mentioned = [b for b in ("泊车", "行车", "整体") if b in answer]
    for item in mentioned:
        ok = item == business or business == "整体"
        slots.append({"slot": "S2业务", "value": item, "ok": ok})
        if not ok:
            errors.append({"type": "E2业务错", "detail": f"提到{item}，数据是{business}"})

    # S4 数值角色 + S5 标度
    for sentence in split_sentences(answer):
        for match in NUM_RE.finditer(sentence):
            raw = match.group(1)
            pct = match.group(2)
            try:
                value = float(raw)
            except ValueError:
                continue
            if abs(value) < 1 and not pct:
                continue
            tail = sentence[match.end():match.end() + 3]
            if tail.startswith("个月") or tail.startswith("月") or tail.startswith("次") or tail.startswith("项"):
                continue
            want = role_for_number(sentence, match.start(), bool(pct))
            hits = match_number(value, pool, allow_scaled=bool(pct))
            if not hits:
                slots.append({"slot": "S4数值", "value": raw, "ok": False})
                errors.append({"type": "E4数值角色错", "detail": f"{raw} 不在 JSON 数值池中"})
                continue

            ok = True
            if want:
                ok = any(role == want for role, _, _ in hits)
            slots.append({"slot": "S4数值", "value": raw, "ok": ok})
            if not ok:
                roles = sorted({role for role, _, _ in hits})
                errors.append(
                    {"type": "E4数值角色错", "detail": f"{raw} 期望{want}，实际是{roles}"}
                )

            if pct and abs(value) < 1 and any(
                role == "change_rate" and not scaled for role, _, scaled in hits
            ):
                slots.append({"slot": "S5标度", "value": raw + "%", "ok": False})
                errors.append({"type": "E5标度错", "detail": f"{raw}% 应为 {value * 100:.2f}%"})
    return slots


def judge_evidence(answer: str, record: dict, errors: list[dict]) -> tuple[int, int]:
    def keywords(item: dict) -> set[str]:
        metric = reason_metric(item, record.get("business_type"))
        return name_variants(CMAP.get(metric or "", ""))

    primary = keywords(record.get("primary_reason") or {})
    contributing: set[str] = set()
    for item in record.get("contributing_reasons") or []:
        contributing |= keywords(item)

    supported = 0
    total = 0
    for sentence in split_sentences(answer):
        if not any(w in sentence for w in CLAIM_WORDS):
            continue
        total += 1
        hit_primary = any(k in sentence for k in primary)
        hit_contrib = any(k in sentence for k in contributing)
        if hit_primary:
            supported += 1
        elif hit_contrib:
            if any(w in sentence for w in ATTRIB_WORDS):
                errors.append(
                    {"type": "E6越权归因", "detail": f"把关联因素说成根因：{sentence[:40]}"}
                )
            else:
                supported += 1
        else:
            errors.append({"type": "E7无证据结论", "detail": sentence[:40]})
    return supported, total


def judge_coverage(answer: str, record: dict) -> tuple[int, int]:
    business = record.get("business_type")
    items: list[str] = []
    for key in ("primary_anomaly", "primary_reason"):
        item = record.get(key) or {}
        name = CMAP.get(reason_metric(item, business) or "")
        items.append(name if name else "【未解析指标】")
    for item in record.get("contributing_reasons") or []:
        if item.get("severity") == "critical":
            name = CMAP.get(reason_metric(item, business) or "")
            items.append(name if name else "【未解析指标】")
    items = list(dict.fromkeys(items))
    covered = sum(1 for name in items if any(v in answer for v in name_variants(name)))
    return covered, len(items)


def build_context(collection, embedder, question: str, rule_ids: list[str]) -> dict:
    vector = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    kwargs = {
        "query_embeddings": [vector],
        "n_results": 2,
        "include": ["metadatas", "documents"],
    }
    if rule_ids:
        clauses = [{"related_rule_ids": {"$contains": r}} for r in rule_ids]
        clauses.append({"related_rule_ids": {"$contains": "*"}})
        kwargs["where"] = {"$or": clauses}
    hits = collection.query(**kwargs)
    return {
        "排查参考": [
            {"knowledge_id": meta["knowledge_id"], "content": doc}
            for doc, meta in zip(hits["documents"][0], hits["metadatas"][0])
        ]
    }


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "—"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diagnosis", "events"], default="diagnosis")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    is_diag = args.mode == "diagnosis"
    rows = load_jsonl(DIAG_PATH if is_diag else EVENT_PATH)
    print(f"模式：{args.mode}，样本 {len(rows)} 条")

    embedder = SentenceTransformer(EMBED_PATH)
    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    print("加载 Qwen2.5-3B-Instruct（4bit）……")
    tokenizer, model = load_generator()

    details = []
    for index, row in enumerate(rows, start=1):
        if is_diag:
            question = (
                f"{row['period']} {row['business_type']} 这个月的成本效率异常，"
                "主要问题是什么，依据是什么？"
            )
            context = {"诊断结果": row}
            context.update(build_context(collection, embedder, question,
                                         row.get("triggered_rule_ids") or []))
        else:
            question = (
                f"{row['period']} {row.get('business_type')} 为什么触发了"
                f"「{row['rule_name']}」？"
            )
            context = {"事件": row}
            context.update(build_context(collection, embedder, question,
                                         [row["rule_id"]] if row.get("rule_id") else []))

        answer = generate(tokenizer, model, question, context)

        errors: list[dict] = []
        slots = judge_slots(answer, row, errors)
        covered, total_cover = judge_coverage(answer, row)
        if covered < total_cover:
            errors.append({"type": "E8关键事实遗漏", "detail": f"{covered}/{total_cover}"})

        supported, claims = (judge_evidence(answer, row, errors) if is_diag else (0, 0))

        detail = {
            "id": row.get("diagnosis_id") or f"{row.get('period')}_{row.get('rule_id')}",
            "question": question,
            "answer": answer,
            "slots_ok": sum(1 for s in slots if s["ok"]),
            "slots_total": len(slots),
            "coverage": covered,
            "coverage_total": total_cover,
            "evidence_ok": supported,
            "claims": claims,
            "errors": errors,
        }
        details.append(detail)
        print(
            f"[{index}/{len(rows)}] {detail['id']}  槽位 {detail['slots_ok']}/{detail['slots_total']}"
            f"  覆盖 {covered}/{total_cover}  错误 {len(errors)}"
        )

    slots_ok = sum(d["slots_ok"] for d in details)
    slots_total = sum(d["slots_total"] for d in details)
    cov_ok = sum(d["coverage"] for d in details)
    cov_total = sum(d["coverage_total"] for d in details)
    ev_ok = sum(d["evidence_ok"] for d in details)
    claims_total = sum(d["claims"] for d in details)

    type_counts: dict[str, int] = {}
    for d in details:
        for e in d["errors"]:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

    lines = ["# RAG 事实质量评测 v1", "", "## 实验配置", "", "| 项 | 值 |", "|---|---|"]
    for key, value in {
        "模式": "诊断（端到端）" if is_diag else "事件（rule 级）",
        "样本数": len(rows),
        "数据版本": f"{sha8(DIAG_PATH if is_diag else EVENT_PATH)}",
        "生成模型": QWEN_PATH + "（4bit nf4, do_sample=False）",
        "提示词": "gen-v2（与生成评测同一份）",
        "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }.items():
        lines.append(f"| {key} | {value} |")

    lines += ["", "## 三个指标", "", "| 指标 | 结果 | 分母 |", "|---|---|---|"]
    lines.append(f"| 事实槽位准确率 | **{pct(slots_ok, slots_total)}** | {slots_ok}/{slots_total} |")
    lines.append(f"| 关键事实覆盖率 | **{pct(cov_ok, cov_total)}** | {cov_ok}/{cov_total} |")
    if is_diag:
        lines.append(f"| 证据支持率 | **{pct(ev_ok, claims_total)}** | {ev_ok}/{claims_total} |")
    lines.append("")

    lines += ["## 错误分类清单", "", "| 类型 | 条次 |", "|---|---|"]
    for kind, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {kind} | {count} |")
    if not type_counts:
        lines.append("| 无 | 0 |")

    lines += ["", "## 逐条", "", "| 记录 | 槽位 | 覆盖 | 证据 | 错误数 |", "|---|---|---|---|---|"]
    for d in details:
        lines.append(
            f"| {d['id']} | {d['slots_ok']}/{d['slots_total']} | "
            f"{d['coverage']}/{d['coverage_total']} | {d['evidence_ok']}/{d['claims']} | "
            f"{len(d['errors'])} |"
        )

    lines += ["", "## 原始输出", ""]
    for d in details:
        lines += [f"### {d['id']}", "", "```", d["answer"].strip(), "```", ""]

    report = "\n".join(lines)
    print()
    print(f"槽位准确率 {pct(slots_ok, slots_total)} ({slots_ok}/{slots_total})")
    print(f"覆盖率     {pct(cov_ok, cov_total)} ({cov_ok}/{cov_total})")
    if is_diag:
        print(f"证据支持率 {pct(ev_ok, claims_total)} ({ev_ok}/{claims_total})")
    print("错误分布：", type_counts)

    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_OUT.open("a", encoding="utf-8") as file:
        for d in details:
            file.write(json.dumps({"mode": args.mode, **d}, ensure_ascii=False) + "\n")

    if args.save:
        target = REPORT_DIAG if is_diag else REPORT_EVENT
        target.write_text(report, encoding="utf-8")
        print(f"已写入：{target}")


if __name__ == "__main__":
    main()
