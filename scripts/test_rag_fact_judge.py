#!/usr/bin/env python3
"""判定器自测：用人工构造的答案考尺子，与模型输出无关。

正例（人工确认正确的答案）必须 0 误判；负例（故意埋错）必须被抓到。
任何一条 FAIL，说明判定器有 bug —— 改判定器是有依据的。
不加载模型，秒级完成。
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.evaluate_rag_facts import (
    CMAP,
    DIAG_PATH,
    judge_coverage,
    judge_evidence,
    judge_slots,
    load_jsonl,
)
from scripts.evaluate_rag_facts import name_variants, reason_metric

RECORD_ID = "2026-08_parking"


def coverage_items(rec: dict) -> list[str]:
    items: list[str] = []
    business = rec.get("business_type")
    for key in ("primary_anomaly", "primary_reason"):
        name = CMAP.get(reason_metric(rec.get(key) or {}, business) or "")
        if name:
            items.append(name)
    for item in rec.get("contributing_reasons") or []:
        if item.get("severity") == "critical":
            name = CMAP.get(reason_metric(item, business) or "")
            if name:
                items.append(name)
    return list(dict.fromkeys(items))


def run_case(
    name: str,
    answer: str,
    rec: dict,
    must_have: set[str],
    must_not: set[str],
    min_slots: int = 0,
) -> bool:
    errors: list[str] = []
    slots = judge_slots(answer, rec, errors)

    cov_errors: list[dict] = []
    covered, total = judge_coverage(answer, rec)
    if covered < total:
        cov_errors.append({"type": "E8关键事实遗漏"})

    ev_errors: list[dict] = []
    judge_evidence(answer, rec, ev_errors)

    kinds = {e["type"] for e in errors + cov_errors + ev_errors}

    ok = (
        must_have.issubset(kinds)
        and not (must_not & kinds)
        and len(slots) >= min_slots
    )
    status = "PASS" if ok else "FAIL"
    detail = ""
    if not ok:
        detail = f"  期望至少含 {sorted(must_have)}，不该含 {sorted(must_not)}；实际 {sorted(kinds)}"
    print(f"[{status}] {name}{detail}")
    return ok


def main() -> None:
    records = load_jsonl(DIAG_PATH)
    rec = next(r for r in records if r["diagnosis_id"] == RECORD_ID)

    pa = rec["primary_anomaly"]
    pr = rec["primary_reason"]
    cur = float(pa["current_value"])
    base = float(pa["baseline_value"])
    rate = float(pa["change_rate"])
    cn = CMAP[pa["evidence"]["metric"]]
    names = coverage_items(rec)
    name_line = "涉及指标：" + "、".join(names) + "。"

    contrib_metric = None
    for item in rec.get("contributing_reasons") or []:
        name = CMAP.get(reason_metric(item, rec["business_type"]) or "")
        if name:
            contrib_metric = name
            break

    results = []

    # G01 完全正确
    g01 = (
        f"{cn}本期为{cur:.2f}元，上期为{base:.2f}元，环比上涨{rate * 100:.2f}%。"
        f"主要原因是{CMAP.get(reason_metric(pr, rec['business_type']) or '', '')}上涨。" + name_line
    )
    results.append(
        run_case(
            "G01 正例：字段、角色、单位全对",
            g01,
            rec,
            set(),
            {"E4数值角色错", "E5标度错", "E6越权归因", "E7无证据结论"},
            min_slots=3,
        )
    )

    # G02 把变化率当本期值
    g02 = f"{cn}本期为{rate * 100:.2f}元。"
    results.append(run_case("G02 负例：变化率当本期值", g02, rec, {"E4数值角色错"}, {"E5标度错"}))

    # G03 标度错：0.1445 写成 0.1445%
    g03 = f"{cn}环比上涨{rate:.4f}%。"
    results.append(run_case("G03 负例：标度错(0.1445%)", g03, rec, {"E5标度错"}, set()))

    # G04 无证据结论
    g04 = "主要原因是采集人员流失严重。"
    results.append(run_case("G04 负例：编造无证据结论", g04, rec, {"E7无证据结论"}, set()))

    # G05 越权归因：关联因素说成根本原因（用合成 contributing，保证一定跑得到）
    rec_g05 = dict(rec)
    rec_g05["contributing_reasons"] = [
        {
            "severity": "critical",
            "rule_id": "A10",
            "evidence": {"clip_count_change_rate": 0.01},
        }
    ]
    g05_name = CMAP.get(
        reason_metric(rec_g05["contributing_reasons"][0], rec.get("business_type")) or ""
    )
    if not g05_name:
        print("[FAIL] G05 无法解析出关联因素指标名，判定器的 reason_metric 仍有缺口")
        results.append(False)
    else:
        g05 = f"根本原因是{g05_name}上涨。"
        results.append(
            run_case("G05 负例：关联因素说成根因", g05, rec_g05, {"E6越权归因"}, {"E7无证据结论"})
        )

    # G06 关键事实遗漏
    g06 = "本期情况需要进一步观察。"
    results.append(run_case("G06 负例：关键事实遗漏", g06, rec, {"E8关键事实遗漏"}, set()))

    # G07 期间错
    g07 = f"2026-05 {cn}本期为{cur:.2f}元。" + name_line
    results.append(run_case("G07 负例：期间说错", g07, rec, {"E1期间错"}, set()))

    # G08 业务错
    g08 = f"行车{cn}本期为{cur:.2f}元。" + name_line
    results.append(run_case("G08 负例：业务说错", g08, rec, {"E2业务错"}, set()))

    # G09 正例：知识卡 ID 不该被当成数字
    g09 = f"{cn}本期为{cur:.2f}元。依据知识卡：K001,K003。" + name_line
    results.append(
        run_case(
            "G09 正例：ID 不产生假数值错",
            g09,
            rec,
            set(),
            {"E4数值角色错"},
            min_slots=2,
        )
    )

    passed = sum(1 for r in results if r)
    print()
    print(f"通过 {passed}/{len(results)}")
    if passed != len(results):
        print("判定器有缺陷，先修判定器，不要拿模型结果下结论。")
        sys.exit(1)
    print("判定器自测通过，模型评测结论可以引用。")


if __name__ == "__main__":
    main()
