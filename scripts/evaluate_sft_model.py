#!/usr/bin/env python3
"""在 SFT 锁定测试集上评估模型（base 或 LoRA adapter）。

与 RAG 评测的区别：这里不走检索，直接用 test.jsonl 里固定的 system+user 提示，
隔离"模型自身能力"，正是 SFT 对照需要的口径。

运行：
  python -m scripts.evaluate_sft_model --tag base
  python -m scripts.evaluate_sft_model --tag sft_v2 --adapter data/sft/output/sft_v2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch

from scripts.evaluate_rag_facts import judge_coverage, judge_slots, load_jsonl
from scripts.evaluate_rag_generation import load_generator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data/sft/v2/test.jsonl"
DIAG_PATH = PROJECT_ROOT / "data/processed/monthly_cost_diagnosis_results.jsonl"
OUT_DIR = PROJECT_ROOT / "docs"


def generate_from_messages(tokenizer, model, messages: list[dict], max_new_tokens: int = 300) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="实验标签，用于报告文件名")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录，不填则为 base")
    parser.add_argument("--test-path", default=str(TEST_PATH), help="测试集路径")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.test_path))
    records = {r["diagnosis_id"]: r for r in load_jsonl(DIAG_PATH)}
    print(f"测试样本 {len(rows)} 条，模型：{args.adapter or 'base'}")

    tokenizer, model = load_generator()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"已加载 LoRA：{args.adapter}")

    details = []
    for index, row in enumerate(rows, start=1):
        messages = row["messages"][:2]
        answer = generate_from_messages(tokenizer, model, messages)
        record = records.get(row["meta"].get("diagnosis_id"))

        if record is None:
            ok = any(w in answer for w in ("无异常", "未触发", "正常")) and not any(
                ch.isdigit() for ch in answer
            )
            detail = {"id": row["meta"]["diagnosis_id"], "type": row["meta"]["type"],
                      "slots": "-", "coverage": "-", "manual_ok": ok,
                      "errors": [] if ok else ["T5判定失败"], "answer": answer}
        else:
            errors: list[dict] = []
            slots = judge_slots(answer, record, errors)
            covered, total = judge_coverage(answer, record)
            if covered < total:
                errors.append({"type": "E8关键事实遗漏", "detail": f"{covered}/{total}"})
            detail = {
                "id": row["meta"]["diagnosis_id"],
                "type": row["meta"]["type"],
                "slots": f"{sum(1 for s in slots if s['ok'])}/{len(slots)}",
                "slots_ok": sum(1 for s in slots if s["ok"]),
                "slots_total": len(slots),
                "coverage": f"{covered}/{total}",
                "cov_ok": covered,
                "cov_total": total,
                "manual_ok": None,
                "errors": errors,
                "answer": answer,
            }
        details.append(detail)
        print(f"[{index}/{len(rows)}] {detail['id']} {detail['type']} "
              f"槽位 {detail['slots']} 覆盖 {detail['coverage']} 错误 {len(detail['errors'])}")

    slot_ok = sum(d.get("slots_ok", 0) for d in details)
    slot_total = sum(d.get("slots_total", 0) for d in details)
    cov_ok = sum(d.get("cov_ok", 0) for d in details)
    cov_total = sum(d.get("cov_total", 0) for d in details)

    lines = [f"# SFT 测试集评估：{args.tag}", "", "| 项 | 值 |", "|---|---|"]
    for key, value in {
        "模型": args.adapter or "base Qwen2.5-3B-Instruct",
        "测试样本": len(rows),
        "槽位准确率": f"{slot_ok/slot_total*100:.1f}% ({slot_ok}/{slot_total})" if slot_total else "—",
        "覆盖率": f"{cov_ok/cov_total*100:.1f}% ({cov_ok}/{cov_total})" if cov_total else "—",
        "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## 逐条", "", "| 记录 | 类型 | 槽位 | 覆盖 | 错误数 |", "|---|---|---|---|---|"]
    for d in details:
        lines.append(f"| {d['id']} | {d['type']} | {d['slots']} | {d['coverage']} | {len(d['errors'])} |")
    lines += ["", "## 原始输出", ""]
    for d in details:
        lines += [f"### {d['id']} ({d['type']})", "", "```", d["answer"].strip(), "```", ""]

    report = "\n".join(lines)
    target = OUT_DIR / f"sft-eval-{args.tag}.md"
    target.write_text(report, encoding="utf-8")
    print()
    print(f"槽位 {slot_ok}/{slot_total}  覆盖 {cov_ok}/{cov_total}")
    print(f"已写入：{target}")


if __name__ == "__main__":
    main()
