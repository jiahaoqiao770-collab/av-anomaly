#!/usr/bin/env python3
"""RAG 生成层评测：拒答准确率。

两种模式：
- 默认         : 只调 answer 层（给两张无关知识卡，不给任何业务数据）
                 测的是"第二道防线"——生成层自己会不会拒
- --end-to-end : 走完整 graph（plan → query → rag → answer），像用户真提问一样
                 测的是"产品真实拒答率"，拒答由 plan_node 的 intent 判定负责

判定四类：明确拒答 / 拒答但夹带数值 / 疑似硬答 / 含糊
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from scripts.evaluate_rag_generation import (
    COLLECTION_NAME,
    EMBED_PATH,
    INDEX_PATH,
    QWEN_PATH,
    extract_numbers,
    generate,
    load_generator,
    load_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDENSET_PATH = PROJECT_ROOT / "data/knowledge/eval/rag_goldenset_v1.jsonl"
REPORT_SINGLE = PROJECT_ROOT / "docs/rag-refusal-eval-v1.md"
REPORT_E2E = PROJECT_ROOT / "docs/rag-refusal-eval-v1-e2e.md"

REFUSAL_WORDS = (
    "不属于", "超出", "无法回答", "无法提供", "无法查询", "仅支持", "只服务",
    "不能回答", "不涉及", "抱歉", "不在", "无法判断", "没有相关",
)


def sha8(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def classify(answer: str) -> str:
    refused = any(word in answer for word in REFUSAL_WORDS)
    has_numbers = bool(extract_numbers(answer))
    if refused and not has_numbers:
        return "明确拒答"
    if refused and has_numbers:
        return "拒答但夹带数值"
    if has_numbers:
        return "疑似硬答"
    return "含糊"


def run_single_layer(rows: list[dict]) -> list[dict]:
    embedder = SentenceTransformer(EMBED_PATH)
    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    print("加载 Qwen2.5-3B-Instruct（4bit）……")
    tokenizer, model = load_generator()

    results = []
    for index, row in enumerate(rows, start=1):
        vector = embedder.encode([row["query"]], normalize_embeddings=True)[0].tolist()
        hits = collection.query(
            query_embeddings=[vector], n_results=2, include=["metadatas", "documents"]
        )
        rag = [
            {"knowledge_id": meta["knowledge_id"], "content": doc}
            for doc, meta in zip(hits["documents"][0], hits["metadatas"][0])
        ]
        answer = generate(tokenizer, model, row["query"], {"排查参考": rag})
        results.append(
            {
                "qid": row["qid"],
                "query": row["query"],
                "verdict": classify(answer),
                "intent": "—",
                "answer": answer,
            }
        )
        print(f"[{index}/{len(rows)}] {row['qid']} -> {results[-1]['verdict']}")
    return results


def run_end_to_end(rows: list[dict]) -> list[dict]:
    import scripts.run_monthly_cost_langgraph_assistant as production

    print("加载生产链路（plan → query → rag → answer）……")
    tokenizer, model = production.load_qwen()
    graph = production.build_graph(tokenizer, model)

    results = []
    for index, row in enumerate(rows, start=1):
        state = graph.invoke({"question": row["query"]})
        answer = state.get("answer", "") or ""
        plan = state.get("plan") or {}
        intent = plan.get("intent", "—")
        results.append(
            {
                "qid": row["qid"],
                "query": row["query"],
                "verdict": classify(answer),
                "intent": intent,
                "answer": answer,
            }
        )
        print(
            f"[{index}/{len(rows)}] {row['qid']}  intent={intent:<16} "
            f"-> {results[-1]['verdict']}"
        )
    return results


def write_report(results: list[dict], mode: str) -> float:
    ok = [r for r in results if r["verdict"] == "明确拒答"]
    rate = len(ok) / len(results) if results else 0.0

    lines = ["# RAG 拒答评测 v1", "", "## 实验配置", "", "| 项 | 值 |", "|---|---|"]
    for key, value in {
        "评测模式": mode,
        "GT 版本": f"rag_goldenset_v1.jsonl @ {sha8(GOLDENSET_PATH)}",
        "生成模型": QWEN_PATH + "（4bit nf4, do_sample=False）",
        "拒答样本数": len(results),
        "判定方式": "拒答用语关键词 + 是否给出数值（人工复核兜底）",
        "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }.items():
        lines.append(f"| {key} | {value} |")

    lines += ["", "## 总体", "", f"- 拒答准确率：**{rate * 100:.1f}%**（{len(ok)}/{len(results)}）", ""]
    lines += ["| 判定 | 条数 |", "|---|---|"]
    counts: dict[str, int] = {}
    for row in results:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    for verdict, count in sorted(counts.items(), key=lambda item: -item[1]):
        lines.append(f"| {verdict} | {count} |")

    if any(r["intent"] != "—" for r in results):
        lines += ["", "## 意图分类分布", "", "| intent | 条数 |", "|---|---|"]
        intents: dict[str, int] = {}
        for row in results:
            intents[row["intent"]] = intents.get(row["intent"], 0) + 1
        for intent, count in sorted(intents.items(), key=lambda item: -item[1]):
            lines.append(f"| {intent} | {count} |")

    lines += ["", "## 逐条", "", "| qid | 提问 | intent | 判定 |", "|---|---|---|---|"]
    for row in results:
        lines.append(f"| {row['qid']} | {row['query']} | {row['intent']} | {row['verdict']} |")

    lines += ["", "## 原始输出", ""]
    for row in results:
        lines += [f"### {row['qid']}", "", "```", row["answer"].strip(), "```", ""]

    report = "\n".join(lines)
    print(f"\n拒答准确率：{rate * 100:.1f}%  ({len(ok)}/{len(results)})")

    target = REPORT_E2E if mode.startswith("端到端") else REPORT_SINGLE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(f"已写入：{target}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-to-end", action="store_true", help="走完整 graph")
    args = parser.parse_args()

    rows = [r for r in load_jsonl(GOLDENSET_PATH) if r.get("expect_refusal")]
    print(f"拒答样本：{len(rows)} 条，模式：{'端到端' if args.end_to_end else '仅生成层'}")

    if args.end_to_end:
        results = run_end_to_end(rows)
        write_report(results, "端到端（plan → query → rag → answer）")
    else:
        results = run_single_layer(rows)
        write_report(results, "仅生成层（给无关知识卡，不给业务数据）")


if __name__ == "__main__":
    main()
