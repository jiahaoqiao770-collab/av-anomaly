#!/usr/bin/env python3
"""RAG 生成层评测：数字一致率（口径 B：归一化数值 + 百分比互换）。

只测一件事：给定结构化诊断 JSON，模型输出里的每个数值是否都能在 JSON 中找到。
不测事实准确率与拒答（下一步）。

口径 B 的判定：
- 输出数值 v 若等于 JSON 中某值 c（相对容差 0.1%）        -> 一致
- 或 v 等于 c*100 / c/100（百分比与小数互换）              -> 一致（记 scaled）
- 否则                                                    -> 判为幻觉

排除规则（否则误报满天飞）：
- 日期 2026-02 / 2026年2月
- 规则/知识 ID A01 B02 K009
- 列表序号 "1." "2、"

运行：
  python -m scripts.evaluate_rag_generation --limit 8
  python -m scripts.evaluate_rag_generation --limit 8 --save
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIAG_PATH = PROJECT_ROOT / "data/processed/monthly_cost_diagnosis_results.jsonl"
INDEX_PATH = PROJECT_ROOT / "data/knowledge/vector_store/monthly_cost_troubleshooting_chroma_v2"
COLLECTION_NAME = "monthly_cost_troubleshooting_v2"
EMBED_PATH = "/home/qiaojiahao/models/bge-small-zh-v1.5"
QWEN_PATH = "/home/qiaojiahao/models/Qwen2.5-3B-Instruct"
REPORT_PATH = PROJECT_ROOT / "docs/rag-gen-eval-v1.md"

MAX_NEW_TOKENS = 400

SYSTEM_PROMPT = (
    "你是采集成本效率分析助手。"
    "查询结果中的数据是事实。"
    "RAG知识只是排查参考。"
    "不得编造数值或已确认根因。"
    "使用简洁中文直接回答。"
    "输出必须遵守以下硬规则："
    "一、只能逐字引用「诊断结果」中已有的数值，不得自行加减、换算、重算或推断任何新数值。"
    "二、禁止使用「从A到B」「由A升至B」这类区间描述；"
    "只允许「指标名 + 本期数值 + 环比变化率」的写法，数值必须逐字来自数据。"
    "三、不得输出超过两位小数的数值，不得输出浮点尾数。"
    "四、若参考了「排查参考」中的知识卡，必须在回答最后单列一行：依据知识卡：K00x。"
)

DATE_RE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*月?")
ID_RE = re.compile(r"\b[ABK]\d{2,3}\b")
LIST_RE = re.compile(r"(?m)^\s*\d{1,2}\s*[.、)）]\s*")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def sha8(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def collect_numbers(obj, acc: set[float]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            collect_numbers(value, acc)
    elif isinstance(obj, list):
        for value in obj:
            collect_numbers(value, acc)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        for match in NUM_RE.finditer(obj):
            acc.add(float(match.group(0)))


def extract_numbers(text: str) -> list[float]:
    cleaned = DATE_RE.sub(" ", text)
    cleaned = ID_RE.sub(" ", cleaned)
    cleaned = LIST_RE.sub("", cleaned)
    return [float(m.group(0)) for m in NUM_RE.finditer(cleaned)]


def match_number(value: float, pool: set[float]) -> str | None:
    for candidate in pool:
        if candidate == 0:
            if abs(value) < 1e-9:
                return "exact"
            continue
        tolerance = max(0.01, abs(candidate) * 0.001)
        if abs(value - candidate) <= tolerance:
            return "exact"
        for scaled in (candidate * 100, candidate / 100):
            if scaled == 0:
                continue
            if abs(value - scaled) <= max(0.01, abs(scaled) * 0.001):
                return "scaled"
    return None


def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        quantization_config=quantization,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def build_rag_context(collection, embedder, record: dict) -> list[dict]:
    rule_ids = record.get("triggered_rule_ids") or []
    question = (
        f"{record['period']} {record['business_type']} 这个月的成本效率异常，"
        "主要问题是什么，依据是什么？"
    )
    vector = embedder.encode([question], normalize_embeddings=True)[0].tolist()

    clauses = [{"related_rule_ids": {"$contains": rid}} for rid in rule_ids]
    clauses.append({"related_rule_ids": {"$contains": "*"}})
    kwargs = {
        "query_embeddings": [vector],
        "n_results": 2,
        "include": ["metadatas", "documents", "distances"],
    }
    if rule_ids:
        kwargs["where"] = {"$or": clauses}

    result = collection.query(**kwargs)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    return [
        {"knowledge_id": meta["knowledge_id"], "content": doc}
        for doc, meta in zip(documents, metadatas)
    ]


def generate(tokenizer, model, question: str, context: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                "查询上下文：\n"
                + json.dumps(context, ensure_ascii=False, default=str)
            ),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )



FLOAT_LEAK_RE = re.compile(r"-?\d+\.\d{5,}")
RANGE_RE = re.compile(r"从\s*(-?\d+(?:\.\d+)?)\s*[^，。；\n]{0,6}?到\s*(-?\d+(?:\.\d+)?)")
VERB_NUM_RE = re.compile(
    r"(环比上涨|环比下降|增长|上涨|上升|下降|降低|下滑)(?:了|达|约|为)?\s*(-?\d+(?:\.\d+)?)\s*%"
)
UP_WORDS = ("上涨", "增长", "上升", "增加", "提高", "升至")
DOWN_WORDS = ("下降", "降低", "减少", "下滑", "降至")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。；\n]", text)
    return [part.strip() for part in parts if part.strip()]


def run_checks(answer: str, knowledge_ids: list[str]) -> list[dict]:
    defects: list[dict] = []

    for match in FLOAT_LEAK_RE.finditer(answer):
        defects.append({"type": "浮点泄漏", "detail": match.group(0)})

    for sentence in split_sentences(answer):
        for match in RANGE_RE.finditer(sentence):
            start, end = float(match.group(1)), float(match.group(2))
            if end < start and any(word in sentence for word in UP_WORDS):
                defects.append(
                    {"type": "区间方向矛盾", "detail": f"{start}->{end} 却描述上涨：{sentence[:50]}"}
                )
            elif end > start and any(word in sentence for word in DOWN_WORDS):
                defects.append(
                    {"type": "区间方向矛盾", "detail": f"{start}->{end} 却描述下降：{sentence[:50]}"}
                )

        seen: dict[str, set[float]] = {}
        for match in VERB_NUM_RE.finditer(sentence):
            seen.setdefault(match.group(1), set()).add(float(match.group(2)))
        for verb, values in seen.items():
            if len(values) >= 2:
                defects.append({"type": "同谓词多值", "detail": f"{verb} 出现 {sorted(values)}"})

    if knowledge_ids and not any(kid in answer for kid in knowledge_ids):
        defects.append({"type": "未引用知识卡", "detail": "输出未出现任何召回 knowledge_id"})

    return defects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=17)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(DIAG_PATH)[: args.limit]
    print(f"诊断记录：{len(records)} 条（共 {len(load_jsonl(DIAG_PATH))} 条）")

    embedder = SentenceTransformer(EMBED_PATH)
    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    print("加载 Qwen2.5-3B-Instruct（4bit）……")
    tokenizer, model = load_generator()

    rows = []
    for index, record in enumerate(records, start=1):
        rag = build_rag_context(collection, embedder, record)
        question = (
            f"{record['period']} {record['business_type']} 这个月的成本效率异常，"
            "主要问题是什么，依据是什么？"
        )
        context = {"诊断结果": record, "排查参考": rag}
        answer = generate(tokenizer, model, question, context)

        pool: set[float] = set()
        collect_numbers(record, pool)

        numbers = extract_numbers(answer)
        details = [(value, match_number(value, pool)) for value in numbers]
        hits = [d for d in details if d[1] is not None]
        rate = len(hits) / len(numbers) if numbers else 1.0
        knowledge_ids = [item["knowledge_id"] for item in rag]
        defects = run_checks(answer, knowledge_ids)

        rows.append(
            {
                "diagnosis_id": record["diagnosis_id"],
                "period": record["period"],
                "business": record["business_type"],
                "knowledge_ids": [item["knowledge_id"] for item in rag],
                "answer": answer,
                "numbers": details,
                "rate": rate,
                "defects": defects,
            }
        )
        print(
            f"[{index}/{len(records)}] {record['diagnosis_id']} "
            f"数字 {len(hits)}/{len(numbers)} = {rate * 100:.1f}%  "
            f"召回卡 {[item['knowledge_id'] for item in rag]}"
        )

    total = sum(len(r["numbers"]) for r in rows)
    total_hits = sum(len([d for d in r["numbers"] if d[1]]) for r in rows)
    overall = total_hits / total if total else 1.0

    lines = []
    lines.append("# RAG 生成层评测：数字一致率 v1")
    lines.append("")
    lines.append("## 实验配置")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for key, value in {
        "诊断数据": f"monthly_cost_diagnosis_results.jsonl @ {sha8(DIAG_PATH)}",
        "知识库索引": COLLECTION_NAME,
        "生成模型": QWEN_PATH + "（4bit nf4, do_sample=False）",
        "提示词版本": "gen-v2：禁自算 + 禁区间句式 + 强制引用知识卡",
        "最大新token": MAX_NEW_TOKENS,
        "评测条数": len(rows),
        "抽取数值总数": total,
        "口径": "B：归一化数值 + 百分比互换，相对容差 0.1%",
        "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }.items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## 总体")
    lines.append("")
    lines.append(f"- 数字一致率：**{overall * 100:.1f}%**（{total_hits}/{total}）")
    lines.append(f"- 完全一致的条数：{sum(1 for r in rows if r['rate'] == 1.0)}/{len(rows)}")
    defect_rows = [r for r in rows if r["defects"]]
    lines.append(
        f"- 一致性缺陷率：**{len(defect_rows) / len(rows) * 100:.1f}%**"
        f"（{len(defect_rows)}/{len(rows)} 条存在至少一处缺陷）"
    )
    by_type: dict[str, int] = {}
    for row in rows:
        for defect in row["defects"]:
            by_type[defect["type"]] = by_type.get(defect["type"], 0) + 1
    lines.append("")
    lines.append("| 缺陷类型 | 条次 |")
    lines.append("|---|---|")
    for kind, count in sorted(by_type.items(), key=lambda item: -item[1]):
        lines.append(f"| {kind} | {count} |")

    lines.append("")
    lines.append("## 逐条")
    lines.append("")
    lines.append("| 诊断 | 期间 | 业务 | 召回卡 | 数字一致率 | 幻觉数值 | 缺陷数 |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        misses = [d[0] for d in row["numbers"] if d[1] is None]
        lines.append(
            f"| {row['diagnosis_id']} | {row['period']} | {row['business']} | "
            f"{'、'.join(row['knowledge_ids'])} | {row['rate'] * 100:.1f}% | "
            f"{misses if misses else '—'} | {len(row['defects'])} |"
        )

    lines.append("")
    lines.append("## 原始输出")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['diagnosis_id']}")
        lines.append("")
        lines.append("```")
        lines.append(row["answer"].strip())
        lines.append("```")
        lines.append("")

    report = "\n".join(lines)
    print()
    print(f"总体数字一致率：{overall * 100:.1f}%  ({total_hits}/{total})")

    if args.save:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"已写入：{REPORT_PATH}")


if __name__ == "__main__":
    main()
