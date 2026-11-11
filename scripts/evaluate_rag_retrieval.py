#!/usr/bin/env python3
"""RAG 检索层评估：Recall@K / MRR / 干扰误召回 / Topic-level 聚合。

两条检索路径，都跑，都报：
- raw     ：纯向量 Top-K，不加重滤
- filtered：先用当月触发的 rule_ids 做 $contains 硬过滤（含通配 "*"），再排序

度量口径（定死，不可中途改）：
- Recall@K：primary_knowledge_ids 全部出现在 Top-K 才算命中（多卡母题从严）
- Recall@1：多主卡样本（2 张卡）在单槽下无定义，从分母剔除，记 N/A
- 拒答组：primary 为空，不参与 Recall，单独计数
- Topic-level：先按 seed_id 内部平均，再对母题平均

运行：
  python -m scripts.evaluate_rag_retrieval
  python -m scripts.evaluate_rag_retrieval --save
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_PATH = PROJECT_ROOT / "data/knowledge/raw/monthly_cost_troubleshooting_knowledge.jsonl"
GOLDENSET_PATH = PROJECT_ROOT / "data/knowledge/eval/rag_goldenset_v1.jsonl"
INDEX_PATH = PROJECT_ROOT / "data/knowledge/vector_store/monthly_cost_troubleshooting_chroma_v2"
COLLECTION_NAME = "monthly_cost_troubleshooting_v2"
MODEL_PATH = "/home/qiaojiahao/models/bge-small-zh-v1.5"
REPORT_PATH = PROJECT_ROOT / "docs/rag-eval-v1.md"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
PREFIX_REPORT_PATH = PROJECT_ROOT / "docs/rag-eval-v2-query-prefix.md"

N_RESULTS = 5
SEED = 42


def sha8(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_collection() -> tuple[object, SentenceTransformer]:
    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(MODEL_PATH)
    return collection, model


def build_where(rule_ids: list[str]) -> dict | None:
    """把触发的规则 ID 转成 $or + $contains；通配卡 "*" 永远放进来。"""
    clauses = [{"related_rule_ids": {"$contains": rid}} for rid in rule_ids]
    clauses.append({"related_rule_ids": {"$contains": "*"}})
    if not clauses:
        return None
    return {"$or": clauses}


def retrieve(collection, embeddings, mode: str, row: dict) -> list[str]:
    kwargs = {
        "query_embeddings": embeddings,
        "n_results": N_RESULTS,
        "include": ["metadatas", "distances"],
    }
    if mode == "filtered" and row["triggered_rule_ids"]:
        kwargs["where"] = build_where(row["triggered_rule_ids"])

    result = collection.query(**kwargs)
    metadatas = result["metadatas"][0]
    return [m["knowledge_id"] for m in metadatas if m]


def first_primary_rank(got: list[str], primary: list[str]) -> int | None:
    for index, kid in enumerate(got, start=1):
        if kid in primary:
            return index
    return None


def evaluate(
    rows: list[dict], collection, model, mode: str, use_prefix: bool = False
) -> tuple[list[dict], list[dict]]:
    scored: list[dict] = []
    detail: list[dict] = []

    queries = [row["query"] for row in rows]
    if use_prefix:
        queries = [QUERY_INSTRUCTION + q for q in queries]
    embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)

    for row, vector in zip(rows, embeddings):
        got = retrieve(collection, [vector.tolist()], mode, row)
        entry = {
            "qid": row["qid"],
            "seed_id": row["seed_id"],
            "group": row["group"],
            "variant_type": row["variant_type"],
            "business": row["business"],
            "query": row["query"],
            "primary": row["primary_knowledge_ids"],
            "secondary": row["secondary_knowledge_ids"],
            "trap": row["trap_knowledge_ids"],
            "expect_refusal": row["expect_refusal"],
            "got": got,
            "rank": None,
            "hit1": None,
            "hit3": None,
            "hit5": None,
            "secondary_hit3": False,
            "trap_hit3": False,
        }
        if not row["primary_knowledge_ids"]:
            entry["trap_hit3"] = any(t in got[:3] for t in row["trap_knowledge_ids"])
            detail.append(entry)
            continue

        rank = first_primary_rank(got, row["primary_knowledge_ids"])
        entry["rank"] = rank
        n_primary = len(row["primary_knowledge_ids"])
        for k in (1, 3, 5):
            if k == 1 and n_primary > 1:
                entry["hit1"] = None  # 单槽装不下两张主卡，R@1 无定义
            else:
                entry[f"hit{k}"] = all(p in got[:k] for p in row["primary_knowledge_ids"])
        entry["secondary_hit3"] = any(s in got[:3] for s in row["secondary_knowledge_ids"])
        entry["trap_hit3"] = any(t in got[:3] for t in row["trap_knowledge_ids"])
        scored.append(entry)
        detail.append(entry)

    return scored, detail


def summarize(scored: list[dict], detail: list[dict]) -> dict:
    out: dict = {}
    n = len(scored)
    out["n"] = n
    for k in (1, 3, 5):
        valid = [r for r in scored if r[f"hit{k}"] is not None]
        out[f"recall@{k}"] = (
            sum(1 for r in valid if r[f"hit{k}"]) / len(valid) if valid else 0.0
        )
        out[f"n@{k}"] = len(valid)

    recip = [1.0 / r["rank"] if r["rank"] else 0.0 for r in scored]
    out["mrr"] = float(np.mean(recip)) if recip else 0.0

    sec_rows = [r for r in scored if r["secondary"]]
    out["secondary_hit3"] = (
        sum(1 for r in sec_rows if r["secondary_hit3"]) / len(sec_rows) if sec_rows else 0.0
    )

    trap_rows = [r for r in detail if r["trap"]]
    out["trap_rate3"] = (
        sum(1 for r in trap_rows if r["trap_hit3"]) / len(trap_rows) if trap_rows else 0.0
    )
    out["n_trap"] = len(trap_rows)

    refusal_rows = [r for r in detail if r["expect_refusal"]]
    out["n_refusal"] = len(refusal_rows)

    by_group: dict[str, dict] = {}
    for group in sorted({r["group"] for r in scored}):
        subset = [r for r in scored if r["group"] == group]
        stat = {"n": len(subset)}
        for k in (1, 3, 5):
            valid = [r for r in subset if r[f"hit{k}"] is not None]
            stat[f"recall@{k}"] = (
                sum(1 for r in valid if r[f"hit{k}"]) / len(valid) if valid else None
            )
        by_group[group] = stat
    out["by_group"] = by_group

    per_seed: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        per_seed[row["seed_id"]].append(row)
    seed_r3 = [sum(1 for r in v if r["hit3"]) / len(v) for v in per_seed.values()]
    seed_r1: list[float] = []
    for v in per_seed.values():
        valid = [r for r in v if r["hit1"] is not None]
        if valid:
            seed_r1.append(sum(1 for r in valid if r["hit1"]) / len(valid))
    out["topic_recall@3"] = float(np.mean(seed_r3)) if seed_r3 else 0.0
    out["topic_recall@1"] = float(np.mean(seed_r1)) if seed_r1 else 0.0
    out["n_seed"] = len(per_seed)

    per_card: dict[str, dict] = defaultdict(lambda: {"n": 0, "hit3": 0})
    for row in scored:
        for card in row["primary"]:
            per_card[card]["n"] += 1
            if row["hit3"]:
                per_card[card]["hit3"] += 1
    out["per_card"] = {
        card: {"n": v["n"], "hit3": v["hit3"] / v["n"]} for card, v in sorted(per_card.items())
    }
    
    by_variant: dict[str, dict] = {}
    for row in scored:
        stat = by_variant.setdefault(row["variant_type"], {"n": 0, "fail": 0})
        stat["n"] += 1
        if not row["hit3"]:
            stat["fail"] += 1
    out["by_variant"] = dict(sorted(by_variant.items()))
    out["fail_by_group"] = {
        g: sum(1 for r in scored if r["group"] == g and not r["hit3"])
        for g in out["by_group"]
    }
    
    out["failures"] = [r for r in scored if not r["hit3"]]
    return out


def pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def render_report(config: dict, raw: dict, filt: dict) -> str:
    lines: list[str] = []
    lines.append(f"# RAG 检索实验：{config.get('实验编号', 'v1')}")
    lines.append("")
    lines.append("## 实验配置")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for key, value in config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## 总体结果")
    lines.append("")
    lines.append(
        "| 路径 | 样本 | Recall@1 | R@1 口径样本 | Recall@3 | Recall@5 | MRR | Topic级 R@3 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, res in (("raw（纯向量）", raw), ("filtered（规则过滤）", filt)):
        lines.append(
            f"| {name} | {res['n']} | {pct(res['recall@1'])} | {res['n@1']} | "
            f"{pct(res['recall@3'])} | {pct(res['recall@5'])} | {res['mrr']:.3f} | "
            f"{pct(res['topic_recall@3'])} |"
        )
    lines.append("")
    lines.append("## 口径说明")
    lines.append("")
    lines.append("- Recall@K = 该条样本的**全部** primary 卡都出现在 Top-K 才算命中。")
    lines.append("- 多主卡样本（需要 2 张卡）在单槽 R@1 下无定义，已从 R@1 分母剔除。")
    lines.append("- 拒答样本 primary 为空，不参与 Recall，单独计数。")
    lines.append(
        "- **filtered 路径下，干扰卡（不挂规则）与大部分辅卡（挂其他规则）被硬过滤排除，"
        "故其干扰率为 0，不能与 raw 直接对比检索能力。**"
    )
    lines.append("")
    lines.append("## 分组结果")
    lines.append("")
    lines.append("| 组 | 路径 | 样本 | R@1 | R@3 | R@5 |")
    lines.append("|---|---|---|---|---|---|")
    for name, res in (("raw", raw), ("filtered", filt)):
        for group, stat in res["by_group"].items():
            lines.append(
                f"| {group} | {name} | {stat['n']} | {pct(stat['recall@1'])} | "
                f"{pct(stat['recall@3'])} | {pct(stat['recall@5'])} |"
            )
    lines.append("")
    lines.append("## 干扰项与辅卡")
    lines.append("")
    lines.append("| 指标 | raw | filtered |")
    lines.append("|---|---|---|")
    lines.append(f"| 干扰卡进入 Top-3 的比例 | {pct(raw['trap_rate3'])} | {pct(filt['trap_rate3'])} |")
    lines.append(f"| 干扰样本数 | {raw['n_trap']} | {filt['n_trap']} |")
    lines.append(
        f"| 辅卡进入 Top-3 的比例 | {pct(raw['secondary_hit3'])} | {pct(filt['secondary_hit3'])} |"
    )
    lines.append(f"| 拒答样本数（不参与 Recall） | {raw['n_refusal']} | {filt['n_refusal']} |")
    lines.append("")
    lines.append("## 按主卡 Recall@3（raw）")
    lines.append("")
    lines.append("| 卡 | 出现次数 | R@3 |")
    lines.append("|---|---|---|")
    for card, stat in raw["per_card"].items():
        lines.append(f"| {card} | {stat['n']} | {pct(stat['hit3'])} |")
    lines.append("")
    lines.append("## 失败分布（raw）")
    lines.append("")
    lines.append("| 维度 | 值 | 样本 | 失败 | 失败率 |")
    lines.append("|---|---|---|---|---|")
    for variant, stat in raw["by_variant"].items():
        rate = stat["fail"] / stat["n"] if stat["n"] else 0
        lines.append(f"| 变体 | {variant} | {stat['n']} | {stat['fail']} | {pct(rate)} |")
    for group, fails in raw["fail_by_group"].items():
        total = raw["by_group"][group]["n"]
        rate = fails / total if total else 0
        lines.append(f"| 分组 | {group} | {total} | {fails} | {pct(rate)} |")
    lines.append("")
    lines.append("## 失败样例（raw，最多 30 条）")
    lines.append("")
    for row in raw["failures"][:30]:
        lines.append(f"- `{row['qid']}` {row['group']}/{row['variant_type']}：{row['query']}")
        lines.append(f"  - 期望 {row['primary']}，实际 Top-3 {row['got'][:3]}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="写入报告文件")
    parser.add_argument(
        "--query-prefix",
        action="store_true",
        help="查询侧加 BGE 检索指令前缀（文档侧不加）",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    rows = load_jsonl(GOLDENSET_PATH)
    collection, model = load_collection()

    use_prefix = bool(args.query_prefix)
    raw_scored, raw_detail = evaluate(rows, collection, model, "raw", use_prefix)
    filt_scored, filt_detail = evaluate(rows, collection, model, "filtered", use_prefix)

    raw_sum = summarize(raw_scored, raw_detail)
    filt_sum = summarize(filt_scored, filt_detail)

    config = {
        "知识库版本": f"{KNOWLEDGE_PATH.name} @ {sha8(KNOWLEDGE_PATH)}",
        "GT 版本": f"{GOLDENSET_PATH.name} @ {sha8(GOLDENSET_PATH)}",
        "GT 条数": len(rows),
        "母题数": raw_sum["n_seed"],
        "向量模型": MODEL_PATH,
        "索引 collection": COLLECTION_NAME,
        "索引目录": str(INDEX_PATH.relative_to(PROJECT_ROOT)),
        "距离度量": "cosine",
        "Top-K": N_RESULTS,
        "随机种子": SEED,
        "实验编号": "E2 查询指令前缀" if use_prefix else "E1 基线",
        "查询指令前缀": (
            f"使用（{QUERY_INSTRUCTION}）" if use_prefix else "未使用（基线）"
        ),
        "运行时间": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    report = render_report(config, raw_sum, filt_sum)
    print(report)

    if args.save:
        target = PREFIX_REPORT_PATH if use_prefix else REPORT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"\n已写入：{target}")

if __name__ == "__main__":
    main()
