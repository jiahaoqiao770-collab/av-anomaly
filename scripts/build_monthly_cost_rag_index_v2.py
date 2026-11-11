#!/usr/bin/env python3
"""Build a v2 Chroma index for monthly cost troubleshooting knowledge.

与 v1 的差异（仅此三处）：
1. metadata 中 related_rule_ids / applicable_business 存「数组」，不再 join 成字符串，
   否则无法用 $in 做按规则过滤；
2. 输出到独立的索引目录与 collection 名，不覆盖 v1；
3. 结尾打印维度与过滤自检，供人工核对。

运行：python -m scripts.build_monthly_cost_rag_index_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_PATH = (
    PROJECT_ROOT
    / "data/knowledge/raw/monthly_cost_troubleshooting_knowledge.jsonl"
)
MODEL_PATH = "/home/qiaojiahao/models/bge-small-zh-v1.5"
INDEX_PATH = (
    PROJECT_ROOT
    / "data/knowledge/vector_store/monthly_cost_troubleshooting_chroma_v2"
)
COLLECTION_NAME = "monthly_cost_troubleshooting_v2"

ALIAS_PATH = PROJECT_ROOT / "config/rag_aliases.json"


def load_aliases() -> dict[str, list[str]]:
    if not ALIAS_PATH.exists():
        return {}
    with ALIAS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_knowledge() -> list[dict]:
    rows = []
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"知识文件第 {line_number} 行不是合法JSON") from error

    if not rows:
        raise ValueError("知识文件为空")

    ids = [row["knowledge_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("knowledge_id 存在重复")

    for row in rows:
        if "related_rule_ids" not in row:
            raise ValueError(f"{row['knowledge_id']} 缺少 related_rule_ids 字段")

    return rows


def build_document(row: dict, aliases: dict[str, list[str]] | None = None) -> str:
    parts = [
        f"标题：{row['title']}",
        f"适用业务：{'、'.join(row['applicable_business'])}",
        f"相关规则：{'、'.join(row['related_rule_ids']) or '无'}",
        f"计算关系：{row['formula']}",
        f"排查步骤：{'；'.join(row['check_steps'])}",
        f"判断规则：{'；'.join(row['decision_rules'])}",
        f"可能业务原因：{'；'.join(row['possible_business_causes'])}",
    ]
    terms = (aliases or {}).get(row["knowledge_id"], [])
    if terms:
        parts.append(f"同义说法：{'、'.join(terms)}")
    return "\n".join(parts)


def build_metadata(row: dict) -> dict:
    """metadata 只放「可过滤的标量或数组」，不放正文。

    related_rule_ids 为空（干扰项）时不写该键：空数组既无意义，
    也可能被不同版本的 Chroma 拒绝；缺键自然不会被 $in 命中。
    """
    meta = {
        "knowledge_id": row["knowledge_id"],
        "title": row["title"],
        "knowledge_type": row["knowledge_type"],
        "applicable_business": list(row["applicable_business"]),
    }
    rule_ids = [x for x in row["related_rule_ids"] if x]
    if rule_ids:
        meta["related_rule_ids"] = rule_ids
    return meta


def main() -> None:
    knowledge = load_knowledge()
    aliases = load_aliases()
    print(f"别名表：{len(aliases)} 张卡")
    documents = [build_document(row, aliases) for row in knowledge]
    knowledge_ids = [row["knowledge_id"] for row in knowledge]
    metadatas = [build_metadata(row) for row in knowledge]

    print(f"读取知识：{len(knowledge)} 条")

    model = SentenceTransformer(MODEL_PATH)
    print(f"向量模型：{MODEL_PATH} dim={model.get_sentence_embedding_dimension()}")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    dim = len(embeddings[0])
    if any(len(vec) != dim for vec in embeddings):
        raise ValueError("向量维度不一致")

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_PATH))

    # 重建前先删同名 collection，避免旧向量残留
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"已删除旧 collection：{COLLECTION_NAME}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=knowledge_ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"\n入库条数：{collection.count()}")
    print(f"collection：{COLLECTION_NAME}")
    print(f"距离度量：{collection.metadata}")
    print(f"索引目录：{INDEX_PATH}")

    # ---- 自检 1：某条卡能否按规则 ID 过滤到 ----
    got = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=3,
        where={"related_rule_ids": {"$contains": "A03"}},
        include=["metadatas", "distances"],
    )
    hit_ids = [m["knowledge_id"] for m in got["metadatas"][0]]
    print(f"\n[自检] where A03 $in -> {hit_ids}")
    print("       期望只含 K001（A01-A04 同卡）")

    # ---- 自检 2：B02 单独过滤 ----
    got = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=3,
        where={"related_rule_ids": {"$contains": "B02"}},
        include=["metadatas"],
    )
    print(f"[自检] where B02 $in -> {[m['knowledge_id'] for m in got['metadatas'][0]]}")
    print("       期望只含 K009")


if __name__ == "__main__":
    main()
