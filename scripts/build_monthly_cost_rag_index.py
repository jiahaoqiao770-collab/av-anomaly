#!/usr/bin/env python3
"""Build a Chroma index for monthly cost troubleshooting knowledge."""

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
    / "data/knowledge/vector_store/monthly_cost_troubleshooting_chroma"
)
COLLECTION_NAME = "monthly_cost_troubleshooting"


def load_knowledge() -> list[dict]:
    rows = []

    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"知识文件第 {line_number} 行不是合法JSON"
                ) from error

            rows.append(row)

    if not rows:
        raise ValueError("知识文件为空")

    knowledge_ids = [row["knowledge_id"] for row in rows]
    if len(knowledge_ids) != len(set(knowledge_ids)):
        raise ValueError("knowledge_id 存在重复")

    return rows


def build_document(row: dict) -> str:
    parts = [
        f"标题：{row['title']}",
        f"适用业务：{'、'.join(row['applicable_business'])}",
        f"相关规则：{'、'.join(row['related_rule_ids'])}",
        f"计算关系：{row['formula']}",
        f"排查步骤：{'；'.join(row['check_steps'])}",
        f"判断规则：{'；'.join(row['decision_rules'])}",
        f"可能业务原因：{'；'.join(row['possible_business_causes'])}",
    ]
    return "\n".join(parts)


def main() -> None:
    knowledge = load_knowledge()
    documents = [build_document(row) for row in knowledge]
    knowledge_ids = [row["knowledge_id"] for row in knowledge]

    print(f"读取知识：{len(knowledge)} 条")
    print(f"加载向量模型：{MODEL_PATH}")

    model = SentenceTransformer(MODEL_PATH)

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    metadata = [
        {
            "knowledge_id": row["knowledge_id"],
            "title": row["title"],
            "knowledge_type": row["knowledge_type"],
            "applicable_business": "、".join(row["applicable_business"]),
            "related_rule_ids": "、".join(row["related_rule_ids"]),
        }
        for row in knowledge
    ]

    collection.upsert(
        ids=knowledge_ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata,
    )

    print(f"索引构建完成：{collection.count()} 条")
    print(f"索引位置：{INDEX_PATH}")


if __name__ == "__main__":
    main()