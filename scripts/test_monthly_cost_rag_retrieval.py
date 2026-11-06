#!/usr/bin/env python3

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = "/home/qiaojiahao/models/bge-small-zh-v1.5"
INDEX_PATH = (
    PROJECT_ROOT
    / "data/knowledge/vector_store/monthly_cost_troubleshooting_chroma"
)
COLLECTION_NAME = "monthly_cost_troubleshooting"


query = input("请输入排查问题：").strip()

model = SentenceTransformer(MODEL_PATH)
query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
).tolist()

client = chromadb.PersistentClient(path=str(INDEX_PATH))
collection = client.get_collection(name=COLLECTION_NAME)

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
)

print(f"\n查询：{query}")
print("\n相关排查知识：")

for index, document in enumerate(results["documents"][0], start=1):
    metadata = results["metadatas"][0][index - 1]
    distance = results["distances"][0][index - 1]

    print(f"\n结果 {index}")
    print(f"知识ID：{metadata['knowledge_id']}")
    print(f"标题：{metadata['title']}")
    print(f"距离：{distance:.4f}")
    print(document)