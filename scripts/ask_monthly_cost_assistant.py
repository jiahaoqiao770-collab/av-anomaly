#!/usr/bin/env python3
"""Answer monthly cost diagnosis questions with local Qwen and RAG."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "data/processed/monthly_cost_diagnosis_results.jsonl"
)
INDEX_PATH = (
    PROJECT_ROOT
    / "data/knowledge/vector_store/monthly_cost_troubleshooting_chroma"
)

QWEN_PATH = "/home/qiaojiahao/models/Qwen2.5-3B-Instruct"
BGE_PATH = "/home/qiaojiahao/models/bge-small-zh-v1.5"
COLLECTION_NAME = "monthly_cost_troubleshooting"


def load_diagnoses() -> list[dict]:
    with DIAGNOSIS_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH,
        local_files_only=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True,
    )

    return tokenizer, model


def call_qwen(
    tokenizer,
    model,
    messages: list[dict],
    max_new_tokens: int,
) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]

    return tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    ).strip()


def parse_json_answer(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"模型没有返回合法JSON：{text}")

    return json.loads(text[start:end + 1])


def understand_question(
    question: str,
    diagnoses: list[dict],
    tokenizer,
    model,
) -> dict:
    available = sorted({
        (item["period"], item["business_type"])
        for item in diagnoses
    })

    messages = [
        {
            "role": "system",
            "content": (
                "你是查询意图识别器。"
                "只输出JSON，不要输出解释。"
                "不得编造不存在的月份或业务类型。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n"
                f"可查询数据：{available}\n\n"
                "请输出：\n"
                "{"
                "\"intent\":\"explain_anomaly\","
                "\"period\":\"YYYY-MM或null\","
                "\"business_type\":\"行车、泊车或null\""
                "}"
            ),
        },
    ]

    answer = call_qwen(
        tokenizer,
        model,
        messages,
        max_new_tokens=120,
    )

    return parse_json_answer(answer)


def find_diagnosis(
    diagnoses: list[dict],
    period: str,
    business_type: str,
) -> dict:
    for diagnosis in diagnoses:
        if (
            diagnosis["period"] == period
            and diagnosis["business_type"] == business_type
        ):
            return diagnosis

    raise ValueError(
        f"没有找到 {period} 的{business_type}诊断结果"
    )


def retrieve_knowledge(
    question: str,
    diagnosis: dict,
) -> list[str]:
    query_parts = [
        question,
        diagnosis["primary_anomaly"]["rule_name"],
    ]

    if diagnosis.get("primary_reason"):
        query_parts.append(
            diagnosis["primary_reason"]["rule_name"]
        )

    query = "；".join(query_parts)

    # BGE使用CPU，避免占用Qwen所需的显存。
    embedding_model = SentenceTransformer(
        BGE_PATH,
        device="cpu",
    )

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    client = chromadb.PersistentClient(
        path=str(INDEX_PATH)
    )
    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2,
    )

    return results["documents"][0]


def compact_diagnosis(diagnosis: dict) -> dict:
    fields = [
        "period",
        "business_type",
        "severity",
        "primary_anomaly",
        "primary_reason",
        "contributing_reasons",
        "supporting_evidence",
        "context",
        "metric_snapshot",
    ]

    return {
        field: diagnosis.get(field)
        for field in fields
    }


def answer_question(
    question: str,
    diagnosis: dict,
    knowledge: list[str],
    tokenizer,
    model,
) -> str:
    diagnosis_text = json.dumps(
        compact_diagnosis(diagnosis),
        ensure_ascii=False,
        indent=2,
    )

    knowledge_text = "\n\n".join(knowledge)

    messages = [
        {
            "role": "system",
            "content": (
                "你是采集成本效率诊断助手。"
                "诊断数据是事实，RAG知识是排查参考。"
                "不得编造数值。"
                "不得把可能原因说成已确认根因。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"诊断数据：\n{diagnosis_text}\n\n"
                f"RAG排查知识：\n{knowledge_text}\n\n"
                "请直接回答用户问题，并说明主要异常、"
                "关联因素和建议的排查顺序。"
            ),
        },
    ]

    return call_qwen(
        tokenizer,
        model,
        messages,
        max_new_tokens=600,
    )


def main() -> None:
    question = input("请输入问题：").strip()

    diagnoses = load_diagnoses()

    print("正在加载Qwen模型……")
    tokenizer, model = load_qwen()

    intent = understand_question(
        question,
        diagnoses,
        tokenizer,
        model,
    )

    period = intent.get("period")
    business_type = intent.get("business_type")

    if not period or not business_type:
        print("无法确定月份或业务类型，请补充后重新提问。")
        return

    diagnosis = find_diagnosis(
        diagnoses,
        period,
        business_type,
    )

    knowledge = retrieve_knowledge(
        question,
        diagnosis,
    )

    answer = answer_question(
        question,
        diagnosis,
        knowledge,
        tokenizer,
        model,
    )

    print("\n===== 回答 =====")
    print(answer)


if __name__ == "__main__":
    main()