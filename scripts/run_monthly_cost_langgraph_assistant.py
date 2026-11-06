#!/usr/bin/env python3
"""Natural-language assistant for monthly Clip cost analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

import chromadb
import pandas as pd
import torch
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.monthly_cost_anomaly_detector import (  # noqa: E402
    get_anomalies,
    load_metrics,
)
from src.diagnosis.monthly_cost_diagnoser import get_diagnoses  # noqa: E402
from src.metrics.clip_cost_metrics import (  # noqa: E402
    compare_metric,
    get_metric,
)


QWEN_PATH = "/home/qiaojiahao/models/Qwen2.5-3B-Instruct"
BGE_PATH = "/home/qiaojiahao/models/bge-small-zh-v1.5"
INDEX_PATH = (
    PROJECT_ROOT
    / "data/knowledge/vector_store/monthly_cost_troubleshooting_chroma"
)
COLLECTION_NAME = "monthly_cost_troubleshooting"

METRICS = {
    "monthly_total_cost": "月度总成本",
    "total_work_hours": "总工时",
    "driving_work_hours": "行车工时",
    "parking_work_hours": "泊车工时",
    "driving_allocated_cost": "行车分摊成本",
    "parking_allocated_cost": "泊车分摊成本",
    "driving_clip_count": "行车Clip数",
    "parking_clip_count": "泊车Clip数",
    "driving_clips_per_hour": "行车每小时Clip数",
    "parking_clips_per_hour": "泊车每小时Clip数",
    "driving_unit_clip_cost": "行车单Clip成本",
    "parking_unit_clip_cost": "泊车单Clip成本",
}

UNITS = {
    "monthly_total_cost": "万元",
    "total_work_hours": "小时",
    "driving_work_hours": "小时",
    "parking_work_hours": "小时",
    "driving_allocated_cost": "万元",
    "parking_allocated_cost": "万元",
    "driving_clip_count": "Clip",
    "parking_clip_count": "Clip",
    "driving_clips_per_hour": "Clip/小时",
    "parking_clips_per_hour": "Clip/小时",
    "driving_unit_clip_cost": "元/Clip",
    "parking_unit_clip_cost": "元/Clip",
}


class AssistantState(TypedDict, total=False):
    question: str
    plan: dict
    query_result: object
    rag_knowledge: list[str]
    answer: str
    error: str


def load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH,
        local_files_only=True,
    )

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
    return tokenizer, model


def call_qwen(tokenizer, model, messages, max_new_tokens):
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


def parse_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"意图识别没有返回JSON：{text}")
    return json.loads(text[start:end + 1])


def previous_month(month: str) -> str:
    return str(pd.Period(month, freq="M") - 1)


def build_graph(tokenizer, model):
    metrics = load_metrics()
    available_months = metrics["month"].astype(str).tolist()

    embedding_model = SentenceTransformer(
    BGE_PATH,
    device="cpu",
    )

    def plan_node(state: AssistantState) -> AssistantState:
        prompt = f"""
你是数据查询计划器，只输出JSON。

可用月份：
{available_months}

可用指标：
{json.dumps(METRICS, ensure_ascii=False)}

intent只能选择：
- general_chat：问候、自我介绍、询问助手能力
- get_metric：查询某月指标
- compare_metric：比较月份或环比
- get_anomalies：查询预警
- explain_diagnosis：解释异常原因
- troubleshooting：查询排查方法
- generate_report：生成诊断报告
- unsupported：与本项目无关或无法回答的问题

输出格式：
{{
  "intent": "上述类型之一",
  "period": "实际YYYY-MM或null",
  "baseline_period": "实际YYYY-MM或null",
  "business_type": "行车或泊车或null",
  "metric_ids": ["合法英文指标ID"]
}}

规则：
1. “11月”应从可用月份中选择对应的完整月份。
2. “上月、前一个月”不明确写baseline_period时可以为null。
3. 用户没指定行车或泊车时，相关指标应同时选择行车和泊车。
4. 不得输出YYYY-MM等占位符。5. “你好”“你是谁”“你能做什么”必须使用general_chat。
6. 与采集成本效率无关的问题使用unsupported。
7. general_chat和unsupported的period、baseline_period、
   business_type必须为null，metric_ids必须为空数组。
8. 缺少月份时不得自行猜测，period必须为null。

示例：
问题：11月Clip成本相对上月怎么样？
输出：
{{"intent":"compare_metric","period":"2026-11",
"baseline_period":"2026-10","business_type":null,
"metric_ids":["driving_unit_clip_cost","parking_unit_clip_cost"]}}
"""

        result = call_qwen(
            tokenizer,
            model,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": state["question"]},
            ],
            220,
        )

        try:
            plan = parse_json(result)
            return {"plan": plan}
        except Exception as error:
            return {"error": str(error)}

    def query_node(state: AssistantState) -> AssistantState:
        if state.get("error"):
            return {}

        plan = state["plan"]
        intent = plan.get("intent")
        period = plan.get("period")
        business_type = plan.get("business_type")
        metric_ids = plan.get("metric_ids") or []

        try:
            if intent in {"general_chat", "unsupported"}:
                return {"query_result": []}
            if intent == "get_metric":
                if not period:
                    raise ValueError("请补充需要查询的月份")

                result = [
                    {
                        "metric_id": metric_id,
                        "metric_name": METRICS[metric_id],
                        "month": period,
                        "value": get_metric(metric_id, period, metrics),
                        "unit": UNITS[metric_id],
                    }
                    for metric_id in metric_ids
                ]

            elif intent == "compare_metric":
                if not period:
                    raise ValueError("请补充当前月份")

                baseline = (
                    plan.get("baseline_period")
                    or previous_month(period)
                )

                result = []
                for metric_id in metric_ids:
                    comparison = compare_metric(
                        metric_id,
                        period,
                        baseline,
                        metrics,
                    )
                    comparison["metric_name"] = METRICS[metric_id]
                    comparison["unit"] = UNITS[metric_id]
                    result.append(comparison)

            elif intent == "get_anomalies":
                result = get_anomalies(
                    month=period,
                    business_type=business_type,
                )

            elif intent in {
                "explain_diagnosis",
                "generate_report",
            }:
                result = get_diagnoses(
                    month=period,
                    business_type=business_type,
                )

            elif intent == "troubleshooting":
                result = []

            else:
                raise ValueError(f"不支持的查询类型：{intent}")

            return {"query_result": result}

        except Exception as error:
            return {"error": str(error)}

    def rag_node(state: AssistantState) -> AssistantState:
        if state.get("error"):
            return {}

        intent = state["plan"].get("intent")
        if intent not in {
            "explain_diagnosis",
            "troubleshooting",
            "generate_report",
        }:
            return {"rag_knowledge": []}

        
        embedding = embedding_model.encode(
            [state["question"]],
            normalize_embeddings=True,
        ).tolist()

        client = chromadb.PersistentClient(
            path=str(INDEX_PATH)
        )
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
        results = collection.query(
            query_embeddings=embedding,
            n_results=2,
        )

        return {
            "rag_knowledge": results["documents"][0]
        }

    def answer_node(state: AssistantState) -> AssistantState:
        if state.get("error"):
            return {
                "answer": f"无法完成查询：{state['error']}"
            }
        intent = state["plan"].get("intent")

        if intent == "general_chat":
            return {
                "answer": (
                    "你好，我可以查询月度成本效率指标、环比变化、"
                    "异常预警、诊断原因和排查路径。"
                )
            }

        if intent == "unsupported":
            return {
                "answer": (
                    "这个问题不属于当前采集成本效率分析范围。"
                    "你可以询问指标、预警、异常原因或排查方法。"
                )
            }
        context = {
            "query_plan": state.get("plan"),
            "query_result": state.get("query_result"),
            "rag_knowledge": state.get("rag_knowledge", []),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "你是采集成本效率分析助手。"
                    "查询结果中的数据是事实。"
                    "RAG知识只是排查参考。"
                    "不得编造数值或已确认根因。"
                    "使用简洁中文直接回答。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{state['question']}\n\n"
                    "查询上下文：\n"
                    + json.dumps(
                        context,
                        ensure_ascii=False,
                        default=str,
                    )
                ),
            },
        ]

        answer = call_qwen(
            tokenizer,
            model,
            messages,
            600,
        )
        return {"answer": answer}

    graph = StateGraph(AssistantState)
    graph.add_node("plan", plan_node)
    graph.add_node("query", query_node)
    graph.add_node("rag", rag_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "query")
    graph.add_edge("query", "rag")
    graph.add_edge("rag", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def main():
    print("正在加载Qwen模型……")
    tokenizer, model = load_qwen()
    assistant = build_graph(tokenizer, model)

    print("模型加载完成。输入 exit 退出。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        if not question:
            continue

        normalized_question = "".join(question.split()).rstrip("？?!")

        if normalized_question == "你知不知道赵伊婷":
            print("知道！她就是奶婷，是著名顶顶的奶龙之女")
            continue

        result = assistant.invoke({"question": question})

        print("\n===== 回答 =====")
        print(result["answer"])

if __name__ == "__main__":
    main()