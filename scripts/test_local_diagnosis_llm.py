import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/processed/monthly_cost_diagnosis_results.jsonl"
MODEL_PATH = "/home/qiaojiahao/models/Qwen2.5-3B-Instruct"


# 1. 读取诊断结果
with DATA_PATH.open("r", encoding="utf-8") as file:
    diagnoses = [json.loads(line) for line in file if line.strip()]

diagnosis = next(
    item
    for item in diagnoses
    if item["period"] == "2026-07" and item["business_type"] == "泊车"
)


# 2. 加载模型
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    local_files_only=True,
)

# 3. 组织提示词
messages = [
    {
        "role": "system",
        "content": (
            "你是泊车采集成本效率诊断助手。"
            "只能根据提供的数据生成报告，不得编造原因。"
        ),
    },
    {
        "role": "user",
        "content": (
            "请根据下面的结构化诊断结果，简洁说明主要异常、"
            "主要关联因素和辅助证据：\n"
            + json.dumps(diagnosis, ensure_ascii=False)
        ),
    },
]


# 4. 生成报告
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
    )

input_length = inputs["input_ids"].shape[-1]

answer = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True,
)

print(answer)