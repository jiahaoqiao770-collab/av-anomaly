#!/usr/bin/env python3
"""月度采集成本助手：QLoRA smoke test。默认只预检。"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    "/home/qiaojiahao/models/Qwen2.5-3B-Instruct"
)
DATA_PATH = (
    PROJECT_ROOT
    / "data/sft/monthly_cost_assistant_train.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/sft/output/monthly_cost_qwen3b_lora_smoke"
)

MAX_LENGTH = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="默认只预检；加 --run 才启动训练。"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="显式启动训练。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="LoRA 输出目录，必须不存在或为空。",
    )
    return parser.parse_args()


def validate_output_dir(path):
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(
                f"输出位置已有内容，拒绝覆盖：{path}\n"
                "请使用 --output-dir 指定一个新的目录。"
            )


def load_records():
    records = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"第 {line_number} 行 JSON 格式错误。"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"第 {line_number} 行必须是 JSON 对象。"
                )

            messages = record.get("messages")

            if (
                not isinstance(messages, list)
                or len(messages) != 3
                or not all(
                    isinstance(message, dict)
                    for message in messages
                )
            ):
                raise ValueError(
                    f"第 {line_number} 行必须包含三条消息。"
                )

            roles = [
                message.get("role")
                for message in messages
            ]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(
                    f"第 {line_number} 行角色顺序错误：{roles}"
                )

            if not all(
                isinstance(message.get("content"), str)
                and message["content"].strip()
                for message in messages
            ):
                raise ValueError(
                    f"第 {line_number} 行包含空内容。"
                )

            records.append(record)

    # 此脚本专门用于当前的 10 条数据试跑。
    if len(records) != 10:
        raise ValueError(
            f"当前读取到 {len(records)} 条记录；"
            "本次 smoke test 要求恰好 10 条。"
        )

    return records


def build_dataset(records, tokenizer):
    rows = []
    lengths = []

    for index, record in enumerate(records, start=1):
        messages = record["messages"]

        # prompt 包含 system、user 和 assistant 起始标记。
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        # full_text 包含完整的 assistant 回答及结束标记。
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not full_text.startswith(prompt):
            raise ValueError(
                f"第 {index} 条聊天模板无法拆分为提示和回答。"
            )

        completion = full_text[len(prompt):]

        length = len(
            tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]
        )
        lengths.append(length)

        # 拒绝静默截断，避免回答部分被截掉。
        if length > MAX_LENGTH:
            raise ValueError(
                f"第 {index} 条有 {length} 个 token，"
                f"超过上限 {MAX_LENGTH}。请先检查样本。"
            )

        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
            }
        )

    print(
        f"样本 token 长度："
        f"最短 {min(lengths)}，最长 {max(lengths)}"
    )

    return Dataset.from_list(rows)


def main():
    args = parse_args()

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在：{MODEL_PATH}"
        )

    validate_output_dir(args.output_dir)
    records = load_records()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(records, tokenizer)

    print(f"基座模型：{MODEL_PATH}")
    print(f"训练数据：{DATA_PATH}")
    print(f"有效样本：{len(dataset)}")
    print(f"输出目录：{args.output_dir}")

    if not args.run:
        print("\n预检通过，没有启动训练。")
        print("训练前请关闭本地 Qwen/LangGraph 进程。")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止训练。")

    print(f"\nGPU：{torch.cuda.get_device_name(0)}")
    print("开始训练：1 epoch，batch size 1。")

    set_seed(42)
    torch.cuda.reset_peak_memory_stats(0)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),

        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,

        max_length=MAX_LENGTH,
        packing=False,
        completion_only_loss=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        use_cache=False,

        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",

        logging_steps=1,
        logging_first_step=True,

        # 结束后手动保存适配器，不保存中间检查点。
        save_strategy="no",
        eval_strategy="no",
        report_to="none",

        dataloader_num_workers=0,
        seed=42,
        data_seed=42,

        model_init_kwargs={
            "local_files_only": True,
            # 固定放到第一张 GPU，避免自动分配到 CPU。
            "device_map": {"": 0},
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )

    trainer = SFTTrainer(
        model=str(MODEL_PATH),
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        quantization_config=quantization_config,
        peft_config=lora_config,
    )

    trainer.model.print_trainable_parameters()

    result = trainer.train()

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    peak_allocated = (
        torch.cuda.max_memory_allocated(0) / 1024**3
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(0) / 1024**3
    )

    print("\n训练及保存完成。")
    print(f"训练指标：{result.metrics}")
    print(f"PyTorch 峰值已分配显存：{peak_allocated:.2f} GiB")
    print(f"PyTorch 峰值预留显存：{peak_reserved:.2f} GiB")
    print(f"LoRA 保存位置：{args.output_dir}")

    expected_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    for filename in expected_files:
        if not (args.output_dir / filename).is_file():
            raise RuntimeError(
                f"保存结果缺少预期文件：{filename}"
            )

    print("LoRA 核心文件检查通过。")


if __name__ == "__main__":
    main()