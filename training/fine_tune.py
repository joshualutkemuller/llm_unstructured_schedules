"""
QLoRA fine-tuning script using HuggingFace TRL / PEFT.

Strategy:
  - Base model: Llama-3.1-8B-Instruct (or Mistral-7B-Instruct-v0.3)
  - Quantization: 4-bit NF4 (bitsandbytes) to fit on a single A100 40GB
  - Adapter: LoRA r=16, alpha=32 on q_proj, v_proj, k_proj, o_proj
  - Training: SFTTrainer (causal LM on the 'text' column)
  - Epochs: 3, lr=2e-4, cosine schedule with warmup

Usage:
    python -m training.fine_tune \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --dataset data/training \
        --output models/collateral-v1 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def build_bnb_config():
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("pip install bitsandbytes transformers accelerate")

    import torch
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config():
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise ImportError("pip install peft")

    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        bias="none",
    )


def build_training_args(output_dir: str, epochs: int, batch_size: int):
    try:
        from transformers import TrainingArguments
    except ImportError:
        raise ImportError("pip install transformers")

    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=4,
        group_by_length=True,
    )


def train(
    base_model: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    max_seq_length: int = 4096,
):
    try:
        from datasets import load_from_disk
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer
    except ImportError:
        raise ImportError("pip install trl peft transformers datasets bitsandbytes accelerate")

    from datasets import load_from_disk
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    logger.info("Loading tokenizer from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model in 4-bit from %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    logger.info("Loading dataset from %s", dataset_path)
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    training_args = build_training_args(output_dir, epochs, batch_size)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,
    )

    logger.info("Starting training…")
    trainer.train()

    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done.")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="data/training")
    parser.add_argument("--output", default="models/collateral-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
