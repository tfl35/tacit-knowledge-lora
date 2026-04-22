#!/usr/bin/env python3
"""
Cross-Scale Qwen3.5 bf16 LoRA Fine-Tuning
==========================================
Unified training script for all Qwen3.5 model sizes on DGX Spark.

Usage:
    python train.py --dataset dataset/data.json --model Qwen/Qwen3.5-9B
    python train.py --dataset dataset/data.json --model Qwen/Qwen3.5-4B
    python train.py --dataset dataset/data.json --model Qwen/Qwen3.5-2B
    python train.py --dataset dataset/data.json --model Qwen/Qwen3.5-0.8B
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# ── Model-specific defaults ──────────────────────────────────
# Proven on 9B; scaled down conservatively for smaller models.
# Key insight from lit review: identical architecture family means
# tokenizer, chat template, and attention mechanism are shared.
# Only parameter capacity varies.
MODEL_CONFIGS = {
    "Qwen/Qwen3.5-9B": {
        "lora_r": 64,
        "lora_alpha": 128,
        "batch_size": 2,
        "grad_accum": 4,
        "max_seq_len": 4096,
        "lr": 2e-5,
    },
    "Qwen/Qwen3.5-4B": {
        "lora_r": 64,
        "lora_alpha": 128,
        "batch_size": 2,
        "grad_accum": 4,
        "max_seq_len": 4096,
        "lr": 2e-5,
    },
    "Qwen/Qwen3.5-2B": {
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 4,
        "grad_accum": 2,
        "max_seq_len": 4096,
        "lr": 2e-5,
    },
    "Qwen/Qwen3.5-0.8B": {
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 4,
        "grad_accum": 2,
        "max_seq_len": 4096,
        "lr": 2e-5,
    },
}

DTYPE = torch.bfloat16

# Short name for output directories
def model_tag(model_name):
    """Qwen/Qwen3.5-9B -> qwen3.5-9b"""
    return model_name.split("/")[-1].lower()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                   help="Qwen3.5 model to fine-tune")
    p.add_argument("--output", default="output")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=None, help="Override default LR")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval_split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def check_kernels():
    """Qwen3.5's linear attention layers need compiled CUDA kernels."""
    try:
        import fla, causal_conv1d  # noqa: F401
        print("[OK] Qwen3.5 fast-path kernels available")
        return True
    except ImportError as e:
        print(f"[FAIL] Missing kernel: {e}")
        print("  Install: pip install flash-linear-attention causal-conv1d")
        return False


def load_dataset_sharegpt(path):
    """Load ShareGPT JSON -> HuggingFace Dataset with 'messages' column."""
    with open(path) as f:
        raw = json.load(f)

    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    processed = []
    for ex in raw:
        msgs = []
        for turn in ex["conversations"]:
            role = role_map.get(turn["from"])
            if role:
                msgs.append({"role": role, "content": turn["value"]})
        if len(msgs) >= 2:
            processed.append({"messages": msgs})

    return processed


def main():
    args = parse_args()

    if not check_kernels():
        sys.exit(1)

    # Merge model defaults with CLI overrides
    cfg = MODEL_CONFIGS[args.model].copy()
    if args.lr is not None: cfg["lr"] = args.lr
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.grad_accum is not None: cfg["grad_accum"] = args.grad_accum
    if args.max_seq_len is not None: cfg["max_seq_len"] = args.max_seq_len
    if args.lora_r is not None: cfg["lora_r"] = args.lora_r
    if args.lora_alpha is not None: cfg["lora_alpha"] = args.lora_alpha

    # Output dir: output/{model_tag}/run_{timestamp}
    tag = model_tag(args.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, tag, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save full config for reproducibility
    run_config = {
        "model": args.model,
        "model_tag": tag,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        **cfg,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training: {args.model}")
    print(f"  Output:   {output_dir}")
    print(f"  Config:   {json.dumps(cfg, indent=2)}")
    print(f"{'='*60}")

    # ── Data ──
    print(f"\n[1/4] Loading dataset: {args.dataset}")
    data = load_dataset_sharegpt(args.dataset)
    print(f"  {len(data)} examples")

    ds = Dataset.from_list(data)
    if args.eval_split > 0 and len(data) > 20:
        split = ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds) if eval_ds else 0}")

    # ── Model ──
    print(f"\n[2/4] Loading model: {args.model} (bf16)")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    # ── LoRA ──
    print(f"\n[3/4] Applying LoRA (r={cfg['lora_r']}, alpha={cfg['lora_alpha']})...")
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"  Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

    # Save parameter counts for cross-scale analysis
    run_config["trainable_params"] = trainable
    run_config["total_params"] = total
    run_config["trainable_pct"] = round(pct, 4)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # ── Training ──
    print(f"\n[4/4] Training ({args.epochs} epochs)...")
    eff_batch = cfg["batch_size"] * cfg["grad_accum"]
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.05))

    print(f"  Effective batch: {eff_batch}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup}")
    print(f"  LR: {cfg['lr']}, Max seq len: {cfg['max_seq_len']}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_steps=warmup,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        max_length=cfg["max_seq_len"],
        max_grad_norm=0.3,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=3,
        load_best_model_at_end=False,
        optim="adamw_torch_fused",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # Save
    print(f"\nSaving to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.state.log_history
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Write completion marker for orchestrator
    with open(os.path.join(output_dir, "TRAINING_COMPLETE"), "w") as f:
        f.write(datetime.now().isoformat())

    print(f"\n{'='*60}")
    print(f"  Training complete: {output_dir}")
    print(f"  Model: {args.model} | Epochs: {args.epochs}")
    print(f"  Trainable: {trainable:,} ({pct:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
