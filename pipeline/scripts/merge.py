#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for standalone deployment.
Parameterized for any Qwen3.5 model size.

Usage:
    python merge.py --adapter output/qwen3.5-9b/run_XXXX --base Qwen/Qwen3.5-9B
    python merge.py --adapter output/qwen3.5-0.8b/run_XXXX --base Qwen/Qwen3.5-0.8B
"""

import argparse
import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--base", required=True, help="Base model ID (e.g., Qwen/Qwen3.5-9B)")
    p.add_argument("--output", default=None, help="Output path (default: merged_{tag})")
    args = p.parse_args()

    tag = args.base.split("/")[-1].lower()
    output_dir = args.output or f"merged_{tag}"

    print(f"\n{'='*60}")
    print(f"  Merging: {args.base}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")

    print(f"\n[1/3] Loading base model: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )

    adapter_path = os.path.abspath(args.adapter)
    print(f"[2/3] Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("[3/3] Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=True
    ).save_pretrained(output_dir)

    # Write completion marker
    with open(os.path.join(output_dir, "MERGE_COMPLETE"), "w") as f:
        json.dump({
            "base_model": args.base,
            "adapter_path": args.adapter,
            "merged_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n  Done! Merged model at: {output_dir}")


if __name__ == "__main__":
    main()
