#!/usr/bin/env python3
"""
Behavioral Divergence Analysis (Zhu et al. 2025 Adaptation)
=============================================================
Computes token-level divergence metrics between base and fine-tuned models:

1. AGREEMENT RATE: How often does the fine-tuned model's top predicted token
   match the base model's top prediction? Low agreement = fine-tuning changed
   the model's prediction behavior. Computed per-token across all eval responses.

2. ENTROPY DELTA: Does fine-tuning make the model more or less certain?
   Measured as H(ft) - H(base) at each token position. Negative = more certain
   after fine-tuning. Computed using Shannon entropy of the top-k distribution.

3. DOMAIN VOCABULARY RANK SHIFT: For a curated set of domain-specific tokens,
   how does their rank in the probability distribution change from base to FT?
   Tokens that move from rank 500+ to top-50 indicate domain knowledge encoding.

These metrics complement the response-level vocabulary shift analysis already
completed, providing mechanistic evidence of HOW fine-tuning altered the model's
token-level prediction behavior.

Usage:
    # Run on base model, save logits
    python divergence.py --model Qwen/Qwen3.5-9B --label 9b-base --output results/

    # Run on fine-tuned model, save logits
    python divergence.py --model merged_qwen3.5-9b --label 9b-ft --output results/

    # Compare base vs fine-tuned
    python divergence.py --compare results/div_9b-base.json results/div_9b-ft.json --output results/

Hardware: Requires enough memory to load the model + generate with logit capture.
Time: ~2x slower than normal eval due to logit storage (estimate 60-90 min per model on 9B).
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Evaluation prompts (subset of the 45-question suite) ────
# Strategic subset: 15 questions covering all 7 knowledge subcategories
# plus diagnostic questioning and epistemic calibration
DIVERGENCE_SUITE = [
    {"id": "PC01", "ks": "D", "prompt": "A vendor report says there are 50,000 data engineers in our metro area. Our recruiting team says that proves there's plenty of talent and we just need better sourcing. Does the data support their conclusion?"},
    {"id": "PC03", "ks": "E", "prompt": "We've decided to go fully remote so location doesn't matter anymore. We can hire anyone anywhere for any role. Can you confirm this is the right approach for our engineering expansion?"},
    {"id": "DS02", "ks": "D", "prompt": "Our team scraped LinkedIn to build a talent map of ML engineers in the Southeast. We found 8,200 profiles. How reliable is this as a basis for our hiring strategy?"},
    {"id": "DS04", "ks": "D", "prompt": "I've been using BLS Occupational Employment Statistics to benchmark our local engineering salaries. Is this sufficient?"},
    {"id": "MS01", "ks": "C", "prompt": "Three things happened this month: our main competitor posted 15 new AI engineering roles (up from 2/month), our senior ML engineer just got a LinkedIn message from their recruiter, and the competitor's CEO gave a keynote about 'AI-first transformation.' What's the picture here?"},
    {"id": "LS01", "ks": "A", "prompt": "The weekly skills tracking feed shows that 'LLM fine-tuning' mentions in our industry's job postings increased 200% quarter-over-quarter, from 15 to 45 postings. Is this something we should act on or just monitor?"},
    {"id": "LF01", "ks": "E", "prompt": "We're considering opening a 30-person analytics team in Denver versus Salt Lake City. What should I be looking at to make this recommendation?"},
    {"id": "SM02", "ks": "F", "prompt": "Our compensation analysis shows we're 20% below market for data science roles. The CFO has already said there's no budget for raises this year. How do I present this?"},
    {"id": "AO01", "ks": "G", "prompt": "I've been doing ad-hoc talent intelligence projects for two years. My VP wants me to propose an 'always-on' intelligence capability. What should that look like?"},
    {"id": "JD01", "ks": "A", "prompt": "Our hiring manager wrote a job description for a 'Senior Full-Stack AI Engineer' requiring React, Python, PyTorch, Kubernetes, 8+ years experience, and a PhD. Compensation range is $160-190K. Can you help me evaluate this against the market?"},
    {"id": "CT01", "ks": "F", "prompt": "Our time-to-fill for senior engineers has increased from 55 to 90 days over the past year. How should I present this to our CFO?"},
    {"id": "DQ01", "ks": "D", "prompt": "We need to hire more engineers. Can you help?"},
    {"id": "EC01", "ks": "D", "prompt": "Will our competitors hire more or fewer engineers next year?"},
    {"id": "EC02", "ks": "D", "prompt": "What's the exact salary for a senior data scientist?"},
    {"id": "JD03", "ks": "B", "prompt": "A startup is offering our senior engineer candidate $140K base plus $200K in stock options over 4 years. We're offering $185K base with no equity. They say the startup offer is worth more. How do I evaluate this?"},
]

SYSTEM_PROMPT = (
    "You are an experienced talent market intelligence analyst specializing "
    "in workforce analytics and labor market interpretation. You provide "
    "nuanced, evidence-aware analysis that distinguishes signal from noise "
    "in talent market data. You reason through problems by evaluating data "
    "sources critically, considering context, and delivering actionable "
    "recommendations grounded in analytical rigor rather than vendor "
    "marketing or hype. You ask sharp, diagnostic questions before giving "
    "answers because the right question often matters more than the right "
    "data. When you don't know something, you say so. When a data source "
    "has limitations, you name them. Your goal is to make your stakeholders "
    "smarter about how they think about talent markets, not just to give "
    "them the answer they want to hear."
)

# Domain-specific tokens to track rank shifts
DOMAIN_TOKENS = [
    "compensation", "salary", "benchmark", "percentile", "attrition",
    "pipeline", "sourcing", "retention", "equity", "vesting",
    "stakeholder", "headcount", "workforce", "labor", "talent",
    "methodology", "skeptic", "caveat", "nuance", "calibrat",
    "competitor", "poaching", "feasibility", "scenario", "dashboard",
    "diagnostic", "scoping", "clarif", "however", "depends",
]


def load_model(model_path):
    """Load model and tokenizer."""
    print(f"  Loading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate_with_logits(model, tokenizer, prompt, max_new_tokens=512, top_k_save=100):
    """
    Generate response while capturing top-k logits at each step.
    Returns: response text, list of per-token logit snapshots
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Extract per-token logit snapshots
    token_snapshots = []
    generated_ids = outputs.sequences[0][input_len:]

    for step_idx, scores in enumerate(outputs.scores):
        if step_idx >= len(generated_ids):
            break

        # scores shape: [1, vocab_size]
        logits = scores[0].float()  # Convert from bf16 for numerical stability

        # Top-k tokens and their log-probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        topk = torch.topk(log_probs, k=min(top_k_save, logits.shape[-1]))

        # Shannon entropy of the full distribution
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()

        # The actually generated token
        gen_token_id = generated_ids[step_idx].item()
        gen_token_logprob = log_probs[gen_token_id].item()

        # Domain token ranks
        domain_ranks = {}
        sorted_indices = torch.argsort(logits, descending=True)
        rank_lookup = torch.zeros(logits.shape[-1], dtype=torch.long, device=logits.device)
        rank_lookup[sorted_indices] = torch.arange(logits.shape[-1], device=logits.device)

        for dt in DOMAIN_TOKENS:
            dt_ids = tokenizer.encode(dt, add_special_tokens=False)
            if dt_ids:
                first_id = dt_ids[0]
                if first_id < logits.shape[-1]:
                    domain_ranks[dt] = rank_lookup[first_id].item()

        snapshot = {
            "step": step_idx,
            "generated_token_id": gen_token_id,
            "generated_token": tokenizer.decode([gen_token_id]),
            "generated_logprob": round(gen_token_logprob, 4),
            "entropy": round(entropy, 4),
            "top1_token_id": topk.indices[0].item(),
            "top1_token": tokenizer.decode([topk.indices[0].item()]),
            "top1_logprob": round(topk.values[0].item(), 4),
            "top5_tokens": [tokenizer.decode([tid.item()]) for tid in topk.indices[:5]],
            "top5_logprobs": [round(lp.item(), 4) for lp in topk.values[:5]],
            "domain_token_ranks": domain_ranks,
        }
        token_snapshots.append(snapshot)

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # Clean leaked markers
    for marker in ["\nuser\n", "\nUser:", "\nHuman:", "<|im_end|>", "<|im_start|>"]:
        if marker in response:
            response = response[:response.index(marker)]

    return response.strip(), token_snapshots


def run_divergence_capture(model, tokenizer, label):
    """Run logit capture on all divergence suite questions."""
    results = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(DIVERGENCE_SUITE),
        "questions": [],
    }

    print(f"\n  Capturing logits: {label} ({len(DIVERGENCE_SUITE)} questions)")
    for i, q in enumerate(DIVERGENCE_SUITE):
        print(f"  [{i+1}/{len(DIVERGENCE_SUITE)}] {q['id']}...", end=" ", flush=True)
        start = time.time()
        response, snapshots = generate_with_logits(model, tokenizer, q["prompt"])
        elapsed = time.time() - start

        avg_entropy = sum(s["entropy"] for s in snapshots) / len(snapshots) if snapshots else 0
        print(f"{len(snapshots)} tokens, H={avg_entropy:.2f}, {elapsed:.1f}s")

        results["questions"].append({
            "id": q["id"],
            "ks": q["ks"],
            "prompt": q["prompt"],
            "response": response,
            "num_tokens": len(snapshots),
            "avg_entropy": round(avg_entropy, 4),
            "time_seconds": round(elapsed, 1),
            "token_snapshots": snapshots,
        })

    return results


def compare_divergence(base_data, ft_data, output_path):
    """
    Compare base and fine-tuned logit captures to compute:
    1. Agreement rate (per question and overall)
    2. Entropy delta (per question and overall)
    3. Domain token rank shift (aggregated)
    """
    print(f"\n  Comparing: {base_data['label']} vs {ft_data['label']}")

    comparison = {
        "base_label": base_data["label"],
        "ft_label": ft_data["label"],
        "timestamp": datetime.now().isoformat(),
        "per_question": [],
        "summary": {},
    }

    all_agreements = []
    all_entropy_deltas = []
    all_domain_rank_shifts = defaultdict(list)

    for bq, fq in zip(base_data["questions"], ft_data["questions"]):
        assert bq["id"] == fq["id"], f"Question mismatch: {bq['id']} vs {fq['id']}"

        # Compare token by token up to the shorter sequence
        min_len = min(len(bq["token_snapshots"]), len(fq["token_snapshots"]))
        if min_len == 0:
            continue

        agreements = 0
        entropy_deltas = []

        for step in range(min_len):
            bs = bq["token_snapshots"][step]
            fs = fq["token_snapshots"][step]

            # Agreement: do base and FT predict the same top-1 token?
            if bs["top1_token_id"] == fs["top1_token_id"]:
                agreements += 1

            # Entropy delta: H(ft) - H(base)
            entropy_deltas.append(fs["entropy"] - bs["entropy"])

        agreement_rate = agreements / min_len
        avg_entropy_delta = sum(entropy_deltas) / len(entropy_deltas)

        all_agreements.append(agreement_rate)
        all_entropy_deltas.append(avg_entropy_delta)

        # Domain token rank shifts
        q_rank_shifts = {}
        for dt in DOMAIN_TOKENS:
            base_ranks = [s["domain_token_ranks"].get(dt) for s in bq["token_snapshots"] if dt in s.get("domain_token_ranks", {})]
            ft_ranks = [s["domain_token_ranks"].get(dt) for s in fq["token_snapshots"] if dt in s.get("domain_token_ranks", {})]

            if base_ranks and ft_ranks:
                avg_base_rank = sum(base_ranks) / len(base_ranks)
                avg_ft_rank = sum(ft_ranks) / len(ft_ranks)
                shift = avg_base_rank - avg_ft_rank  # Positive = moved up (lower rank = higher probability)
                q_rank_shifts[dt] = round(shift, 1)
                all_domain_rank_shifts[dt].append(shift)

        comparison["per_question"].append({
            "id": bq["id"],
            "ks": bq["ks"],
            "agreement_rate": round(agreement_rate, 4),
            "avg_entropy_delta": round(avg_entropy_delta, 4),
            "base_avg_entropy": round(bq["avg_entropy"], 4),
            "ft_avg_entropy": round(fq["avg_entropy"], 4),
            "base_tokens": len(bq["token_snapshots"]),
            "ft_tokens": len(fq["token_snapshots"]),
            "compared_tokens": min_len,
            "domain_rank_shifts": q_rank_shifts,
        })

    # Overall summary
    comparison["summary"] = {
        "overall_agreement_rate": round(sum(all_agreements) / len(all_agreements), 4) if all_agreements else 0,
        "overall_entropy_delta": round(sum(all_entropy_deltas) / len(all_entropy_deltas), 4) if all_entropy_deltas else 0,
        "interpretation": {
            "agreement_rate": "Fraction of tokens where base and FT predict the same top-1 token. Lower = more behavioral change.",
            "entropy_delta": "H(ft) - H(base). Negative = FT is more certain. Positive = FT is less certain.",
        },
    }

    # By knowledge subcategory
    ks_groups = defaultdict(list)
    for q in comparison["per_question"]:
        ks_groups[q["ks"]].append(q)

    comparison["summary"]["by_knowledge_subcategory"] = {}
    for ks in sorted(ks_groups.keys()):
        qs = ks_groups[ks]
        comparison["summary"]["by_knowledge_subcategory"][ks] = {
            "agreement_rate": round(sum(q["agreement_rate"] for q in qs) / len(qs), 4),
            "entropy_delta": round(sum(q["avg_entropy_delta"] for q in qs) / len(qs), 4),
            "n_questions": len(qs),
        }

    # Domain token rank shifts (aggregated)
    comparison["summary"]["domain_token_rank_shifts"] = {}
    for dt in sorted(all_domain_rank_shifts.keys()):
        shifts = all_domain_rank_shifts[dt]
        avg_shift = sum(shifts) / len(shifts)
        comparison["summary"]["domain_token_rank_shifts"][dt] = {
            "avg_rank_shift": round(avg_shift, 1),
            "direction": "promoted" if avg_shift > 0 else "demoted",
            "n_observations": len(shifts),
        }

    # Save
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BEHAVIORAL DIVERGENCE: {base_data['label']} vs {ft_data['label']}")
    print(f"{'='*60}")
    print(f"  Overall agreement rate: {comparison['summary']['overall_agreement_rate']:.1%}")
    print(f"  Overall entropy delta:  {comparison['summary']['overall_entropy_delta']:+.3f}")
    print(f"\n  By knowledge subcategory:")
    for ks, data in comparison["summary"]["by_knowledge_subcategory"].items():
        print(f"    {ks}: agree={data['agreement_rate']:.1%}  \u0394H={data['entropy_delta']:+.3f} ({data['n_questions']} Qs)")
    print(f"\n  Top domain token rank shifts (positive = promoted):")
    shifts = comparison["summary"]["domain_token_rank_shifts"]
    for dt in sorted(shifts.keys(), key=lambda k: -abs(shifts[k]["avg_rank_shift"]))[:10]:
        s = shifts[dt]
        arrow = "\u25B2" if s["avg_rank_shift"] > 0 else "\u25BC"
        print(f"    {arrow} {dt:20s}: {s['avg_rank_shift']:+.1f} ranks ({s['direction']})")
    print(f"{'='*60}")

    return comparison


def main():
    p = argparse.ArgumentParser(description="Behavioral Divergence Analysis")
    p.add_argument("--model", type=str, help="Model path for logit capture")
    p.add_argument("--label", type=str, help="Label for this run")
    p.add_argument("--compare", nargs=2, type=str, help="Compare two logit capture JSONs")
    p.add_argument("--output", type=str, default="results", help="Output directory")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.compare:
        # Comparison mode
        base_path, ft_path = args.compare
        with open(base_path) as f:
            base_data = json.load(f)
        with open(ft_path) as f:
            ft_data = json.load(f)
        output_path = os.path.join(args.output, f"divergence_{base_data['label']}_vs_{ft_data['label']}.json")
        compare_divergence(base_data, ft_data, output_path)

    elif args.model and args.label:
        # Capture mode
        model, tokenizer = load_model(args.model)
        results = run_divergence_capture(model, tokenizer, args.label)

        # Save (without token snapshots for the summary, full version separately)
        summary_path = os.path.join(args.output, f"div_{args.label}.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {summary_path}")

    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
