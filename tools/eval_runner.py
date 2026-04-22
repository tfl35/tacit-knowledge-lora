#!/usr/bin/env python3
"""
Domain-Agnostic Evaluation Runner
===================================
Runs a multi-method evaluation of a fine-tuned model using questions
and signal definitions from a domain_config.yaml file.

Supports three evaluation methods:
  1. Signal scoring with fuzzy matching
  2. Visible signal density (signals per 100 words)
  3. LLM-as-judge via Anthropic API (optional, requires ANTHROPIC_API_KEY)

Produces a knowledge fidelity matrix showing per-subcategory performance
and a summary JSON suitable for cross-scale comparison.

Usage:
    # Evaluate via Ollama (simplest)
    python eval_runner.py --config domain_config.yaml --ollama my-model-4b

    # Evaluate via Hugging Face model path
    python eval_runner.py --config domain_config.yaml --model /path/to/merged_model

    # Compare base vs fine-tuned
    python eval_runner.py --config domain_config.yaml --ollama qwen3.5:4b --label base-4b
    python eval_runner.py --config domain_config.yaml --ollama my-model-4b --label ft-4b

    # Include LLM-as-judge (requires ANTHROPIC_API_KEY env var)
    python eval_runner.py --config domain_config.yaml --ollama my-model-4b --judge
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def log(msg, level="INFO"):
    colors = {
        "INFO": "\033[94m", "OK": "\033[92m", "WARN": "\033[93m",
        "FAIL": "\033[91m", "SECTION": "\033[95m", "DIM": "\033[90m",
    }
    reset = "\033[0m"
    prefix = colors.get(level, "")
    print(f"{prefix}[{level}]{reset} {msg}")


# ── CONFIG LOADING ───────────────────────────────────────────

def load_config(config_path):
    """Load domain configuration from YAML."""
    if not HAS_YAML:
        log("PyYAML required. Install with: pip install pyyaml", "FAIL")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["domain", "system_prompt", "subcategories", "eval_questions"]
    for field in required:
        if field not in config:
            log(f"Missing required field in config: {field}", "FAIL")
            sys.exit(1)

    return config


# ── MODEL INFERENCE ──────────────────────────────────────────

def generate_ollama(model_name, system_prompt, question, max_tokens=2048):
    """Generate a response using Ollama CLI."""
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log(f"  Timeout on question (120s)", "WARN")
        return ""
    except FileNotFoundError:
        log("Ollama not found. Install from https://ollama.ai", "FAIL")
        sys.exit(1)


def generate_transformers(model, tokenizer, system_prompt, question, max_tokens=2048):
    """Generate a response using Hugging Face Transformers."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── SIGNAL SCORING ───────────────────────────────────────────

def strip_think_tags(text):
    """Remove <think>...</think> blocks for visible content analysis."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def fuzzy_signal_match(response, signal):
    """
    Enhanced signal matching with hyphenation variants and
    case-insensitive substring matching.
    """
    response_lower = response.lower()
    signal_lower = signal.lower()

    # Direct substring
    if signal_lower in response_lower:
        return True

    # Hyphenation variants (e.g., "cross-functional" / "cross functional")
    if "-" in signal_lower:
        if signal_lower.replace("-", " ") in response_lower:
            return True
        if signal_lower.replace("-", "") in response_lower:
            return True
    else:
        # Try adding hyphens at common points
        if " " in signal_lower:
            if signal_lower.replace(" ", "-") in response_lower:
                return True

    return False


def score_response(response, signals, anti_signals=None):
    """Score a response based on signal presence."""
    visible = strip_think_tags(response)

    matched = [s for s in signals if fuzzy_signal_match(visible, s)]
    score = len(matched) / len(signals) if signals else 0

    anti_matched = []
    if anti_signals:
        anti_matched = [s for s in anti_signals if fuzzy_signal_match(visible, s)]

    word_count = len(visible.split())
    density = (len(matched) / word_count * 100) if word_count > 0 else 0

    return {
        "score": score,
        "matched_signals": matched,
        "total_signals": len(signals),
        "anti_signals_matched": anti_matched,
        "word_count": word_count,
        "density": density,
    }


# ── LLM-AS-JUDGE ────────────────────────────────────────────

def judge_response(question, response, domain_name, what_it_tests):
    """
    Use Claude as an independent qualitative judge.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        log("anthropic package required for judge. Install with: pip install anthropic", "WARN")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    judge_prompt = f"""You are evaluating a domain expert AI model's response in the field of {domain_name}.

The question was designed to test: {what_it_tests}

QUESTION:
{question}

MODEL RESPONSE:
{response}

Score the response on these 5 dimensions (1-5 scale each):

1. Reasoning Strategy: Does the response demonstrate a structured analytical approach?
2. Analytical Depth: Does it go beyond surface-level observations?
3. Epistemic Calibration: Does it express appropriate confidence/uncertainty?
4. Actionability: Can the reader act on the recommendations?
5. Delivery Quality: Is the response well-structured and appropriately concise?

Respond ONLY with a JSON object:
{{"reasoning_strategy": N, "analytical_depth": N, "epistemic_calibration": N, "actionability": N, "delivery_quality": N, "overall": N, "brief_rationale": "one sentence"}}
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = message.content[0].text.strip()
        # Parse JSON from response
        text = re.sub(r"```json\s*|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        log(f"  Judge error: {e}", "WARN")
        return None


# ── MAIN EVALUATION ──────────────────────────────────────────

def run_evaluation(config, generate_fn, label="eval", run_judge=False):
    """Run the full evaluation suite."""
    questions = config["eval_questions"]
    system_prompt = config["system_prompt"]
    domain_name = config["domain"]["name"]
    subcategories = config["subcategories"]

    log(f"\nRunning evaluation: {label}")
    log(f"  Domain: {domain_name}")
    log(f"  Questions: {len(questions)}")
    log(f"  Judge: {'enabled' if run_judge else 'disabled'}")
    log("")

    results = []

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"].strip()
        subcat = q["subcategory"]
        signals = q.get("signals", [])
        anti_signals = q.get("anti_signals", [])
        what_it_tests = q.get("what_it_tests", "")

        log(f"  [{i+1}/{len(questions)}] {qid}: {question[:60]}...", "DIM")

        # Generate response
        start = time.time()
        response = generate_fn(system_prompt, question)
        gen_time = time.time() - start

        # Score
        scoring = score_response(response, signals, anti_signals)

        result = {
            "id": qid,
            "question": question,
            "subcategory": subcat,
            "behavioral_category": q.get("behavioral_category"),
            "what_it_tests": what_it_tests,
            "response": response,
            "score": scoring["score"],
            "matched_signals": scoring["matched_signals"],
            "total_signals": scoring["total_signals"],
            "anti_signals_matched": scoring["anti_signals_matched"],
            "word_count": scoring["word_count"],
            "density": scoring["density"],
            "generation_time": gen_time,
        }

        # Optional LLM-as-judge
        if run_judge and subcat != "general":
            judge_result = judge_response(question, response, domain_name, what_it_tests)
            if judge_result:
                result["judge"] = judge_result
                log(f"    Judge: {judge_result.get('overall', '?')}/5", "DIM")

        log(f"    Score: {scoring['score']:.2f} | Words: {scoring['word_count']} | "
            f"Density: {scoring['density']:.2f}/100w | Time: {gen_time:.1f}s", "DIM")

        results.append(result)

    return results


def compute_knowledge_fidelity_matrix(results, subcategories):
    """Compute per-subcategory signal scores — the knowledge fidelity matrix."""
    matrix = {}
    for subcat_id, subcat_info in subcategories.items():
        subcat_results = [r for r in results if r["subcategory"] == subcat_id]
        if subcat_results:
            avg_score = sum(r["score"] for r in subcat_results) / len(subcat_results)
            avg_density = sum(r["density"] for r in subcat_results) / len(subcat_results)
            avg_words = sum(r["word_count"] for r in subcat_results) / len(subcat_results)
            matrix[subcat_id] = {
                "name": subcat_info["name"],
                "avg_score": round(avg_score, 3),
                "avg_density": round(avg_density, 3),
                "avg_words": round(avg_words, 1),
                "n_questions": len(subcat_results),
            }
    return matrix


def compute_summary(results, matrix, label):
    """Compute overall summary statistics."""
    domain_results = [r for r in results if r["subcategory"] != "general"]
    general_results = [r for r in results if r["subcategory"] == "general"]

    summary = {
        "label": label,
        "n_questions": len(results),
        "overall_signal_score": round(
            sum(r["score"] for r in domain_results) / len(domain_results), 3
        ) if domain_results else 0,
        "overall_density": round(
            sum(r["density"] for r in domain_results) / len(domain_results), 3
        ) if domain_results else 0,
        "avg_word_count": round(
            sum(r["word_count"] for r in domain_results) / len(domain_results), 1
        ) if domain_results else 0,
        "general_knowledge_score": round(
            sum(r["score"] for r in general_results) / len(general_results), 3
        ) if general_results else None,
        "knowledge_fidelity_matrix": matrix,
    }

    # Add judge scores if available
    judged = [r for r in domain_results if "judge" in r]
    if judged:
        avg_judge = sum(r["judge"]["overall"] for r in judged) / len(judged)
        summary["judge_overall"] = round(avg_judge, 2)
        summary["judge_n"] = len(judged)

        # Per-dimension averages
        dims = ["reasoning_strategy", "analytical_depth", "epistemic_calibration",
                "actionability", "delivery_quality"]
        for dim in dims:
            vals = [r["judge"][dim] for r in judged if dim in r["judge"]]
            if vals:
                summary[f"judge_{dim}"] = round(sum(vals) / len(vals), 2)

    return summary


def print_results(summary, matrix):
    """Print formatted results."""
    log("")
    log("=" * 60)
    log(f"RESULTS: {summary['label']}", "SECTION")
    log("=" * 60)

    log(f"  Overall signal score:  {summary['overall_signal_score']:.3f}")
    log(f"  Overall density:       {summary['overall_density']:.2f} signals/100 words")
    log(f"  Avg response length:   {summary['avg_word_count']:.0f} words")

    if summary.get("general_knowledge_score") is not None:
        log(f"  General knowledge:     {summary['general_knowledge_score']:.3f}")

    if summary.get("judge_overall"):
        log(f"  Judge score:           {summary['judge_overall']}/5 (n={summary['judge_n']})")

    log("")
    log("  Knowledge Fidelity Matrix:", "SECTION")
    log(f"  {'Subcategory':<30} {'Score':>8} {'Density':>10} {'Words':>8} {'N':>4}")
    log(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*4}")

    for subcat_id in sorted(matrix.keys()):
        m = matrix[subcat_id]
        log(f"  {subcat_id}. {m['name']:<27} {m['avg_score']:>8.3f} {m['avg_density']:>10.2f} "
            f"{m['avg_words']:>8.1f} {m['n_questions']:>4}")


def main():
    parser = argparse.ArgumentParser(
        description="Domain-agnostic evaluation runner for fine-tuned models",
        epilog="Reads evaluation questions and signal definitions from domain_config.yaml. "
               "Produces signal scores, density metrics, and an optional LLM-as-judge evaluation.",
    )
    parser.add_argument("--config", required=True, help="Path to domain_config.yaml")
    parser.add_argument("--ollama", help="Ollama model name (e.g., my-model-4b)")
    parser.add_argument("--model", help="Hugging Face model path (alternative to --ollama)")
    parser.add_argument("--label", default="eval", help="Label for this evaluation run")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--output", default="results", help="Output directory for results JSON")
    args = parser.parse_args()

    if not args.ollama and not args.model:
        log("Provide either --ollama or --model", "FAIL")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Set up generation function
    if args.ollama:
        generate_fn = lambda sys, q: generate_ollama(args.ollama, sys, q)
        log(f"Using Ollama model: {args.ollama}")
    else:
        log(f"Loading model: {args.model}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto"
        )
        generate_fn = lambda sys, q: generate_transformers(model, tokenizer, sys, q)

    # Run evaluation
    results = run_evaluation(config, generate_fn, args.label, args.judge)

    # Compute analysis
    matrix = compute_knowledge_fidelity_matrix(results, config["subcategories"])
    summary = compute_summary(results, matrix, args.label)

    # Print
    print_results(summary, matrix)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"eval_{args.label}.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    log(f"\nResults saved to {results_path}", "OK")

    summary_path = output_dir / f"summary_{args.label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved to {summary_path}", "OK")


if __name__ == "__main__":
    main()
