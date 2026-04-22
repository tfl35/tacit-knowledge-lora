#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation
========================
Sends existing model responses to Claude Sonnet 4.6 for qualitative scoring
across 5 dimensions: reasoning strategy, analytical depth, epistemic calibration,
actionability, and delivery quality.

Uses the Anthropic API (available in this container's network allowlist).
Evaluates a strategic subset of questions where models diverge most,
plus key theoretical test cases (diagnostic questioning, live signals, switching).

No GPU required — this re-evaluates existing text responses.
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


JUDGE_SYSTEM = """You are an expert evaluator assessing AI model responses to talent intelligence questions. You score responses on 5 dimensions using a 1-5 scale. Be calibrated: a score of 3 means competent but unremarkable, 4 means genuinely strong, 5 is reserved for responses a senior practitioner would admire. Score based on the VISIBLE response quality — analytical reasoning, not keyword matching."""


def build_judge_prompt(question_prompt, what_it_tests, expected_signals, model_response):
    return f"""Evaluate this talent intelligence model response.

QUESTION ASKED: {question_prompt}

WHAT THIS TESTS: {what_it_tests}

EXPECTED REASONING PATTERNS (concepts the response should address):
{', '.join(expected_signals)}

MODEL RESPONSE:
{model_response}

Score on these 5 dimensions (1-5 scale):
1. REASONING STRATEGY: Does it use the right analytical approach? (1=off-topic, 3=competent, 5=expert-level)
2. ANALYTICAL DEPTH: Beyond the obvious? (1=superficial, 3=specific, 5=reveals non-obvious insights)
3. EPISTEMIC CALIBRATION: Knows what it knows vs doesn't? (1=overconfident, 3=hedged appropriately, 5=expert calibration)
4. ACTIONABILITY: Could a stakeholder act on this? (1=abstract, 3=has actionable elements, 5=immediately actionable with trade-offs)
5. DELIVERY QUALITY: Structure, length proportionality, insight-first? (1=incoherent, 3=adequate, 5=exemplary)

Respond in EXACTLY this JSON format, nothing else:
{{"reasoning_strategy": {{"justification": "one sentence", "score": N}}, "analytical_depth": {{"justification": "one sentence", "score": N}}, "epistemic_calibration": {{"justification": "one sentence", "score": N}}, "actionability": {{"justification": "one sentence", "score": N}}, "delivery_quality": {{"justification": "one sentence", "score": N}}}}"""


def call_judge(prompt, api_key=None, max_retries=3):
    """Call Claude Sonnet as judge via Anthropic API."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 600,
        "temperature": 0,
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["content"][0]["text"]
                # Parse JSON from response
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(text)
            elif resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error {resp.status_code}: {resp.text[:200]}")
                return None
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
            print(f"    Raw text: {text[:300]}")
            return None
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def get_visible_response(response):
    """Extract visible portion (after </think> if present)."""
    if '</think>' in response:
        return response.split('</think>', 1)[1].strip()
    return response


def select_strategic_subset(eval_data):
    """
    Select questions that maximize diagnostic value:
    - All diagnostic questioning tests (theoretical prediction)
    - All live signal tests (9B anomaly)
    - Switching test
    - Questions where signal scoring likely misrepresents quality
    - Sample from each knowledge subcategory
    """
    priority_categories = {
        "Diagnostic Questioning",
        "Live Signal Interpretation",
        "Switching",
        "Epistemic Calibration",
    }
    priority_names = {
        "TI prompt activates domain",
        "CEO Miami bias",
        "BLS data for local decisions",
        "Build always-on capability",
        "Competitive intel cadence",
        "M&A talent due diligence",
        "Delivering bad comp news",
        "Equity vs base trade-off",
    }

    selected = []
    ks_covered = set()

    for test in eval_data["tests"]:
        if test["category"] in priority_categories:
            selected.append(test)
            if test.get("knowledge_sub"):
                ks_covered.add(test["knowledge_sub"])
        elif test["name"] in priority_names:
            selected.append(test)
            if test.get("knowledge_sub"):
                ks_covered.add(test["knowledge_sub"])

    # Fill any uncovered subcategories
    for test in eval_data["tests"]:
        ks = test.get("knowledge_sub")
        if ks and ks not in ks_covered and test.get("signals_expected"):
            selected.append(test)
            ks_covered.add(ks)

    # Deduplicate by ID
    seen = set()
    deduped = []
    for t in selected:
        if t["id"] not in seen:
            deduped.append(t)
            seen.add(t["id"])

    return deduped


def run_judge_evaluation(eval_data, api_key=None, subset_only=True):
    """Run LLM-as-judge on a model's responses."""
    label = eval_data["label"]
    tests = select_strategic_subset(eval_data) if subset_only else eval_data["tests"]

    print(f"\n  Judging {label}: {len(tests)} questions")
    results = []

    for i, test in enumerate(tests):
        if not test.get("signals_expected"):
            continue

        visible = get_visible_response(test["response"])
        prompt = build_judge_prompt(
            test["prompt"],
            test["what_it_tests"],
            test["signals_expected"],
            visible,
        )

        print(f"  [{i+1}/{len(tests)}] {test['name']}...", end=" ", flush=True)
        scores = call_judge(prompt, api_key)

        if scores:
            dims = ["reasoning_strategy", "analytical_depth", "epistemic_calibration",
                     "actionability", "delivery_quality"]
            score_vals = [scores[d]["score"] for d in dims if d in scores]
            avg = sum(score_vals) / len(score_vals) if score_vals else 0
            dim_strs = [d[:4] + "=" + str(scores[d]["score"]) for d in dims if d in scores]
            print(f"avg={avg:.1f} [{', '.join(dim_strs)}]")

            results.append({
                "id": test["id"],
                "name": test["name"],
                "category": test["category"],
                "knowledge_sub": test.get("knowledge_sub"),
                "signal_score": test.get("signal_score", 0),
                "judge_scores": scores,
                "judge_avg": round(avg, 2),
                "visible_word_count": len(visible.split()),
            })
        else:
            print("FAILED")
            results.append({
                "id": test["id"],
                "name": test["name"],
                "judge_scores": None,
                "judge_avg": None,
                "error": True,
            })

        time.sleep(0.5)  # Rate limiting buffer

    return results


def aggregate_judge_results(results):
    """Compute summary stats from judge scores."""
    valid = [r for r in results if r.get("judge_scores")]
    if not valid:
        return {}

    dims = ["reasoning_strategy", "analytical_depth", "epistemic_calibration",
            "actionability", "delivery_quality"]

    summary = {
        "n_judged": len(valid),
        "avg_overall": round(sum(r["judge_avg"] for r in valid) / len(valid), 2),
    }

    for dim in dims:
        scores = [r["judge_scores"][dim]["score"] for r in valid if dim in r["judge_scores"]]
        summary[f"avg_{dim}"] = round(sum(scores) / len(scores), 2) if scores else 0

    # By category
    cat_groups = {}
    for r in valid:
        cat = r.get("category", "unknown")
        if cat not in cat_groups:
            cat_groups[cat] = []
        cat_groups[cat].append(r["judge_avg"])

    summary["by_category"] = {
        cat: round(sum(scores) / len(scores), 2)
        for cat, scores in cat_groups.items()
    }

    # Signal score vs judge score correlation
    pairs = [(r["signal_score"], r["judge_avg"]) for r in valid if r.get("signal_score") is not None]
    if len(pairs) >= 3:
        sig_vals = [p[0] for p in pairs]
        judge_vals = [p[1] for p in pairs]
        mean_s = sum(sig_vals) / len(sig_vals)
        mean_j = sum(judge_vals) / len(judge_vals)
        cov = sum((s - mean_s) * (j - mean_j) for s, j in zip(sig_vals, judge_vals)) / len(pairs)
        var_s = sum((s - mean_s) ** 2 for s in sig_vals) / len(pairs)
        var_j = sum((j - mean_j) ** 2 for j in judge_vals) / len(pairs)
        if var_s > 0 and var_j > 0:
            corr = cov / (var_s ** 0.5 * var_j ** 0.5)
            summary["signal_judge_correlation"] = round(corr, 3)

    return summary


def main():
    eval_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    api_key = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        print("ERROR: No API key. Set ANTHROPIC_API_KEY or pass as arg 3.")
        print("Usage: python judge.py [eval_dir] [output_dir] [api_key]")
        sys.exit(1)

    if not HAS_REQUESTS:
        print("Installing requests...")
        os.system("pip install requests --break-system-packages -q")
        import requests

    # Target models: 9B-ft, 4B-ft, 2B-ft (the three fine-tuned models worth comparing)
    targets = ["qwen3.5-9b-ft", "qwen3.5-4b-ft", "qwen3.5-2b-ft"]

    all_judge_results = {}
    all_judge_summaries = {}

    for target in targets:
        # Find the eval file
        fname = f"eval_{target.replace('.', '_').replace('-', '-')}"
        # Try to match filename pattern
        path = None
        for f in os.listdir(eval_dir):
            if f.startswith("eval_") and f.endswith(".json"):
                with open(os.path.join(eval_dir, f)) as fh:
                    data = json.load(fh)
                if data.get("label") == target:
                    path = os.path.join(eval_dir, f)
                    break

        if not path:
            print(f"  Skipping {target}: eval file not found")
            continue

        with open(path) as fh:
            eval_data = json.load(fh)

        results = run_judge_evaluation(eval_data, api_key, subset_only=True)
        summary = aggregate_judge_results(results)

        all_judge_results[target] = results
        all_judge_summaries[target] = summary

        print(f"\n  {target} SUMMARY:")
        print(f"    Overall: {summary.get('avg_overall', 'n/a')}")
        for dim in ["reasoning_strategy", "analytical_depth", "epistemic_calibration",
                     "actionability", "delivery_quality"]:
            print(f"    {dim}: {summary.get(f'avg_{dim}', 'n/a')}")
        if "signal_judge_correlation" in summary:
            print(f"    Signal↔Judge correlation: {summary['signal_judge_correlation']}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "methodology": "LLM-as-judge using Claude Sonnet 4.6, strategic subset, 5 dimensions",
        "judge_model": "claude-sonnet-4-20250514",
        "dimensions": ["reasoning_strategy", "analytical_depth", "epistemic_calibration",
                        "actionability", "delivery_quality"],
        "results": all_judge_results,
        "summaries": all_judge_summaries,
    }

    outpath = os.path.join(output_dir, "judge_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outpath}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  LLM-AS-JUDGE CROSS-MODEL COMPARISON")
    print(f"{'='*70}")
    dims_short = ["reas", "depth", "epist", "action", "deliv"]
    dims_full = ["reasoning_strategy", "analytical_depth", "epistemic_calibration",
                  "actionability", "delivery_quality"]
    header = f"  {'Model':<20} {'Overall':>7}" + "".join(f" {d:>7}" for d in dims_short)
    print(header)
    print(f"  {'─'*65}")
    for target in targets:
        s = all_judge_summaries.get(target, {})
        row = f"  {target:<20} {s.get('avg_overall', 0):>6.2f}"
        for dim in dims_full:
            row += f" {s.get(f'avg_{dim}', 0):>6.2f}"
        print(row)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
