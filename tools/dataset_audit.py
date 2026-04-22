#!/usr/bin/env python3
"""
Dataset Structural Audit
========================
Analyzes a ShareGPT-format training dataset and flags structural issues
that empirically degrade fine-tuning quality.

Every check in this script corresponds to a lesson learned during
iterative training-evaluation cycles. See docs/practitioners_guide.md
for the full context behind each finding.

Usage:
    python dataset_audit.py your_dataset.json
    python dataset_audit.py your_dataset.json --config domain_config.yaml
    python dataset_audit.py your_dataset.json --format sharegpt
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ── Defaults (overridden by domain_config.yaml if provided) ──

DEFAULTS = {
    "min_examples": 100,
    "max_examples": 300,
    "target_multi_turn_pct": 0.25,
    "max_single_length_cluster_pct": 0.40,
    "target_lead_with_insight_pct": 0.20,
    "target_diagnostic_question_pct": 0.25,
    "max_response_words": 500,
    "min_response_words": 30,
}


def log(msg, level="INFO"):
    colors = {
        "INFO": "\033[94m", "OK": "\033[92m", "WARN": "\033[93m",
        "FAIL": "\033[91m", "SECTION": "\033[95m", "DIM": "\033[90m",
    }
    reset = "\033[0m"
    prefix = colors.get(level, "")
    print(f"{prefix}[{level}]{reset} {msg}")


def load_config(config_path):
    """Load thresholds from domain config YAML."""
    if not HAS_YAML:
        log("PyYAML not installed — using default thresholds. Install with: pip install pyyaml", "WARN")
        return DEFAULTS

    with open(config_path) as f:
        config = yaml.safe_load(f)

    thresholds = config.get("audit_thresholds", {})
    merged = {**DEFAULTS, **thresholds}
    return merged


def load_dataset(path, fmt="sharegpt"):
    """Load a ShareGPT-format JSON dataset."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        log("Dataset is not a JSON array", "FAIL")
        sys.exit(1)

    return data


def strip_think_tags(text):
    """Remove <think>...</think> blocks to analyze visible content only."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_responses(example):
    """Extract assistant responses from a ShareGPT example."""
    return [
        msg["value"]
        for msg in example.get("conversations", [])
        if msg.get("from") == "gpt"
    ]


def extract_system_prompt(example):
    """Extract system prompt from a ShareGPT example."""
    for msg in example.get("conversations", []):
        if msg.get("from") == "system":
            return msg["value"]
    return None


def word_count(text):
    """Count words in visible text (excluding think blocks)."""
    visible = strip_think_tags(text)
    return len(visible.split())


def leads_with_insight(text):
    """
    Heuristic: does the response lead with a recommendation or actionable
    statement rather than building context first?

    Checks if the first sentence (after think blocks) contains action
    verbs or recommendation language.
    """
    visible = strip_think_tags(text)
    first_sentence = visible.split(".")[0].lower() if visible else ""
    lead_signals = [
        "recommend", "start with", "don't ", "do not", "my advice",
        "the answer is", "the short answer", "first,", "stop ",
        "before you", "you should", "i'd suggest", "here's what",
        "the key issue", "the real question", "what you need",
    ]
    return any(signal in first_sentence for signal in lead_signals)


def has_diagnostic_question(text):
    """
    Heuristic: does the response ask a scoping/diagnostic question
    as part of its analytical approach?

    Checks for question marks in the visible response, excluding
    rhetorical framing.
    """
    visible = strip_think_tags(text)
    # Look for question marks that aren't just "what is X?"
    questions = re.findall(r"[^.!]*\?", visible)
    # Filter out very short questions (likely rhetorical)
    substantive = [q for q in questions if len(q.split()) > 4]
    return len(substantive) > 0


def length_bucket(wc, bucket_size=100):
    """Assign a word count to a bucket (0-99, 100-199, etc.)."""
    return f"{(wc // bucket_size) * bucket_size}-{(wc // bucket_size) * bucket_size + bucket_size - 1}"


# ── AUDIT CHECKS ─────────────────────────────────────────────

def audit_dataset_size(data, thresholds):
    """Check 1: Is the dataset in the productive range?"""
    log("=" * 60)
    log("CHECK 1: Dataset Size", "SECTION")
    log("=" * 60)

    n = len(data)
    log(f"  Examples: {n}")

    if n < thresholds["min_examples"]:
        log(
            f"  Below minimum ({thresholds['min_examples']}). Domain encoding will be "
            f"present but less calibrated. Consider adding more examples.",
            "WARN"
        )
    elif n > thresholds["max_examples"]:
        log(
            f"  Above maximum ({thresholds['max_examples']}). Risk of general knowledge "
            f"erosion. Monitor general-purpose performance after training.",
            "WARN"
        )
    else:
        log(f"  In the productive range ({thresholds['min_examples']}-{thresholds['max_examples']})", "OK")

    return n


def audit_system_prompts(data):
    """Check 2: Is there exactly one system prompt?"""
    log("")
    log("=" * 60)
    log("CHECK 2: System Prompt Consistency", "SECTION")
    log("=" * 60)

    prompts = set()
    missing = 0
    for ex in data:
        sp = extract_system_prompt(ex)
        if sp is None:
            missing += 1
        else:
            prompts.add(sp.strip())

    if missing > 0:
        log(f"  {missing} examples have no system prompt", "WARN")
    if len(prompts) > 1:
        log(f"  CRITICAL: {len(prompts)} different system prompts detected", "FAIL")
        log("  Multiple system prompts cause catastrophic interference during training.", "FAIL")
        log("  Normalize to a single prompt before training.", "FAIL")
        for i, p in enumerate(prompts):
            log(f"  Prompt {i+1} (first 80 chars): {p[:80]}...", "DIM")
    elif len(prompts) == 1:
        log("  Single normalized system prompt across all examples", "OK")
    else:
        log("  No system prompts found", "WARN")

    return len(prompts)


def audit_response_lengths(data, thresholds):
    """Check 3: Response length distribution — flags clustering."""
    log("")
    log("=" * 60)
    log("CHECK 3: Response Length Distribution", "SECTION")
    log("=" * 60)

    all_lengths = []
    too_long = 0
    too_short = 0

    for ex in data:
        for resp in extract_responses(ex):
            wc = word_count(resp)
            all_lengths.append(wc)
            if wc > thresholds["max_response_words"]:
                too_long += 1
            if wc < thresholds["min_response_words"]:
                too_short += 1

    if not all_lengths:
        log("  No assistant responses found", "FAIL")
        return []

    avg = sum(all_lengths) / len(all_lengths)
    log(f"  Total responses: {len(all_lengths)}")
    log(f"  Length range: {min(all_lengths)}-{max(all_lengths)} words")
    log(f"  Average: {avg:.0f} words")

    if too_long:
        log(f"  {too_long} responses exceed {thresholds['max_response_words']} words — consider trimming", "WARN")
    if too_short:
        log(f"  {too_short} responses under {thresholds['min_response_words']} words", "DIM")

    # Check for clustering
    buckets = Counter(length_bucket(wc) for wc in all_lengths)
    total = len(all_lengths)
    max_bucket_pct = max(count / total for count in buckets.values())
    max_bucket_name = max(buckets, key=buckets.get)

    if max_bucket_pct > thresholds["max_single_length_cluster_pct"]:
        log(
            f"  {max_bucket_pct:.0%} of responses cluster in the {max_bucket_name} word range",
            "WARN"
        )
        log(
            "  The model will learn this length as default regardless of question complexity.",
            "WARN"
        )
        log("  Deliberately vary response lengths to teach proportionality.", "WARN")
    else:
        log(f"  Length distribution is varied (largest bucket: {max_bucket_pct:.0%} in {max_bucket_name}w)", "OK")

    log("")
    log("  Distribution:", "DIM")
    for bucket in sorted(buckets.keys(), key=lambda x: int(x.split("-")[0])):
        count = buckets[bucket]
        pct = count / total
        bar = "█" * int(pct * 40)
        log(f"    {bucket:>10}w  {bar} {count} ({pct:.0%})", "DIM")

    return all_lengths


def audit_structure_mix(data, thresholds):
    """Check 4: Single-turn vs multi-turn ratio."""
    log("")
    log("=" * 60)
    log("CHECK 4: Structural Mix (Single-Turn vs Multi-Turn)", "SECTION")
    log("=" * 60)

    single = 0
    multi = 0
    turn_counts = []

    for ex in data:
        n_responses = len(extract_responses(ex))
        turn_counts.append(n_responses)
        if n_responses == 1:
            single += 1
        else:
            multi += 1

    total = single + multi
    multi_pct = multi / total if total else 0

    log(f"  Single-turn: {single} ({single/total:.0%})")
    log(f"  Multi-turn:  {multi} ({multi_pct:.0%})")

    if multi_pct < thresholds["target_multi_turn_pct"]:
        log(
            f"  Below target ({thresholds['target_multi_turn_pct']:.0%}). Consider converting "
            f"some single-turn examples into multi-turn diagnostic sequences.",
            "WARN"
        )
    else:
        log(f"  Multi-turn representation meets target ({thresholds['target_multi_turn_pct']:.0%})", "OK")

    return single, multi


def audit_lead_with_insight(data, thresholds):
    """Check 5: Do enough responses lead with the recommendation?"""
    log("")
    log("=" * 60)
    log("CHECK 5: Lead-with-Insight Ratio", "SECTION")
    log("=" * 60)

    leads = 0
    total = 0

    for ex in data:
        responses = extract_responses(ex)
        if responses:
            # Check first assistant response
            total += 1
            if leads_with_insight(responses[0]):
                leads += 1

    lead_pct = leads / total if total else 0

    log(f"  Responses leading with insight/recommendation: {leads}/{total} ({lead_pct:.0%})")

    if lead_pct < thresholds["target_lead_with_insight_pct"]:
        log(
            f"  Below target ({thresholds['target_lead_with_insight_pct']:.0%}). "
            f"If most responses build context first, the model will bury the lead.",
            "WARN"
        )
        log("  Restructure some responses to lead with the actionable insight.", "WARN")
    else:
        log(f"  Meets target ({thresholds['target_lead_with_insight_pct']:.0%})", "OK")

    return leads, total


def audit_diagnostic_questioning(data, thresholds):
    """Check 6: Do enough responses demonstrate diagnostic questioning?"""
    log("")
    log("=" * 60)
    log("CHECK 6: Diagnostic Questioning Representation", "SECTION")
    log("=" * 60)

    has_dq = 0
    total = 0

    for ex in data:
        responses = extract_responses(ex)
        if responses:
            total += 1
            if has_diagnostic_question(responses[0]):
                has_dq += 1

    dq_pct = has_dq / total if total else 0

    log(f"  Responses with diagnostic questions: {has_dq}/{total} ({dq_pct:.0%})")

    if dq_pct < thresholds["target_diagnostic_question_pct"]:
        log(
            f"  Below target ({thresholds['target_diagnostic_question_pct']:.0%}). "
            f"Diagnostic questioning contradicts the base model's tendency to answer "
            f"immediately and requires concentrated representation to transfer.",
            "WARN"
        )
    else:
        log(f"  Meets target ({thresholds['target_diagnostic_question_pct']:.0%})", "OK")

    return has_dq, total


def audit_summary(results):
    """Print a summary of all checks."""
    log("")
    log("=" * 60)
    log("AUDIT SUMMARY", "SECTION")
    log("=" * 60)

    warnings = results.get("warnings", 0)
    failures = results.get("failures", 0)

    if failures > 0:
        log(f"  {failures} critical issue(s) — fix before training", "FAIL")
    if warnings > 0:
        log(f"  {warnings} warning(s) — review and consider adjusting", "WARN")
    if failures == 0 and warnings == 0:
        log("  All checks passed", "OK")

    log("")
    log("  Key metrics:", "DIM")
    log(f"    Dataset size:          {results['n_examples']}", "DIM")
    log(f"    System prompts:        {results['n_prompts']}", "DIM")
    log(f"    Avg response length:   {results['avg_length']:.0f} words", "DIM")
    log(f"    Multi-turn ratio:      {results['multi_turn_pct']:.0%}", "DIM")
    log(f"    Lead-with-insight:     {results['lead_pct']:.0%}", "DIM")
    log(f"    Diagnostic questions:  {results['dq_pct']:.0%}", "DIM")


def main():
    parser = argparse.ArgumentParser(
        description="Audit a training dataset's structural distribution",
        epilog="Each check corresponds to a lesson from iterative fine-tuning evaluation. "
               "See docs/practitioners_guide.md for the full context.",
    )
    parser.add_argument("dataset", help="Path to ShareGPT-format JSON training dataset")
    parser.add_argument("--config", help="Path to domain_config.yaml (optional, uses defaults otherwise)")
    parser.add_argument("--format", default="sharegpt", choices=["sharegpt"], help="Dataset format")
    args = parser.parse_args()

    # Load config
    thresholds = load_config(args.config) if args.config else DEFAULTS

    # Load data
    data = load_dataset(args.dataset, args.format)

    log("")
    log("=" * 60)
    log(f"DATASET STRUCTURAL AUDIT", "SECTION")
    log(f"  File: {args.dataset}", "DIM")
    log("=" * 60)

    # Run checks
    warnings = 0
    failures = 0

    n = audit_dataset_size(data, thresholds)

    n_prompts = audit_system_prompts(data)
    if n_prompts > 1:
        failures += 1
    elif n_prompts == 0:
        warnings += 1

    lengths = audit_response_lengths(data, thresholds)
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    single, multi = audit_structure_mix(data, thresholds)
    total = single + multi
    multi_pct = multi / total if total else 0

    leads, lead_total = audit_lead_with_insight(data, thresholds)
    lead_pct = leads / lead_total if lead_total else 0

    dq, dq_total = audit_diagnostic_questioning(data, thresholds)
    dq_pct = dq / dq_total if dq_total else 0

    # Count warnings from each check
    if n < thresholds["min_examples"] or n > thresholds["max_examples"]:
        warnings += 1
    if lengths:
        buckets = Counter(length_bucket(wc) for wc in lengths)
        max_bucket_pct = max(count / len(lengths) for count in buckets.values())
        if max_bucket_pct > thresholds["max_single_length_cluster_pct"]:
            warnings += 1
    if multi_pct < thresholds["target_multi_turn_pct"]:
        warnings += 1
    if lead_pct < thresholds["target_lead_with_insight_pct"]:
        warnings += 1
    if dq_pct < thresholds["target_diagnostic_question_pct"]:
        warnings += 1

    audit_summary({
        "n_examples": n,
        "n_prompts": n_prompts,
        "avg_length": avg_length,
        "multi_turn_pct": multi_pct,
        "lead_pct": lead_pct,
        "dq_pct": dq_pct,
        "warnings": warnings,
        "failures": failures,
    })


if __name__ == "__main__":
    main()
