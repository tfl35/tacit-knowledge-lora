#!/usr/bin/env python3
"""
Cross-Scale Knowledge Fidelity Analysis
========================================
Consumes evaluation JSONs from all model scales and produces:
  1. Knowledge fidelity matrix (7 subcategories × 4 scales)
  2. Behavioral category comparison
  3. Degradation analysis (at what scale does each dimension drop?)
  4. Base vs fine-tuned delta per scale
  5. Summary statistics for the capstone report

Usage:
    python analyze.py results/
    python analyze.py results/ --output results/analysis_report.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime


KNOWLEDGE_SUBCATEGORIES = {
    "A": "Labor Market Interpretation",
    "B": "Compensation & Benefits Analysis",
    "C": "Competitive Intelligence",
    "D": "Data Quality & Methodological Judgment",
    "E": "Strategic Workforce Planning",
    "F": "Stakeholder Communication & Commercial Acumen",
    "G": "Operational Intelligence Design",
}

MODEL_ORDER = ["qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b"]
PARAM_COUNTS = {
    "qwen3.5-9b": 9_000_000_000,
    "qwen3.5-4b": 4_000_000_000,
    "qwen3.5-2b": 2_000_000_000,
    "qwen3.5-0.8b": 800_000_000,
}


def load_results(results_dir):
    """Load all eval JSON files from the results directory."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("eval_") and fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            results[data["label"]] = data
            print(f"  Loaded: {data['label']} ({len(data['tests'])} questions)")
    return results


def build_knowledge_fidelity_matrix(results):
    """
    Build the 7×N knowledge fidelity matrix.
    Rows = knowledge subcategories (A-G)
    Columns = model labels (base and fine-tuned at each scale)
    Values = average signal score
    """
    matrix = {}
    for label, data in results.items():
        matrix[label] = {}
        ks_scores = defaultdict(list)
        for test in data["tests"]:
            ks = test.get("knowledge_sub")
            if ks and test.get("signals_expected"):
                ks_scores[ks].append(test["signal_score"])
        for ks in KNOWLEDGE_SUBCATEGORIES:
            if ks in ks_scores:
                matrix[label][ks] = round(sum(ks_scores[ks]) / len(ks_scores[ks]), 3)
            else:
                matrix[label][ks] = None
    return matrix


def build_category_matrix(results):
    """Build behavioral category × model matrix."""
    matrix = {}
    for label, data in results.items():
        matrix[label] = {}
        cat_scores = defaultdict(list)
        for test in data["tests"]:
            cat = test["category"]
            if test.get("signals_expected"):
                cat_scores[cat].append(test["signal_score"])
        for cat, scores in cat_scores.items():
            matrix[label][cat] = round(sum(scores) / len(scores), 3)
    return matrix


def compute_deltas(matrix, base_suffix="-base", ft_suffix="-ft"):
    """Compute fine-tuned minus base delta for each scale and subcategory."""
    deltas = {}
    for label in matrix:
        if label.endswith(ft_suffix):
            scale = label.replace(ft_suffix, "")
            base_label = scale + base_suffix
            if base_label in matrix:
                deltas[scale] = {}
                for ks in matrix[label]:
                    ft_val = matrix[label].get(ks)
                    base_val = matrix[base_label].get(ks)
                    if ft_val is not None and base_val is not None:
                        deltas[scale][ks] = round(ft_val - base_val, 3)
    return deltas


def find_degradation_thresholds(matrix, ft_suffix="-ft"):
    """
    For each knowledge subcategory, find where performance drops
    significantly relative to the largest model.

    Returns: {subcategory: {threshold_model, drop_pct, scores_by_scale}}
    """
    # Get fine-tuned models only, ordered by size
    ft_labels = [m + ft_suffix for m in MODEL_ORDER if m + ft_suffix in matrix]
    if not ft_labels:
        return {}

    reference_label = ft_labels[0]  # Largest model
    thresholds = {}

    for ks in KNOWLEDGE_SUBCATEGORIES:
        ref_score = matrix.get(reference_label, {}).get(ks)
        if ref_score is None:
            continue

        scores_by_scale = {}
        threshold_model = None
        for label in ft_labels:
            scale = label.replace(ft_suffix, "")
            score = matrix.get(label, {}).get(ks)
            scores_by_scale[scale] = score
            # Significant drop = >20% relative decrease from reference
            if score is not None and ref_score > 0:
                drop = (ref_score - score) / ref_score
                if drop > 0.20 and threshold_model is None:
                    threshold_model = scale

        thresholds[ks] = {
            "subcategory_name": KNOWLEDGE_SUBCATEGORIES[ks],
            "reference_score": ref_score,
            "reference_model": MODEL_ORDER[0],
            "degradation_threshold": threshold_model,
            "scores_by_scale": scores_by_scale,
        }

    return thresholds


def diagnostic_questioning_analysis(results, ft_suffix="-ft"):
    """
    Special analysis for diagnostic questioning, which the lit review
    predicts will resist encoding.
    """
    dq_results = {}
    for label, data in results.items():
        dq_tests = [t for t in data["tests"] if t["category"] == "Diagnostic Questioning"]
        if dq_tests:
            # Check if model asks questions (presence of '?' in response)
            asks_questions = sum(1 for t in dq_tests if "?" in t["response"])
            signal_scores = [t["signal_score"] for t in dq_tests if t.get("signals_expected")]
            avg_signal = sum(signal_scores) / len(signal_scores) if signal_scores else 0
            avg_words = sum(t["word_count"] for t in dq_tests) / len(dq_tests)

            dq_results[label] = {
                "num_tests": len(dq_tests),
                "asks_questions_count": asks_questions,
                "asks_questions_rate": round(asks_questions / len(dq_tests), 2),
                "avg_signal_score": round(avg_signal, 3),
                "avg_word_count": round(avg_words),
            }
    return dq_results


def print_report(kf_matrix, cat_matrix, deltas, thresholds, dq_analysis, results):
    """Print a formatted summary report."""
    print(f"\n{'='*70}")
    print(f"  CROSS-SCALE KNOWLEDGE FIDELITY ANALYSIS")
    print(f"{'='*70}")

    # ── Knowledge Fidelity Matrix ──
    print(f"\n{'─'*70}")
    print("  KNOWLEDGE FIDELITY MATRIX (signal scores, 0-1)")
    print(f"{'─'*70}")

    # Header
    labels = sorted(kf_matrix.keys())
    header = f"  {'Subcategory':42s}"
    for label in labels:
        short = label.replace("qwen3.5-", "").upper()
        header += f" {short:>8s}"
    print(header)
    print(f"  {'─'*42}" + "─" * (9 * len(labels)))

    for ks in sorted(KNOWLEDGE_SUBCATEGORIES.keys()):
        name = KNOWLEDGE_SUBCATEGORIES[ks]
        row = f"  {ks}. {name:39s}"
        for label in labels:
            val = kf_matrix.get(label, {}).get(ks)
            if val is not None:
                row += f" {val:>7.0%} "
            else:
                row += f" {'n/a':>7s} "
        print(row)

    # ── Fine-tuning Deltas ──
    if deltas:
        print(f"\n{'─'*70}")
        print("  FINE-TUNING DELTA (ft - base)")
        print(f"{'─'*70}")

        for scale in MODEL_ORDER:
            if scale in deltas:
                print(f"\n  {scale.upper()}:")
                for ks in sorted(deltas[scale].keys()):
                    d = deltas[scale][ks]
                    arrow = "▲" if d > 0 else "▼" if d < 0 else "="
                    name = KNOWLEDGE_SUBCATEGORIES.get(ks, ks)
                    print(f"    {ks}. {name:39s} {arrow} {d:+.0%}")

    # ── Degradation Thresholds ──
    if thresholds:
        print(f"\n{'─'*70}")
        print("  DEGRADATION ANALYSIS (>20% drop from 9B)")
        print(f"{'─'*70}")

        for ks in sorted(thresholds.keys()):
            t = thresholds[ks]
            threshold = t["degradation_threshold"] or "None (survives all scales)"
            scores = t["scores_by_scale"]
            score_str = " → ".join(
                f"{s.replace('qwen3.5-', '')}:{v:.0%}" if v else f"{s}:n/a"
                for s, v in scores.items()
            )
            print(f"  {ks}. {t['subcategory_name']}")
            print(f"     Drops at: {threshold}")
            print(f"     Scores:   {score_str}")

    # ── Diagnostic Questioning ──
    if dq_analysis:
        print(f"\n{'─'*70}")
        print("  DIAGNOSTIC QUESTIONING ANALYSIS (Nonaka & Takeuchi prediction)")
        print(f"{'─'*70}")
        for label in sorted(dq_analysis.keys()):
            dq = dq_analysis[label]
            print(f"  {label:30s}: asks questions {dq['asks_questions_rate']:.0%} of time, "
                  f"signal {dq['avg_signal_score']:.0%}, {dq['avg_word_count']}w avg")

    # ── Overall Summary ──
    print(f"\n{'─'*70}")
    print("  OVERALL MODEL COMPARISON")
    print(f"{'─'*70}")
    for label in sorted(results.keys()):
        s = results[label].get("summary", {})
        print(f"  {label:30s}: signal {s.get('avg_signal_score', 0):.0%} | "
              f"{s.get('avg_word_count', 0)} avg words | "
              f"{s.get('total_time_seconds', 0):.0f}s total")

    print(f"\n{'='*70}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", help="Directory containing eval_*.json files")
    p.add_argument("--output", default=None, help="Output analysis JSON path")
    args = p.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: {args.results_dir} is not a directory")
        sys.exit(1)

    print(f"\n  Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        print("  No eval_*.json files found!")
        sys.exit(1)

    print(f"\n  Found {len(results)} evaluation runs")

    # Build matrices
    kf_matrix = build_knowledge_fidelity_matrix(results)
    cat_matrix = build_category_matrix(results)
    deltas = compute_deltas(kf_matrix)
    thresholds = find_degradation_thresholds(kf_matrix)
    dq_analysis = diagnostic_questioning_analysis(results)

    # Print formatted report
    print_report(kf_matrix, cat_matrix, deltas, thresholds, dq_analysis, results)

    # Save full analysis
    output_path = args.output or os.path.join(args.results_dir, "cross_scale_analysis.json")
    analysis = {
        "generated_at": datetime.now().isoformat() if 'datetime' in dir() else None,
        "models_evaluated": list(results.keys()),
        "knowledge_fidelity_matrix": kf_matrix,
        "behavioral_category_matrix": cat_matrix,
        "finetuning_deltas": deltas,
        "degradation_thresholds": thresholds,
        "diagnostic_questioning_analysis": dq_analysis,
        "per_model_summaries": {
            label: data.get("summary", {}) for label, data in results.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Full analysis saved: {output_path}")


if __name__ == "__main__":
    main()
