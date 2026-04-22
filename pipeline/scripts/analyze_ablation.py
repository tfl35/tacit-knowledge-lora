#!/usr/bin/env python3
"""
Ablation Study Analysis
========================
Processes evaluation results from the 9B model trained on 50, 100, 150, 200,
250, 300, and 350 examples. Produces:

1. Ablation curve: signal score and signal density as a function of dataset size
2. Per-subcategory ablation: which knowledge types need more/fewer examples
3. Minimum viable dataset size per subcategory (where score crosses threshold)
4. Diminishing returns analysis: marginal improvement per additional example

The 350-example result comes from the main pipeline run (eval_qwen3.5-9b-ft.json).
The 50-250 results come from the ablation training runs.

Usage:
    python analyze_ablation.py results/
"""

import json
import os
import sys
from collections import defaultdict


KNOWLEDGE_SUBCATEGORIES = {
    "A": "Labor Market Interpretation",
    "B": "Compensation & Benefits Analysis",
    "C": "Competitive Intelligence",
    "D": "Data Quality & Methodological Judgment",
    "E": "Strategic Workforce Planning",
    "F": "Stakeholder Communication & Commercial Acumen",
    "G": "Operational Intelligence Design",
}

ABLATION_SIZES = [50, 100, 150, 200, 250, 300, 350]


def load_ablation_results(results_dir):
    """Load all ablation eval files plus the main 350-example result."""
    results = {}

    for f in sorted(os.listdir(results_dir)):
        if not f.startswith("eval_") or not f.endswith(".json"):
            continue
        path = os.path.join(results_dir, f)
        with open(path) as fh:
            data = json.load(fh)

        label = data["label"]

        # Match ablation labels: qwen3.5-9b-abl50-ft, qwen3.5-9b-abl100-ft, etc.
        if "abl" in label and "-ft" in label:
            size = int(label.split("abl")[1].split("-")[0])
            results[size] = data
            print(f"  Loaded: {label} ({size} examples)")

        # The main 350-example result
        elif label == "qwen3.5-9b-ft":
            results[350] = data
            print(f"  Loaded: {label} (350 examples)")

    # Also load the 9B base for reference
    for f in os.listdir(results_dir):
        if f.startswith("eval_") and "9b-base" in f:
            path = os.path.join(results_dir, f)
            with open(path) as fh:
                results["base"] = json.load(fh)
            print(f"  Loaded: {results['base']['label']} (base reference)")
            break

    return results


def compute_ablation_metrics(results):
    """Compute signal scores, density, and per-subcategory metrics for each ablation size."""
    metrics = {}

    for size in sorted(results.keys()):
        if size == "base":
            continue

        data = results[size]
        scored = [t for t in data["tests"] if t.get("signals_expected")]

        # Overall signal score
        avg_signal = sum(t["signal_score"] for t in scored) / len(scored) if scored else 0

        # Signal density (signals per 100 words)
        total_signals = sum(len(t["signals_found"]) for t in scored)
        total_words = sum(t["word_count"] for t in scored)
        density = (total_signals / total_words * 100) if total_words > 0 else 0

        # Visible portions (strip <think> tags)
        visible_words = 0
        visible_signals = 0
        for t in scored:
            resp = t["response"]
            if "</think>" in resp:
                resp = resp.split("</think>", 1)[1]
            vis_w = len(resp.split())
            visible_words += vis_w
            # Count signals in visible portion
            r_lower = resp.lower()
            vis_sig = sum(1 for s in t["signals_expected"] if s.lower() in r_lower)
            visible_signals += vis_sig

        visible_density = (visible_signals / visible_words * 100) if visible_words > 0 else 0

        # Per subcategory
        ks_metrics = {}
        ks_groups = defaultdict(list)
        for t in scored:
            if t.get("knowledge_sub"):
                ks_groups[t["knowledge_sub"]].append(t)

        for ks, tests in ks_groups.items():
            ks_sig = sum(t["signal_score"] for t in tests) / len(tests)
            ks_words = sum(t["word_count"] for t in tests)
            ks_found = sum(len(t["signals_found"]) for t in tests)
            ks_density = (ks_found / ks_words * 100) if ks_words > 0 else 0
            ks_metrics[ks] = {
                "signal_score": round(ks_sig, 4),
                "density": round(ks_density, 4),
                "n_questions": len(tests),
            }

        metrics[size] = {
            "n_examples": size,
            "avg_signal_score": round(avg_signal, 4),
            "signal_density": round(density, 4),
            "visible_density": round(visible_density, 4),
            "avg_word_count": round(total_words / len(scored), 0) if scored else 0,
            "by_subcategory": ks_metrics,
        }

    return metrics


def find_minimum_viable_sizes(metrics, threshold_pct=0.80):
    """
    For each subcategory, find the smallest dataset size that achieves
    at least threshold_pct of the 350-example performance.
    """
    if 350 not in metrics:
        return {}

    ref = metrics[350]["by_subcategory"]
    min_viable = {}

    for ks in KNOWLEDGE_SUBCATEGORIES:
        ref_score = ref.get(ks, {}).get("signal_score", 0)
        if ref_score == 0:
            min_viable[ks] = {"name": KNOWLEDGE_SUBCATEGORIES[ks], "min_size": "N/A", "threshold": 0}
            continue

        threshold = ref_score * threshold_pct
        found_size = None

        for size in sorted(metrics.keys()):
            if size == "base":
                continue
            score = metrics[size].get("by_subcategory", {}).get(ks, {}).get("signal_score", 0)
            if score >= threshold:
                found_size = size
                break

        min_viable[ks] = {
            "name": KNOWLEDGE_SUBCATEGORIES[ks],
            "ref_score_at_350": ref_score,
            "threshold_80pct": round(threshold, 4),
            "min_size": found_size if found_size else ">350",
            "scores_by_size": {
                s: metrics[s].get("by_subcategory", {}).get(ks, {}).get("signal_score", 0)
                for s in sorted(metrics.keys()) if s != "base"
            },
        }

    return min_viable


def compute_marginal_returns(metrics):
    """
    Compute marginal improvement per additional example at each dataset size step.
    """
    sizes = sorted(s for s in metrics.keys() if isinstance(s, int))
    marginal = []

    for i in range(1, len(sizes)):
        prev_size = sizes[i-1]
        curr_size = sizes[i]
        delta_examples = curr_size - prev_size

        prev_score = metrics[prev_size]["avg_signal_score"]
        curr_score = metrics[curr_size]["avg_signal_score"]
        delta_score = curr_score - prev_score

        prev_density = metrics[prev_size]["visible_density"]
        curr_density = metrics[curr_size]["visible_density"]
        delta_density = curr_density - prev_density

        marginal.append({
            "from": prev_size,
            "to": curr_size,
            "delta_examples": delta_examples,
            "delta_signal": round(delta_score, 4),
            "delta_density": round(delta_density, 4),
            "signal_per_example": round(delta_score / delta_examples, 6) if delta_examples > 0 else 0,
            "density_per_example": round(delta_density / delta_examples, 6) if delta_examples > 0 else 0,
        })

    return marginal


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else results_dir

    print(f"\n  Loading ablation results from: {results_dir}")
    results = load_ablation_results(results_dir)

    if not results:
        print("  No ablation results found!")
        sys.exit(1)

    sizes_found = sorted(s for s in results.keys() if isinstance(s, int))
    print(f"\n  Found results for sizes: {sizes_found}")

    # Compute metrics
    metrics = compute_ablation_metrics(results)

    # Find minimum viable sizes
    min_viable = find_minimum_viable_sizes(metrics)

    # Compute marginal returns
    marginal = compute_marginal_returns(metrics)

    # Assemble output
    analysis = {
        "study": "Ablation Study: Dataset Size Effect on 9B Model",
        "model": "Qwen3.5-9B",
        "sizes_tested": sizes_found,
        "metrics_by_size": metrics,
        "minimum_viable_dataset_sizes": min_viable,
        "marginal_returns": marginal,
    }

    output_path = os.path.join(output_dir, "ablation_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Print report
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY RESULTS — 9B MODEL")
    print(f"{'='*70}")

    print(f"\n  {'Size':>6} {'Signal':>8} {'Density':>8} {'VisDens':>8} {'Words':>7}")
    print(f"  {'-'*43}")
    for size in sizes_found:
        m = metrics[size]
        print(f"  {size:>6} {m['avg_signal_score']:>7.3f} {m['signal_density']:>7.3f} "
              f"{m['visible_density']:>7.3f} {m['avg_word_count']:>6.0f}")

    print(f"\n  Marginal Returns:")
    for m in marginal:
        print(f"    {m['from']:>3} → {m['to']:>3}: "
              f"\u0394signal={m['delta_signal']:+.4f} "
              f"\u0394density={m['delta_density']:+.4f} "
              f"(signal/ex={m['signal_per_example']:.6f})")

    print(f"\n  Minimum Viable Dataset Size (80% of 350-example performance):")
    for ks in sorted(min_viable.keys()):
        mv = min_viable[ks]
        print(f"    {ks}. {mv['name']:<45s}: {mv['min_size']}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
