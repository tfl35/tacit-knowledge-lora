#!/usr/bin/env python3
"""
Enhanced Cross-Scale Re-Scoring & Analysis
==========================================
Re-scores all 360 existing model responses with three methodological improvements:

1. FUZZY SIGNAL MATCHING
   - Hyphenation variants: "base rate" matches "base-rate" and vice versa
   - Proximity matching: multi-word signals match when component words appear
     within a reasonable window (60 chars), catching paraphrases like
     "the absolute number is small" for signal "small numbers"
   - Possessive/contraction variants: "who's" matches "who is"

2. THINK-TAG AWARE SCORING
   - Separates <think> content from visible response
   - Reports scores for: full response, visible-only, think-only
   - Base model scaffolding detection: identifies "Thinking Process:" patterns
     in 4B/9B base models and reports scaffolding vs content word counts

3. SIGNAL DENSITY METRIC
   - Signals found per 100 words (controls for verbosity differences)
   - Computed at question, category, subcategory, and model levels
   - Density ratio (ft/base) as primary knowledge transfer metric

Outputs:
   - enhanced_analysis.json: full re-scored results with all new metrics
   - enhanced_summary.json: condensed findings for report tables

Usage:
    python rescore.py /path/to/eval_jsons/ [--output results/]
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Knowledge subcategory definitions ────────────────────────
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
MODEL_PARAMS = {"9b": 9e9, "4b": 4e9, "2b": 2e9, "0.8b": 0.8e9}


# ── Enhanced Signal Matching ─────────────────────────────────

def normalize_text(text):
    """Normalize text for matching: lowercase, normalize whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def fuzzy_signal_check(response, signals):
    """
    Enhanced signal matching with three strategies:
    1. Exact substring match (original behavior)
    2. Hyphenation/spacing variants
    3. Proximity match for multi-word signals

    Returns: (found_signals, score, match_details)
    """
    r_norm = normalize_text(response)
    found = []
    match_details = {}

    for signal in signals:
        s_norm = normalize_text(signal)
        matched = False
        method = None

        # Strategy 1: Exact substring
        if s_norm in r_norm:
            matched = True
            method = "exact"

        # Strategy 2: Hyphenation/spacing variants
        if not matched:
            # "base rate" <-> "base-rate"
            if '-' in s_norm and s_norm.replace('-', ' ') in r_norm:
                matched = True
                method = "hyphen_to_space"
            elif ' ' in s_norm and s_norm.replace(' ', '-') in r_norm:
                matched = True
                method = "space_to_hyphen"

        # Strategy 2b: Contraction variants
        if not matched:
            contractions = {
                "who's": "who is", "what's": "what is", "don't": "do not",
                "doesn't": "does not", "can't": "cannot", "won't": "will not",
                "it's": "it is", "there's": "there is",
            }
            for short, long in contractions.items():
                if short in s_norm:
                    expanded = s_norm.replace(short, long)
                    if expanded in r_norm:
                        matched = True
                        method = "contraction_expand"
                        break
                if long in s_norm:
                    contracted = s_norm.replace(long, short)
                    if contracted in r_norm:
                        matched = True
                        method = "contraction_contract"
                        break

        # Strategy 3: Proximity match for multi-word signals
        # All component words must appear within 60 characters of each other
        if not matched and ' ' in s_norm:
            words = s_norm.split()
            if len(words) >= 2 and all(len(w) > 2 for w in words):
                # Find all positions of each word
                word_positions = {}
                for w in words:
                    positions = [m.start() for m in re.finditer(re.escape(w), r_norm)]
                    if not positions:
                        break
                    word_positions[w] = positions
                else:
                    # All words found — check if any combination is within window
                    if _check_proximity(word_positions, window=60):
                        matched = True
                        method = "proximity"

        if matched:
            found.append(signal)
            match_details[signal] = method

    score = len(found) / len(signals) if signals else 0
    return found, round(score, 4), match_details


def _check_proximity(word_positions, window=60):
    """Check if there exists a set of positions (one per word) all within window."""
    words = list(word_positions.keys())
    if len(words) == 2:
        for p1 in word_positions[words[0]]:
            for p2 in word_positions[words[1]]:
                if abs(p1 - p2) <= window:
                    return True
        return False
    # For 3+ words, use greedy approach
    for anchor in word_positions[words[0]]:
        all_close = True
        for w in words[1:]:
            if not any(abs(anchor - p) <= window for p in word_positions[w]):
                all_close = False
                break
        if all_close:
            return True
    return False


# ── Think-Tag & Scaffolding Parsing ──────────────────────────

def parse_response_structure(response):
    """
    Separate response into think/visible portions and detect scaffolding.

    Returns dict with:
        full_text: complete response
        visible_text: portion after </think> (or full if no think tag)
        think_text: content inside think tags (empty if none)
        has_think_tag: bool
        has_scaffolding: bool (base model "Thinking Process:" pattern)
        scaffolding_words: estimated word count of scaffolding
        visible_words: word count of visible portion
        full_words: total word count
    """
    result = {
        "full_text": response,
        "has_think_tag": False,
        "has_scaffolding": False,
        "think_text": "",
        "visible_text": response,
        "scaffolding_words": 0,
    }

    # Check for <think> tags (fine-tuned models)
    if '</think>' in response:
        result["has_think_tag"] = True
        parts = response.split('</think>', 1)
        result["think_text"] = parts[0].replace('<think>', '').strip()
        result["visible_text"] = parts[1].strip() if len(parts) > 1 else ""

    # Check for base model scaffolding (4B/9B base pattern)
    scaffolding_patterns = [
        r"^Here's a thinking process",
        r"^Thinking Process:",
        r"^Here is my thinking process",
        r"^\d+\.\s+\*\*Analyze the Request",
    ]
    for pattern in scaffolding_patterns:
        if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
            result["has_scaffolding"] = True
            # Estimate scaffolding word count: everything before the actual response
            # Usually marked by a shift from numbered/bulleted analysis to prose
            # Use a heuristic: find the last "---" or transition to the actual response
            break

    result["full_words"] = len(response.split())
    result["visible_words"] = len(result["visible_text"].split())
    result["think_words"] = len(result["think_text"].split())

    return result


# ── Signal Density ───────────────────────────────────────────

def compute_signal_density(signals_found, word_count):
    """Signals per 100 words."""
    if word_count == 0:
        return 0
    return round(len(signals_found) / word_count * 100, 4)


# ── Re-scoring Pipeline ─────────────────────────────────────

def rescore_model(eval_data):
    """Re-score all questions for a single model with enhanced methods."""
    label = eval_data["label"]
    rescored_tests = []

    for test in eval_data["tests"]:
        signals = test.get("signals_expected", [])
        response = test["response"]

        # Parse response structure
        structure = parse_response_structure(response)

        # Original scoring (for comparison)
        orig_found = [s for s in signals if s.lower() in response.lower()]
        orig_score = len(orig_found) / len(signals) if signals else 0

        # Enhanced fuzzy scoring on full response
        fuzzy_found, fuzzy_score, match_details = fuzzy_signal_check(
            response, signals
        )

        # Visible-only scoring (excludes think tags)
        vis_found, vis_score, vis_details = fuzzy_signal_check(
            structure["visible_text"], signals
        )

        # Think-only scoring
        think_found, think_score, think_details = fuzzy_signal_check(
            structure["think_text"], signals
        ) if structure["think_text"] else ([], 0, {})

        # Signal density metrics
        full_density = compute_signal_density(fuzzy_found, structure["full_words"])
        visible_density = compute_signal_density(vis_found, structure["visible_words"])

        rescored = {
            # Identifiers
            "id": test["id"],
            "name": test["name"],
            "category": test["category"],
            "knowledge_sub": test.get("knowledge_sub"),

            # Original metrics (preserved for comparison)
            "original_signal_score": round(orig_score, 4),
            "original_signals_found": orig_found,

            # Enhanced metrics
            "fuzzy_signal_score": fuzzy_score,
            "fuzzy_signals_found": fuzzy_found,
            "match_methods": match_details,
            "signals_expected": signals,

            # Visible-only metrics
            "visible_signal_score": vis_score,
            "visible_signals_found": vis_found,

            # Think-only metrics
            "think_signal_score": think_score,
            "think_signals_found": think_found,

            # Response structure
            "has_think_tag": structure["has_think_tag"],
            "has_scaffolding": structure["has_scaffolding"],
            "full_words": structure["full_words"],
            "visible_words": structure["visible_words"],
            "think_words": structure["think_words"],

            # Signal density
            "full_signal_density": full_density,
            "visible_signal_density": visible_density,

            # Scoring delta
            "fuzzy_vs_original_delta": round(fuzzy_score - orig_score, 4),
        }

        rescored_tests.append(rescored)

    return rescored_tests


def aggregate_scores(rescored_tests):
    """Compute summary statistics from re-scored tests."""

    # Filter to scored questions (those with expected signals)
    scored = [t for t in rescored_tests if t["signals_expected"]]

    # Overall averages
    summary = {
        "n_questions": len(rescored_tests),
        "n_scored": len(scored),
        "avg_original_signal": _mean([t["original_signal_score"] for t in scored]),
        "avg_fuzzy_signal": _mean([t["fuzzy_signal_score"] for t in scored]),
        "avg_visible_signal": _mean([t["visible_signal_score"] for t in scored]),
        "avg_full_words": _mean([t["full_words"] for t in rescored_tests]),
        "avg_visible_words": _mean([t["visible_words"] for t in rescored_tests]),
        "avg_full_density": _mean([t["full_signal_density"] for t in scored]),
        "avg_visible_density": _mean([t["visible_signal_density"] for t in scored]),
        "think_tag_rate": _mean([1 if t["has_think_tag"] else 0 for t in rescored_tests]),
        "scaffolding_rate": _mean([1 if t["has_scaffolding"] else 0 for t in rescored_tests]),
    }

    # By knowledge subcategory
    ks_groups = defaultdict(list)
    for t in scored:
        if t.get("knowledge_sub"):
            ks_groups[t["knowledge_sub"]].append(t)

    summary["by_knowledge_subcategory"] = {}
    for ks in sorted(ks_groups.keys()):
        tests = ks_groups[ks]
        summary["by_knowledge_subcategory"][ks] = {
            "name": KNOWLEDGE_SUBCATEGORIES.get(ks, ks),
            "n_questions": len(tests),
            "avg_original_signal": _mean([t["original_signal_score"] for t in tests]),
            "avg_fuzzy_signal": _mean([t["fuzzy_signal_score"] for t in tests]),
            "avg_visible_signal": _mean([t["visible_signal_score"] for t in tests]),
            "avg_full_density": _mean([t["full_signal_density"] for t in tests]),
            "avg_visible_density": _mean([t["visible_signal_density"] for t in tests]),
            "total_signals_found": sum(len(t["fuzzy_signals_found"]) for t in tests),
            "total_signals_expected": sum(len(t["signals_expected"]) for t in tests),
            "total_words": sum(t["full_words"] for t in tests),
            "total_visible_words": sum(t["visible_words"] for t in tests),
        }

    # By behavioral category
    cat_groups = defaultdict(list)
    for t in scored:
        cat_groups[t["category"]].append(t)

    summary["by_behavioral_category"] = {}
    for cat in sorted(cat_groups.keys()):
        tests = cat_groups[cat]
        summary["by_behavioral_category"][cat] = {
            "n_questions": len(tests),
            "avg_original_signal": _mean([t["original_signal_score"] for t in tests]),
            "avg_fuzzy_signal": _mean([t["fuzzy_signal_score"] for t in tests]),
            "avg_visible_signal": _mean([t["visible_signal_score"] for t in tests]),
            "avg_full_density": _mean([t["full_signal_density"] for t in tests]),
            "avg_visible_density": _mean([t["visible_signal_density"] for t in tests]),
        }

    # Match method distribution
    all_methods = defaultdict(int)
    for t in scored:
        for signal, method in t.get("match_methods", {}).items():
            all_methods[method] += 1
    summary["match_method_distribution"] = dict(all_methods)

    return summary


def _mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else 0


# ── Cross-Scale Analysis ─────────────────────────────────────

def cross_scale_analysis(all_summaries):
    """Build the enhanced cross-scale comparison."""
    analysis = {}

    # 1. Enhanced Knowledge Fidelity Matrix
    analysis["knowledge_fidelity_matrix"] = {}
    for label, summary in all_summaries.items():
        analysis["knowledge_fidelity_matrix"][label] = {}
        for ks, data in summary["by_knowledge_subcategory"].items():
            analysis["knowledge_fidelity_matrix"][label][ks] = {
                "original": data["avg_original_signal"],
                "fuzzy": data["avg_fuzzy_signal"],
                "visible_only": data["avg_visible_signal"],
                "density": data["avg_full_density"],
                "visible_density": data["avg_visible_density"],
            }

    # 2. Signal Density Ratios (ft/base)
    analysis["density_ratios"] = {}
    for scale in ["9b", "4b", "2b", "0.8b"]:
        base_key = f"qwen3.5-{scale}-base"
        ft_key = f"qwen3.5-{scale}-ft"
        if base_key in all_summaries and ft_key in all_summaries:
            base_density = all_summaries[base_key]["avg_full_density"]
            ft_density = all_summaries[ft_key]["avg_full_density"]
            ratio = round(ft_density / base_density, 2) if base_density > 0 else 0

            # Per-subcategory density ratios
            ks_ratios = {}
            for ks in KNOWLEDGE_SUBCATEGORIES:
                bd = all_summaries[base_key]["by_knowledge_subcategory"].get(ks, {}).get("avg_full_density", 0)
                fd = all_summaries[ft_key]["by_knowledge_subcategory"].get(ks, {}).get("avg_full_density", 0)
                ks_ratios[ks] = round(fd / bd, 2) if bd > 0 else 0

            analysis["density_ratios"][scale] = {
                "overall": ratio,
                "by_subcategory": ks_ratios,
            }

    # 3. Enhanced Deltas (fuzzy scores)
    analysis["finetuning_deltas"] = {}
    for scale in ["9b", "4b", "2b", "0.8b"]:
        base_key = f"qwen3.5-{scale}-base"
        ft_key = f"qwen3.5-{scale}-ft"
        if base_key in all_summaries and ft_key in all_summaries:
            deltas = {}
            for ks in KNOWLEDGE_SUBCATEGORIES:
                b = all_summaries[base_key]["by_knowledge_subcategory"].get(ks, {}).get("avg_fuzzy_signal", 0)
                f = all_summaries[ft_key]["by_knowledge_subcategory"].get(ks, {}).get("avg_fuzzy_signal", 0)
                deltas[ks] = round(f - b, 4)
            analysis["finetuning_deltas"][scale] = deltas

    # 4. Degradation thresholds (using fuzzy scores)
    analysis["degradation_thresholds"] = {}
    for ks in KNOWLEDGE_SUBCATEGORIES:
        ref_key = "qwen3.5-9b-ft"
        ref_score = all_summaries.get(ref_key, {}).get(
            "by_knowledge_subcategory", {}
        ).get(ks, {}).get("avg_fuzzy_signal", 0)

        scores = {}
        threshold = None
        for scale in ["9b", "4b", "2b", "0.8b"]:
            ft_key = f"qwen3.5-{scale}-ft"
            s = all_summaries.get(ft_key, {}).get(
                "by_knowledge_subcategory", {}
            ).get(ks, {}).get("avg_fuzzy_signal", 0)
            scores[scale] = s
            if ref_score > 0 and threshold is None:
                drop = (ref_score - s) / ref_score
                if drop > 0.20 and scale != "9b":
                    threshold = scale

        analysis["degradation_thresholds"][ks] = {
            "name": KNOWLEDGE_SUBCATEGORIES[ks],
            "reference_score": ref_score,
            "threshold": threshold,
            "scores": scores,
        }

    # 5. Verbosity analysis
    analysis["verbosity_comparison"] = {}
    for label, summary in all_summaries.items():
        analysis["verbosity_comparison"][label] = {
            "avg_full_words": summary["avg_full_words"],
            "avg_visible_words": summary["avg_visible_words"],
            "think_tag_rate": summary["think_tag_rate"],
            "scaffolding_rate": summary["scaffolding_rate"],
        }

    return analysis


# ── Report Table Generator ───────────────────────────────────

def generate_report_tables(all_summaries, cross_analysis):
    """Generate formatted tables for the Week 3/4 reports."""
    tables = {}

    # Table 1: Cross-scale metrics table
    t1 = {}
    for scale in ["9b", "4b", "2b", "0.8b"]:
        base = all_summaries.get(f"qwen3.5-{scale}-base", {})
        ft = all_summaries.get(f"qwen3.5-{scale}-ft", {})
        t1[scale] = {
            "base_signal_score_original": base.get("avg_original_signal", 0),
            "base_signal_score_fuzzy": base.get("avg_fuzzy_signal", 0),
            "ft_signal_score_original": ft.get("avg_original_signal", 0),
            "ft_signal_score_fuzzy": ft.get("avg_fuzzy_signal", 0),
            "ft_delta_original": round(
                ft.get("avg_original_signal", 0) - base.get("avg_original_signal", 0), 4
            ),
            "ft_delta_fuzzy": round(
                ft.get("avg_fuzzy_signal", 0) - base.get("avg_fuzzy_signal", 0), 4
            ),
            "base_avg_words": base.get("avg_full_words", 0),
            "ft_avg_words": ft.get("avg_visible_words", 0),
            "signal_density_ratio": cross_analysis["density_ratios"].get(
                scale, {}
            ).get("overall", 0),
            "general_knowledge_preserved": "Yes" if ft.get(
                "by_behavioral_category", {}
            ).get("General Knowledge", {}).get("avg_fuzzy_signal", 0) >= 0.67 else "Partial",
        }
    tables["week3_metrics"] = t1

    # Table 2: Knowledge fidelity matrix (fuzzy, ft models only)
    t2 = {}
    for scale in ["9b", "4b", "2b", "0.8b"]:
        ft_key = f"qwen3.5-{scale}-ft"
        t2[scale] = {}
        for ks in sorted(KNOWLEDGE_SUBCATEGORIES.keys()):
            t2[scale][ks] = {
                "name": KNOWLEDGE_SUBCATEGORIES[ks],
                "fuzzy_score": all_summaries.get(ft_key, {}).get(
                    "by_knowledge_subcategory", {}
                ).get(ks, {}).get("avg_fuzzy_signal", 0),
                "density": all_summaries.get(ft_key, {}).get(
                    "by_knowledge_subcategory", {}
                ).get(ks, {}).get("avg_visible_density", 0),
            }
    tables["knowledge_fidelity"] = t2

    return tables


# ── LLM-as-Judge Rubric Generator ────────────────────────────

def generate_judge_rubric():
    """
    Generate the evaluation rubric for Opus 4.6 LLM-as-judge scoring.
    This produces the prompt template and scoring criteria.
    """
    rubric = {
        "description": (
            "LLM-as-judge evaluation rubric for talent intelligence responses. "
            "Each response is scored on 5 dimensions using a 1-5 scale. "
            "The judge receives the question, expected reasoning pattern, "
            "and the model's response."
        ),
        "dimensions": {
            "reasoning_strategy": {
                "name": "Reasoning Strategy",
                "description": "Does the response demonstrate the correct analytical approach?",
                "scoring": {
                    1: "No relevant reasoning; generic or off-topic response",
                    2: "Touches on the topic but uses surface-level or textbook reasoning",
                    3: "Demonstrates relevant analytical thinking but misses key dimensions",
                    4: "Strong analytical reasoning with appropriate frameworks applied",
                    5: "Expert-level reasoning that a senior TI analyst would recognize as their own",
                },
            },
            "analytical_depth": {
                "name": "Analytical Depth",
                "description": "Does it go beyond stating the obvious to provide genuine insight?",
                "scoring": {
                    1: "Superficial; restates the question or gives a generic answer",
                    2: "Identifies the right area but stays at a high level",
                    3: "Provides specific analytical points with some depth",
                    4: "Multi-layered analysis considering second-order effects",
                    5: "Reveals non-obvious insights that reframe the question",
                },
            },
            "epistemic_calibration": {
                "name": "Epistemic Calibration",
                "description": "Does it distinguish what it knows from what it doesn't?",
                "scoring": {
                    1: "Overconfident or uniformly hedged; no calibration",
                    2: "Some hedging but not matched to actual uncertainty",
                    3: "Appropriate uncertainty on most claims",
                    4: "Well-calibrated confidence with explicit reasoning about limits",
                    5: "Expert-level calibration: confident where warranted, uncertain where appropriate, and explicit about the boundary",
                },
            },
            "actionability": {
                "name": "Actionability",
                "description": "Could a stakeholder act on this response?",
                "scoring": {
                    1: "No actionable content; purely academic or abstract",
                    2: "Vaguely directional but no specific next steps",
                    3: "Contains actionable elements but requires interpretation",
                    4: "Clear recommendations with rationale",
                    5: "Immediately actionable with prioritized steps and trade-offs made explicit",
                },
            },
            "delivery_quality": {
                "name": "Delivery Quality",
                "description": "Is the response appropriately structured for its complexity?",
                "scoring": {
                    1: "Incoherent, repetitive, or severely misformatted",
                    2: "Readable but poorly structured or inappropriately long/short",
                    3: "Adequate structure; response length roughly matches complexity",
                    4: "Well-structured with clear organization; leads with key insight",
                    5: "Exemplary delivery: concise, well-organized, insight-first, proportional to complexity",
                },
            },
        },
        "prompt_template": """You are evaluating a talent intelligence model's response quality.

CONTEXT: A model fine-tuned on talent intelligence training data was asked the following question with a talent intelligence system prompt.

QUESTION: {question}

WHAT THIS TESTS: {what_it_tests}

EXPECTED REASONING PATTERNS: The response should demonstrate awareness of these concepts: {expected_signals}

MODEL RESPONSE:
{response}

Score this response on each dimension using the 1-5 scale below. Provide a brief justification (1-2 sentences) for each score, then give the numeric score.

DIMENSIONS:
1. Reasoning Strategy (1-5): Does it demonstrate the correct analytical approach for this type of TI question?
2. Analytical Depth (1-5): Does it go beyond stating the obvious?
3. Epistemic Calibration (1-5): Does it distinguish what it knows from what it doesn't?
4. Actionability (1-5): Could a stakeholder act on this?
5. Delivery Quality (1-5): Is it appropriately structured and proportioned?

Respond in this exact JSON format:
{{
    "reasoning_strategy": {{"justification": "...", "score": N}},
    "analytical_depth": {{"justification": "...", "score": N}},
    "epistemic_calibration": {{"justification": "...", "score": N}},
    "actionability": {{"justification": "...", "score": N}},
    "delivery_quality": {{"justification": "...", "score": N}},
    "overall_impression": "One sentence summary"
}}""",
    }
    return rubric


def build_judge_batch(eval_data, max_questions=None):
    """
    Build batch of prompts ready for LLM-as-judge evaluation.
    Returns list of dicts with prompt and metadata.
    """
    batch = []
    rubric = generate_judge_rubric()

    for test in eval_data["tests"]:
        if not test.get("signals_expected"):
            continue

        # Use visible portion only for judging
        response = test["response"]
        if '</think>' in response:
            response = response.split('</think>', 1)[1].strip()

        prompt = rubric["prompt_template"].format(
            question=test["prompt"],
            what_it_tests=test["what_it_tests"],
            expected_signals=", ".join(test["signals_expected"]),
            response=response,
        )

        batch.append({
            "id": test["id"],
            "name": test["name"],
            "category": test["category"],
            "knowledge_sub": test.get("knowledge_sub"),
            "model_label": eval_data["label"],
            "judge_prompt": prompt,
        })

        if max_questions and len(batch) >= max_questions:
            break

    return batch


# ── Main ─────────────────────────────────────────────────────

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(output_dir, exist_ok=True)

    # Load all eval files
    print("\n  Loading evaluation results...")
    all_evals = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("eval_") and fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            all_evals[data["label"]] = data
            print(f"    {data['label']}: {len(data['tests'])} questions")

    print(f"\n  Loaded {len(all_evals)} models, {sum(len(d['tests']) for d in all_evals.values())} total responses")

    # Re-score all models
    print("\n  Re-scoring with enhanced signal matching...")
    all_rescored = {}
    all_summaries = {}
    for label, eval_data in sorted(all_evals.items()):
        rescored = rescore_model(eval_data)
        summary = aggregate_scores(rescored)
        all_rescored[label] = rescored
        all_summaries[label] = summary

        # Print comparison
        orig = summary["avg_original_signal"]
        fuzzy = summary["avg_fuzzy_signal"]
        delta = fuzzy - orig
        density = summary["avg_full_density"]
        vis_density = summary["avg_visible_density"]
        print(f"    {label:30s}: orig={orig:.3f} fuzzy={fuzzy:.3f} "
              f"(Δ{delta:+.3f}) density={density:.3f} vis_density={vis_density:.3f}")

    # Cross-scale analysis
    print("\n  Running cross-scale analysis...")
    cross = cross_scale_analysis(all_summaries)

    # Generate report tables
    tables = generate_report_tables(all_summaries, cross)

    # Generate LLM-as-judge rubric and sample batch
    print("\n  Generating LLM-as-judge rubric...")
    rubric = generate_judge_rubric()

    # Build judge batch for strategic subset: 9B-ft and 4B-ft (the two top models)
    judge_batches = {}
    for target in ["qwen3.5-9b-ft", "qwen3.5-4b-ft", "qwen3.5-2b-ft"]:
        if target in all_evals:
            batch = build_judge_batch(all_evals[target])
            judge_batches[target] = batch
            print(f"    {target}: {len(batch)} questions ready for judge")

    # ── Save outputs ──
    # 1. Full enhanced analysis
    enhanced = {
        "methodology": {
            "description": "Enhanced re-scoring of existing evaluation responses",
            "improvements": [
                "Fuzzy signal matching: hyphenation, contractions, proximity",
                "Think-tag aware scoring: separate visible vs hidden content",
                "Signal density: signals per 100 words, controls for verbosity",
                "Scaffolding detection: identifies base model reasoning traces",
            ],
            "total_responses_rescored": sum(len(v) for v in all_rescored.values()),
        },
        "all_summaries": all_summaries,
        "cross_scale_analysis": cross,
        "report_tables": tables,
    }
    with open(os.path.join(output_dir, "enhanced_analysis.json"), "w") as f:
        json.dump(enhanced, f, indent=2)
    print(f"\n  Saved: {output_dir}/enhanced_analysis.json")

    # 2. LLM-as-judge materials
    judge_output = {
        "rubric": rubric,
        "batches": {k: v for k, v in judge_batches.items()},
    }
    with open(os.path.join(output_dir, "judge_rubric_and_batch.json"), "w") as f:
        json.dump(judge_output, f, indent=2)
    print(f"  Saved: {output_dir}/judge_rubric_and_batch.json")

    # 3. Per-question rescored details (for deep inspection)
    with open(os.path.join(output_dir, "rescored_details.json"), "w") as f:
        json.dump(all_rescored, f, indent=2)
    print(f"  Saved: {output_dir}/rescored_details.json")

    # ── Print summary report ──
    print(f"\n{'='*70}")
    print(f"  ENHANCED CROSS-SCALE ANALYSIS — SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Model':<28} {'Orig':>6} {'Fuzzy':>6} {'VisSig':>6} {'Density':>8} {'VisDens':>8} {'Words':>6}")
    print(f"  {'─'*70}")
    for label in sorted(all_summaries.keys()):
        s = all_summaries[label]
        print(f"  {label:<28} {s['avg_original_signal']:>5.3f} {s['avg_fuzzy_signal']:>5.3f} "
              f"{s['avg_visible_signal']:>5.3f} {s['avg_full_density']:>7.3f} "
              f"{s['avg_visible_density']:>7.3f} {s['avg_visible_words']:>5.0f}")

    print(f"\n  Signal Density Ratios (ft/base):")
    for scale, data in sorted(cross["density_ratios"].items(), key=lambda x: -float(x[0].replace('b','').replace('.',''))):
        print(f"    {scale}: {data['overall']}×")
        for ks in sorted(data["by_subcategory"].keys()):
            name = KNOWLEDGE_SUBCATEGORIES[ks]
            r = data["by_subcategory"][ks]
            print(f"      {ks}. {name:<45s} {r}×")

    print(f"\n  Degradation Thresholds (fuzzy, >20% drop from 9B-ft):")
    for ks in sorted(cross["degradation_thresholds"].keys()):
        t = cross["degradation_thresholds"][ks]
        scores_str = " → ".join(f"{s}:{v:.3f}" for s, v in t["scores"].items())
        threshold = t["threshold"] or "survives all"
        print(f"    {ks}. {t['name']:<45s} drops at: {threshold:<15s} {scores_str}")

    # Match method breakdown
    print(f"\n  Match Method Distribution (all models combined):")
    total_methods = defaultdict(int)
    for label, summary in all_summaries.items():
        for method, count in summary.get("match_method_distribution", {}).items():
            total_methods[method] += count
    for method, count in sorted(total_methods.items(), key=lambda x: -x[1]):
        print(f"    {method:<25s}: {count}")

    print(f"\n{'='*70}")
    print(f"  Next step: Run judge_rubric_and_batch.json through Opus 4.6")
    print(f"  for qualitative scoring on the 5 rubric dimensions.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
