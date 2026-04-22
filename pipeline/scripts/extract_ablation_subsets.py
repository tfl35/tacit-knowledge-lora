#!/usr/bin/env python3
"""
Ablation Study Subset Extraction
==================================
Creates stratified subsets of the 350-example production dataset at:
  50, 100, 150, 200, 250, 300 examples

Stratification ensures each subset preserves the approximate distribution
of single-turn vs multi-turn examples and topic diversity from the full set.
The 350-example results already exist, so we don't re-extract that.

Each subset is a proper subset of the next larger one:
  50 ⊂ 100 ⊂ 150 ⊂ 200 ⊂ 250 ⊂ 300 ⊂ 350

This nested design means improvements from 50→100 reflect the additional
50 examples, not random variation from different subsets.

Usage:
    python extract_ablation_subsets.py /path/to/ti_350_production.json /output/dir/
"""

import json
import os
import sys
import random
from collections import defaultdict


def classify_example(item):
    """
    Classify an example into one of 7 knowledge subcategories based on
    content analysis of both user and assistant text.
    Returns the primary category or 'mixed' if no clear match.
    """
    all_text = ' '.join(c['value'].lower() for c in item['conversations'])

    # Priority-ordered keyword sets (check most specific first)
    classifiers = [
        ('G_operational', [
            'dashboard', 'always-on', 'cadence', 'monitoring', 'intelligence product',
            'battlecard', 'field alert', 'quarterly brief', 'weekly report',
            'tracking', 'real-time', 'alerting', 'delivery mechanism', 'intelligence feed'
        ]),
        ('B_compensation', [
            'compensation', 'salary', 'pay band', 'total comp', 'equity', 'benefits',
            'percentile', 'comp data', 'base pay', 'bonus', 'vesting', 'stock option',
            'retention package', 'pay equity', 'market rate', 'comp bench'
        ]),
        ('C_competitive', [
            'competitor', 'competitive intelligence', 'poaching', 'talent war',
            'competitive landscape', 'market position', 'competitor analysis',
            'm&a', 'acquisition', 'due diligence', 'referral bonus'
        ]),
        ('D_data_quality', [
            'data source', 'methodology', 'sample size', 'benchmark',
            'reliability', 'confidence', 'survey', 'vendor report', 'linkedin data',
            'data quality', 'self-reported', 'bias', 'cross-reference', 'validate',
            'wef report', 'bls data', 'o*net'
        ]),
        ('F_stakeholder', [
            'stakeholder', 'board presentation', 'ceo', 'chro', 'cfo',
            'executive', 'business case', 'roi', 'present to', 'deliver to',
            'hiring manager', 'leadership', 'convince', 'persuade', 'frame'
        ]),
        ('E_strategic', [
            'workforce planning', 'scenario', 'forecast', 'future of',
            'succession', 'skills gap', 'strategic', 'long-term', 'five-year',
            'future-proof', 'reskill', 'upskill', 'transformation'
        ]),
        ('A_labor_market', [
            'labor market', 'job postings', 'hiring trend', 'talent pool',
            'talent supply', 'employment', 'job market', 'attrition', 'turnover',
            'recruiting', 'sourcing', 'pipeline', 'time-to-fill', 'remote',
            'location', 'offshore', 'nearshore', 'hub', 'geo', 'city'
        ]),
    ]

    for cat, keywords in classifiers:
        score = sum(1 for kw in keywords if kw in all_text)
        if score >= 2:
            return cat

    # Fallback: any single keyword match
    for cat, keywords in classifiers:
        if any(kw in all_text for kw in keywords):
            return cat

    return 'mixed'


def is_multi_turn(item):
    """Check if example has multiple assistant responses."""
    return sum(1 for c in item['conversations'] if c['from'] == 'gpt') > 1


def create_nested_subsets(data, sizes=[50, 100, 150, 200, 250, 300]):
    """
    Create nested stratified subsets: 50 ⊂ 100 ⊂ 150 ⊂ 200 ⊂ 250 ⊂ 300.
    Each larger subset contains all examples from smaller ones plus new additions.
    """
    random.seed(42)  # Reproducible

    # Classify all examples
    classified = defaultdict(list)
    for i, item in enumerate(data):
        cat = classify_example(item)
        classified[cat].append(i)

    print(f"\n  Dataset classification:")
    for cat in sorted(classified.keys()):
        mt = sum(1 for i in classified[cat] if is_multi_turn(data[i]))
        st = len(classified[cat]) - mt
        print(f"    {cat}: {len(classified[cat])} ({st} single, {mt} multi)")

    # Shuffle within each category
    for cat in classified:
        random.shuffle(classified[cat])

    # Build the largest subset first (250), then take nested subsets
    # Proportional allocation: each category gets its share of the target size
    subsets = {}
    all_selected = []

    for target_size in sorted(sizes):
        # How many from each category for this target?
        total_available = sum(len(v) for v in classified.values())
        needed = target_size - len(all_selected)

        # Calculate proportional allocation for new additions
        remaining_by_cat = {}
        for cat, indices in classified.items():
            already_in = len([i for i in indices if i in set(all_selected)])
            remaining_by_cat[cat] = [i for i in indices if i not in set(all_selected)]

        # Distribute needed examples proportionally
        if needed > 0:
            total_remaining = sum(len(v) for v in remaining_by_cat.values())
            new_additions = []

            for cat in sorted(remaining_by_cat.keys()):
                avail = remaining_by_cat[cat]
                if total_remaining > 0:
                    allocation = max(1, round(needed * len(avail) / total_remaining))
                    allocation = min(allocation, len(avail))
                    new_additions.extend(avail[:allocation])

            # If we're short or over, adjust
            if len(new_additions) < needed:
                # Add more from largest remaining pools
                for cat in sorted(remaining_by_cat.keys(), key=lambda c: -len(remaining_by_cat[c])):
                    for idx in remaining_by_cat[cat]:
                        if idx not in set(all_selected) and idx not in set(new_additions):
                            new_additions.append(idx)
                            if len(new_additions) >= needed:
                                break
                    if len(new_additions) >= needed:
                        break
            elif len(new_additions) > needed:
                random.shuffle(new_additions)
                new_additions = new_additions[:needed]

            all_selected.extend(new_additions)

        subsets[target_size] = list(all_selected)

    return subsets


def validate_subsets(data, subsets):
    """Validate that subsets are properly nested and stratified."""
    print(f"\n  Subset validation:")
    prev_set = set()
    for size in sorted(subsets.keys()):
        indices = set(subsets[size])
        assert len(indices) == size, f"Size mismatch: expected {size}, got {len(indices)}"
        assert prev_set.issubset(indices), f"Not nested: {len(prev_set)} subset not contained in {size}"

        mt = sum(1 for i in indices if is_multi_turn(data[i]))
        st = size - mt

        # Category distribution
        cats = defaultdict(int)
        for i in indices:
            cats[classify_example(data[i])] += 1

        cat_str = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in sorted(cats.items()))
        print(f"    {size}: {st} single + {mt} multi | {cat_str}")

        prev_set = indices


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ablation_subsets.py <dataset.json> [output_dir]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Loading: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)
    print(f"  Total examples: {len(data)}")

    # Create nested subsets
    sizes = [50, 100, 150, 200, 250, 300]
    subsets = create_nested_subsets(data, sizes)

    # Validate
    validate_subsets(data, subsets)

    # Write subset files
    print(f"\n  Writing subset files to {output_dir}/:")
    for size, indices in sorted(subsets.items()):
        subset_data = [data[i] for i in sorted(indices)]
        outpath = os.path.join(output_dir, f"ti_{size}_ablation.json")
        with open(outpath, 'w') as f:
            json.dump(subset_data, f, indent=2, ensure_ascii=False)

        words = sum(
            len(c['value'].split())
            for item in subset_data
            for c in item['conversations']
            if c['from'] == 'gpt'
        )
        print(f"    ti_{size}_ablation.json: {len(subset_data)} examples, ~{words} words")

    # Also write the index map for reproducibility
    index_map = {str(size): sorted(indices) for size, indices in subsets.items()}
    with open(os.path.join(output_dir, "ablation_index_map.json"), 'w') as f:
        json.dump(index_map, f, indent=2)
    print(f"    ablation_index_map.json: index mapping for reproducibility")

    print(f"\n  Done. Run ablation with:")
    for size in sizes:
        print(f"    bash run_all.sh --models 9B --dataset dataset/ti_{size}_ablation.json --skip-base-eval")
    print()


if __name__ == "__main__":
    main()
