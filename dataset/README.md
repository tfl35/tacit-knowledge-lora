# Dataset

## Overview

The production training dataset (`ti_350_production.json`) contains 350 expert-curated talent intelligence examples: 242 single-turn and 108 multi-turn conversations totaling approximately 185,000 words.

## Format Specification

Training data uses the **ShareGPT format** — a JSON array of conversation objects, each containing a `conversations` array of turns:

```json
[
  {
    "conversations": [
      {"from": "system", "value": "System prompt..."},
      {"from": "human", "value": "User question..."},
      {"from": "gpt",   "value": "Model response..."}
    ]
  }
]
```

### Role Mapping

| ShareGPT role | Chat role | Notes |
|---|---|---|
| `system` | system | Identical across all examples — one prompt, no exceptions |
| `human` | user | Stakeholder questions, analytical scenarios |
| `gpt` | assistant | Expert responses demonstrating analytical reasoning |

### Multi-Turn Format

Multi-turn examples alternate `human` and `gpt` turns after the system prompt. These are used primarily for diagnostic questioning sequences where the expert asks clarifying questions before providing analysis:

```json
{
  "conversations": [
    {"from": "system", "value": "..."},
    {"from": "human", "value": "Initial stakeholder question"},
    {"from": "gpt",   "value": "Diagnostic questions + framing"},
    {"from": "human", "value": "Stakeholder provides clarification"},
    {"from": "gpt",   "value": "Targeted analysis based on clarified scope"}
  ]
}
```

## Quality Criteria

### What Makes a Good Training Example

1. **Behavioral demonstration, not knowledge description.** The model needs to see the expert *do* the reasoning — evaluate a source, challenge a premise, calibrate a recommendation — not explain that they do it.

2. **One normalized system prompt.** Every example uses the identical system prompt. Competing prompt formats in training data cause catastrophic interference: the model produces template fragments rather than analytical reasoning.

3. **Structural variety.** Response lengths vary from 42 to 433 words. Some responses lead with the recommendation. Others lead with diagnostic questions. Others build context before delivering the insight. If most responses follow the same structure, the model learns that structure as a fixed template.

4. **Calibrated confidence.** Responses acknowledge uncertainty where it exists. "I don't know" and "this data source has limitations" appear where warranted. The expert demonstrates epistemic humility, not performative confidence.

## Knowledge Coverage

Examples span seven knowledge subcategories mapped to Culshaw's (2022) talent intelligence frameworks:

| Code | Subcategory | Description |
|---|---|---|
| A | Labor Market Interpretation | Evaluating market signals, supply-demand dynamics |
| B | Compensation & Benefits Analysis | Survey methodology, benchmarking, pay equity |
| C | Competitive Intelligence | Detecting strategic moves from hiring patterns |
| D | Data Quality & Methodological Judgment | Source evaluation, survey design critique |
| E | Strategic Workforce Planning | Build/buy/borrow decisions, scenario planning |
| F | Stakeholder Communication & Commercial Acumen | Framing analysis for decision-makers |
| G | Operational Intelligence Design | Building repeatable TI processes and products |

## Behavioral Categories

Examples are tagged with 12 behavioral categories:

1. Premise Challenge
2. Data Source Skepticism
3. Multi-Signal Synthesis
4. Live Signal Interpretation
5. Location Feasibility
6. Stakeholder Management
7. Always-On Intelligence
8. Job Description Analysis
9. Forward-Looking/Scenario
10. Commercial Translation
11. Diagnostic Questioning
12. Epistemic Calibration

## Construction Methodology

The dataset was constructed through a three-phase process:

1. **Phase 1 — AI-Assisted Generation with Expert Direction** (150 examples): The researcher directed generation by specifying analytical trajectories, stakeholder personas, and reasoning patterns drawn from professional talent intelligence experience. A three-pass revision process ensured quality.

2. **Phase 2 — Framework-Driven Gap Filling** (102 examples): Deep analysis of Culshaw (2022) generated additional examples across eight thematic batches addressing gaps in analytical frameworks and operationalized intelligence scenarios.

3. **Phase 3 — Evaluation-Driven Refinement** (98 examples): Iterative training-evaluation cycles identified encoding gaps and structural imbalances. Key finding: knowledge encoding and output formatting are separable concerns — fine-tuning encodes *what* and *how* the model reasons; inference-time system prompting controls presentation format.

## Format Sample

See [`ti_350_production.json`](ti_350_production.json) for the full training dataset.
