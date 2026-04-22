# Model Card

## Model Description

LoRA adapters encoding talent intelligence analytical reasoning into the Qwen3.5 model family. Trained on 350 expert-curated behavioral examples from a single domain practitioner using bf16 LoRA fine-tuning.

**Base models:** Qwen/Qwen3.5-9B, Qwen/Qwen3.5-4B, Qwen/Qwen3.5-2B, Qwen/Qwen3.5-0.8B

**Architecture:** Dense hybrid attention (Gated DeltaNet + full softmax, 3:1 ratio). All four sizes share identical architecture, tokenizer, and chat template.

**Training method:** bf16 LoRA (not QLoRA — the Qwen3.5 hybrid attention layers produce NaN loss under 4-bit NF4 quantization)

**Intended use:** Thought partner for talent intelligence analysis. The model should assist analytical reasoning, not replace it.

## Training Data

- 350 examples covering 7 knowledge subcategories across talent intelligence
- Single-analyst training data: encodes one practitioner's analytical priorities and reasoning patterns
- Constructed through AI-assisted generation directed by domain expertise, with manual review and iterative evaluation-driven refinement
- Single normalized system prompt across all examples
- See [`../dataset/README.md`](../dataset/README.md) for format specification and quality criteria

## Evaluation Results

### Recommended Model: Qwen3.5-4B

| Dimension | Score |
|---|---|
| Overall judge score | 3.18 / 5 |
| Maturity level | Level 3 (Operational) per Culshaw (2022) |
| Signal density ratio | 2.4× base model |
| Hardware requirement | 8 GB RAM, CPU-only |
| Inference speed | ~20 tokens/sec on CPU |

### Cross-Scale Summary

| Metric | 9B | 4B | 2B | 0.8B |
|---|---|---|---|---|
| Judge score (1–5) | 3.46 | 3.18 | 2.45 | Below threshold |
| Signal density (FT/Base) | 1.7× | 2.4× | 3.2× | 1.6× |
| General knowledge preserved | 0.88 | 0.88 | 0.71 | 0.71 |

### Strongest and Weakest Subcategories

**Strongest:** Compensation & Benefits — highest cross-scale scores, most consistent ablation performance, highest token agreement (5.9%) in divergence analysis. Structured frameworks with defensible answers produce the most robust encoding.

**Weakest:** Competitive Intelligence — lowest scores at 2B and 0.8B, with "competitor" token demoted 10.3 rank positions in divergence analysis. The model reframes competitive analysis as benchmarking. Correctable with targeted training examples.

## Known Biases and Limitations

### Single-Analyst Bias

The adapter encodes one practitioner's analytical priorities. Vocabulary shift analysis quantified the emphasis distribution:

- Compensation terms: +1.87× amplification
- Labor market terms: +1.54× amplification
- Data quality terms: 0.45× suppression

Users should understand that the model's analytical emphasis reflects this specific practitioner's priorities.

### Behavioral Limitations

- **Diagnostic questioning** transferred successfully at 9B (judge score 4.0/5) but did not consistently generalize across smaller scales
- **Actionability** was the weakest qualitative dimension at 9B (2.82/5) — the model learned to reason like a senior analyst at the cost of less immediately actionable output
- **System prompt switching** instability was observed at 9B: when the inference-time system prompt conflicted with the training prompt, the model sometimes reproduced the training persona. The 4B did not exhibit this behavior.

### What This Model Is Not

- **Not an oracle.** It encodes one analyst's judgment, not ground truth.
- **Not tested on real workforce data.** Evaluation used synthetic stakeholder scenarios, not operational data with messy real-world inputs.
- **Not a replacement for expertise.** It is a thought partner that can structure analytical reasoning, not a substitute for domain knowledge.

## Responsible Use

This model should be used as a reasoning aid that helps users structure their thinking about talent intelligence questions. Specific guidance:

- **Validate against additional sources.** The model's analytical framing should be treated as one perspective, not the definitive analysis.
- **Be aware of emphasis biases.** The training data's emphasis distribution means the model will naturally gravitate toward compensation and labor market framing. Data quality and methodology judgment is underrepresented.
- **Understand the maturity level.** The 4B operates at Culshaw's Level 3 (Operational) — structured analytical reasoning with source evaluation. It does not deliver Level 5 (Transformational) capabilities: relationship-dependent trust, political navigation of senior leadership, or strategic trade-off judgment.
- **Context matters.** The model was trained on talent intelligence scenarios. Responses to questions outside this domain fall back to the base model's general knowledge.

## Technical Specifications

| Parameter | 9B | 4B | 2B | 0.8B |
|---|---|---|---|---|
| LoRA rank | 64 | 64 | 32 | 16 |
| LoRA alpha | 128 | 128 | 64 | 32 |
| Epochs | 3 | 3 | 3 | 3 |
| Precision | bf16 | bf16 | bf16 | bf16 |
| Target modules | q_proj, v_proj | q_proj, v_proj | q_proj, v_proj | q_proj, v_proj |
| Adapter size | ~150 MB | ~100 MB | ~50 MB | ~25 MB |

## References

Culshaw, T. (2022). *Talent intelligence: Use business and people data to drive organizational performance.* Kogan Page.
