# Tools: Domain-Agnostic Adapter Toolkit

Standalone tools for building, auditing, and evaluating domain expertise adapters in any field. These tools encode the methodology discovered during the talent intelligence project into reusable code.

The training pipeline (`../pipeline/`) handles the mechanics of fine-tuning. These tools handle the harder questions: *Is your training data structured well? How do you know if the adapter actually works?*

## Workflow

```
1. Define your domain          →  domain_config.yaml
2. Curate training examples    →  your_dataset.json (ShareGPT format)
3. Audit your dataset          →  python dataset_audit.py your_dataset.json
4. Train your adapter          →  ../pipeline/run_all.sh
5. Evaluate your adapter       →  python eval_runner.py --config domain_config.yaml
6. Iterate (steps 2-5)         →  target weak subcategories, retrain
```

## Quick Start

```bash
cd tools/

# 1. Copy and edit the domain config template for your field
cp domain_config.yaml my_domain.yaml
# Edit: system prompt, subcategories, eval questions, signals

# 2. Audit your training dataset before training
python dataset_audit.py ../dataset/ti_sample_format.json
python dataset_audit.py your_dataset.json --config my_domain.yaml

# 3. After training, evaluate your adapter
python eval_runner.py --config my_domain.yaml --ollama my-adapter-4b --label ft-4b

# 4. Compare against the base model
python eval_runner.py --config my_domain.yaml --ollama qwen3.5:4b --label base-4b

# 5. Optionally add LLM-as-judge evaluation
export ANTHROPIC_API_KEY="your-key"
python eval_runner.py --config my_domain.yaml --ollama my-adapter-4b --judge --label ft-4b-judged
```

## Tool Reference

### `domain_config.yaml`

The template that defines everything domain-specific about your adapter. You fill in:

- **Domain identity** — name, description, reference framework
- **System prompt** — must exactly match your training data's prompt
- **Knowledge subcategories** — the 3-7 expertise areas your adapter should encode (each becomes a row in your knowledge fidelity matrix)
- **Behavioral categories** — the analytical behaviors you expect (diagnostic questioning, premise challenge, epistemic calibration, etc.)
- **Evaluation questions** — 15-20+ questions with signal definitions and anti-signals
- **Audit thresholds** — structural distribution targets for your training data

The template ships with an acupuncture/TCM example to show the pattern. Replace it with your domain.

### `dataset_audit.py`

Analyzes your training dataset's structural distribution and flags issues that empirically degrade fine-tuning quality. Every check in this script corresponds to a lesson learned during iterative training-evaluation cycles.

```bash
# Basic audit with default thresholds
python dataset_audit.py your_dataset.json

# Audit with domain-specific thresholds
python dataset_audit.py your_dataset.json --config my_domain.yaml
```

**What it checks:**

| Check | What It Catches | Why It Matters |
|---|---|---|
| Dataset size | Too few (<100) or too many (>300) examples | Below 100: weak calibration. Above 300: general knowledge erosion. |
| System prompt consistency | Multiple or missing system prompts | Multiple prompts cause catastrophic interference. |
| Response length distribution | Clustering in a narrow word-count range | Model learns the cluster as default length for all questions. |
| Structural mix | Insufficient multi-turn examples | Below 25%: diagnostic questioning behavior won't transfer. |
| Lead-with-insight ratio | Too many context-first responses | Model buries the actionable insight at the end. |
| Diagnostic questioning | Insufficient scoping question examples | This behavior contradicts base model tendencies and needs concentrated representation. |

**Requirements:** Python 3.8+. Optional: `pip install pyyaml` for config file support.

### `eval_runner.py`

Runs your evaluation questions against a model and produces signal scores, density metrics, and an optional LLM-as-judge assessment. Outputs a knowledge fidelity matrix showing per-subcategory performance.

```bash
# Via Ollama (simplest)
python eval_runner.py --config my_domain.yaml --ollama my-model --label my-eval

# Via Hugging Face model
python eval_runner.py --config my_domain.yaml --model /path/to/model --label my-eval

# With LLM-as-judge
export ANTHROPIC_API_KEY="your-key"
python eval_runner.py --config my_domain.yaml --ollama my-model --judge --label my-eval
```

**Three evaluation methods:**

1. **Signal scoring** — checks for expected domain vocabulary in responses using fuzzy matching (hyphenation variants, substring matching). Produces a 0-1 score per question.

2. **Visible signal density** — signals per 100 visible words. Normalizes for verbosity so that a concise expert response isn't penalized relative to a verbose generic one. This is the metric that revealed the compression-not-degradation finding.

3. **LLM-as-judge** (optional) — sends each response to Claude for qualitative scoring across 5 dimensions: reasoning strategy, analytical depth, epistemic calibration, actionability, and delivery quality. This is the method that reversed the diagnostic questioning finding. Requires `ANTHROPIC_API_KEY`.

**Output:**
- `results/eval_{label}.json` — full results with every response and score
- `results/summary_{label}.json` — summary with knowledge fidelity matrix

**Requirements:** Python 3.8+, `pyyaml`. For Ollama: [Ollama](https://ollama.ai) installed. For HF models: `transformers`, `torch`. For judge: `anthropic` package + API key.

## Writing Good Evaluation Questions

The evaluation questions are the most important part of the config. They determine whether you can tell if your adapter actually works. Guidelines from the research:

**Coverage.** Write at least 2-3 questions per knowledge subcategory. If a subcategory has no evaluation questions, you can't measure whether it transferred.

**Ambiguity.** Include at least 2-3 deliberately ambiguous questions that require scoping before answering. These test diagnostic questioning — the hardest behavior to transfer.

**Premise challenge.** Include at least 1-2 questions with flawed premises. A good domain adapter should push back rather than accept the premise.

**Signal selection.** Signals should be domain-specific terms that a base model would not use without fine-tuning. If a signal would appear in a base model response, it doesn't measure transfer. Anti-signals are terms that indicate a generic or evasive response.

**General knowledge.** Include 1-2 non-domain questions (e.g., "What causes a rainbow?") to test for catastrophic forgetting.

**Expect your evaluation to mislead you.** The central methodological finding of this research is that automated signal scoring becomes less reliable as knowledge transfer deepens (correlation with qualitative assessment dropped to r = .08 at the deepest transfer level). Build multiple evaluation methods and investigate disagreements between them — the disagreement is often the most informative finding.

## Adapting for a Hackathon

These tools are designed to support a 2-day hackathon workflow:

**Day 1:** Participants define their domain config (subcategories, system prompt), curate 30-50 training examples, run `dataset_audit.py` to check structural distribution, and submit for training.

**Day 2:** Participants write 10-15 evaluation questions, run `eval_runner.py` against their adapter and the base model, compare results, write targeted additional examples to address gaps, and retrain.

The config template, audit script, and eval runner provide enough scaffolding that participants focus on the hard part — curating their domain expertise — rather than building evaluation infrastructure from scratch.
