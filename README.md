# Encoding Tacit Knowledge into Open-Weight Language Models via LoRa

A cross-scale study of encoding domain expertise into downloadable model weights through parameter-efficient fine-tuning, and what it reveals about how expertise actually works.

---

## The Finding That Changed the Project

The model that most deeply internalized domain expertise produced the shortest responses, scored zero on the keyword evaluation built to measure it, and was rated 4.6/5 by an independent qualitative judge. It had learned to reason rather than recite, and the evaluation framework designed to detect expertise was systematically penalizing mastery. 

That inversion (expertise producing *fewer* words, not more, and becoming *invisible* to surface-level measurement as it deepens) is the central finding of this project. It has implications for how organizations measure capability, how knowledge management theory models the limits of expertise transfer, and how practitioners should approach fine-tuning for domain reasoning.

## The Problem

Talent intelligence, the discipline of interpreting labor market signals to inform workforce strategy, depends on analytical judgment that no dataset captures. An experienced analyst evaluates compensation survey methodology before trusting it, challenges a stakeholder's premise before analyzing their data, and synthesizes noisy signals into calibrated recommendations. That expertise is concentrated in large enterprises with dedicated teams. Community workforce programs, small nonprofits, and local governments making labor market investment decisions operate without any of it. 

This project tests whether parameter-efficient fine-tuning can encode that analytical reasoning into open-weight model weights, and if so, which dimensions of expertise survive compression to smaller models.

## What This Project Does

Fine-tunes the [Qwen3.5](https://huggingface.co/Qwen) model family across four parameter scales (9B, 4B, 2B, 0.8B) on 350 expert-curated talent intelligence examples using bf16 LoRA. All four models share the same hybrid attention architecture ([Gated DeltaNet](https://arxiv.org/abs/2412.06464)), tokenizer, and chat template, isolating parameter capacity as the single independent variable. Evaluates using eight complementary methods including a dataset size ablation study and logit-level behavioral divergence analysis. The full pipeline runs in **~6 GPU-hours** on an NVIDIA DGX Spark. 

The training data encodes one analyst's reasoning patterns: diagnostic questioning, source evaluation, premise challenging, and calibrated recommendations across seven knowledge subcategories validated against Culshaw's (2022) talent intelligence frameworks.

## Key Findings

### Compression, Not Degradation

Fine-tuning compressed domain reasoning by 1.6× to 3.2× across all scales. Average response length dropped from 1,227 words (base) to 272 words (fine-tuned) while information density (domain signals per 100 words) increased throughout. The model stopped announcing that it was analyzing and started doing it.

| Metric | 9B | 4B | 2B | 0.8B |
|---|---|---|---|---|
| Signal density (sig/100 words) | 1.02 | 1.35 | 1.03 | 0.48 |
| Density ratio (FT/Base) | 1.7× | 2.4× | 3.2× | 1.6× |
| Avg words (Base → FT) | 1,227 → 272 | 1,184 → 253 | 766 → 249 | 714 → 395 |
| LLM-as-judge score (1–5) | 3.46 | 3.18 | 2.45 | — |
| Signal–judge correlation | .08 | .44 | .37 | — |

The 4B model is recommended for deployment: **Level 3 talent intelligence reasoning on any laptop with 8 GB RAM**, free, offline, no API required.

### Different Expertise Dimensions Have Different Scale Thresholds

Not all knowledge compresses equally. The knowledge fidelity matrix reveals that stakeholder communication survives to 0.8B with *increasing* performance. The most interpersonal dimension is the most scale-invariant. Compensation reasoning requires a minimum 4B. Data quality judgment degrades already at 4B.

| Subcategory | 9B | 4B | 2B | 0.8B | Degrades at |
|---|---|---|---|---|---|
| A. Labor Market Interpretation | 0.167 | 0.320 | 0.193 | 0.147 | Survives 0.8B |
| B. Compensation & Benefits | 0.405 | 0.452 | 0.143 | 0.198 | 2B |
| C. Competitive Intelligence | 0.202 | 0.171 | 0.146 | 0.098 | 2B |
| D. Data Quality & Methodology | 0.260 | 0.146 | 0.140 | 0.153 | 4B |
| E. Strategic Workforce Planning | 0.307 | 0.337 | 0.190 | 0.075 | 2B |
| F. Stakeholder Communication | 0.227 | 0.216 | 0.223 | 0.249 | Survives 0.8B ↑ |
| G. Operational Intel Design | 0.396 | 0.216 | 0.317 | 0.136 | 4B |

These thresholds provide practical guidance for adapter deployment decisions, though they are measured via proxy metrics (signal scoring) and should be validated against task-specific evaluation in each deployment context.

### The Diagnostic Questioning Reversal

Nonaka and Takeuchi (1995) predicted that "negative knowledge" (knowing what to ask before answering, what to ignore, when to push back) would resist externalization. Signal-based scoring confirmed this at every scale: the behavior appeared to fail. The LLM-as-judge reversed the finding. The 9B scored **4.0/5** on diagnostic questioning, with reasoning strategy at 4.75 and epistemic calibration at 4.75. The behavior had encoded so deeply it left no vocabulary trace. The model learned *when* to ask, not which keywords to use. 

This is a single result in a single domain at a single scale, not a general overthrow of the theory. But it suggests that some dimensions of tacit expertise may transfer more readily through behavioral demonstration than through structured articulation, and that the ceiling on externalization may depend on the medium more than the field has assumed.

### The 125-Example Sweet Spot

The ablation study (50–300 examples on the 9B) established that **100–150 well-curated behavioral examples** is sufficient for meaningful domain transfer with full general knowledge preservation. Below 100, encoding is present but less calibrated. Above 250, general knowledge begins to erode.

| Metric | 50 ex | 100 ex | 150 ex | 200 ex | 250 ex | 300 ex |
|---|---|---|---|---|---|---|
| Signal density | 0.46 | 0.57 | 0.58 | 0.66 | 0.68 | 0.78 |
| Avg words | 853 | 620 | 470 | 356 | 298 | 243 |
| General knowledge preserved | 0.875 | 1.0 | 1.0 | 1.0 | 0.50 | 0.835 |

One domain expert, two to three days of focused authoring, produces a functional adapter. That is the unit economics finding that makes this approach practical.

### Token-Level Evidence

Behavioral divergence analysis confirmed near-complete restructuring of generative behavior: **2.1% overall token agreement** between base and fine-tuned models. The fine-tuned model is *less certain* than the base (entropy delta +0.143), not because it knows less, but because it learned when to hedge. This is mechanistically consistent with the judge's high epistemic calibration scores.

### What This Has Not Yet Tested

The evaluation measures whether the model *reasons like* an expert: whether its analytical framing, source skepticism, and epistemic calibration match expert patterns. It does not yet measure whether following the model's recommendations improves actual workforce decisions. Downstream task evaluation with real labor market data, messy inputs, and measurable outcomes is the critical next step. The model should be understood as a proof of concept with validated analytical behavior, not a validated decision-support system.

## Replicating This in Your Own Domain

The methodology is domain-agnostic. The talent intelligence application was the first test case; the process generalizes to any field where practitioner judgment is the valuable layer above accessible data.

The critical insight from eight weeks of iteration: **the bottleneck is example quality, not technical complexity**. The training pipeline is learnable. The hard part is recognizing what you do that creates value and demonstrating it clearly enough for a pattern learner to absorb.

What matters in the training examples:
- **Behavioral demonstrations, not knowledge descriptions.** The model doesn't need you to explain your reasoning. It needs to see you do it, across enough contexts, with enough variety.
- **A single normalized system prompt.** Competing prompt formats in training data cause catastrophic interference. One prompt. No exceptions.
- **Structural variety.** If 90% of your responses build context before delivering the insight, the model will bury the lead on every question. Vary response lengths, vary whether you lead with the recommendation or the diagnostic question, vary the depth.
- **At least two evaluation methods that disagree by construction.** A single evaluation method will mislead you, and you will not know it until a second method contradicts it.

See [`docs/practitioners_guide.md`](docs/practitioners_guide.md) for the full eight-lesson methodology distilled from this project's experimental findings.

**A note on the pipeline:** The training scripts (`train.py`, `merge.py`) and GGUF conversion kit are domain-agnostic. Swap in your dataset and they work. The evaluation suite (`evaluate.py`) is talent-intelligence-specific: its 45 questions, signal definitions, and behavioral categories encode what "good" looks like in this domain. Adapting to a new domain requires writing your own evaluation questions and defining your own quality signals. That evaluation design work is part of the replication, not an afterthought.

## Trained Models

Trained adapters in GGUF format are available on HuggingFace:

| Model | Examples | Description | Link |
|---|---|---|---|
| TI-Analyst-9B | 350 | Full power: highest reasoning quality, requires GPU | [HuggingFace](https://huggingface.co/tfl35/ti-analyst-9b) |
| **TI-Analyst-4B-150** | 150 | Recommended: sweet spot dataset size, Level 3 TI reasoning on 8 GB RAM | [HuggingFace](https://huggingface.co/tfl35/ti-analyst-4b-150) *model being added*|

## Repository Structure

```
tacit-knowledge-lora/
├── pipeline/               # Reproducible training and evaluation code
│   ├── 0_setup.sh          # Docker image build + model download
│   ├── run_all.sh          # Full cross-scale pipeline (train → merge → eval × 4)
│   ├── run_ablation.sh     # Dataset size ablation study
│   ├── scripts/            # Training, evaluation, analysis scripts
│   └── gguf/               # GGUF conversion + Ollama deployment kit
├── tools/                  # Domain-agnostic adapter toolkit
│   ├── domain_config.yaml  # Template: define your domain's eval structure
│   ├── dataset_audit.py    # Audit training data structural distribution
│   └── eval_runner.py      # Run multi-method evaluation from config
├── dataset/                # Full training dataset + documentation
│   ├── ti_350_production.json
│   └── README.md           # Format spec, quality criteria, construction methodology
├── results/                # Cross-scale analysis, ablation, divergence summaries
├── demo/                   # Screenshare video comparing base and fine-tuned 9b models
├── research/               # Full academic report
│   └── final_report.md
└── docs/
    ├── practitioners_guide.md   # 8 lessons for domain adaptation
    ├── model_card.md            # Training provenance + responsible use
    └── hardware.md              # DGX Spark notes + memory constraints
```

The `pipeline/` directory contains the talent-intelligence-specific code used in the research. The `tools/` directory contains domain-agnostic versions of the evaluation methodology. Start here if you're adapting this for your own domain. See [`tools/README.md`](tools/README.md) for the full workflow.

The full 350-example training dataset is included in the [`dataset/`](dataset/) directory along with format documentation and quality criteria.

## Quick Start

### Using the trained models (no training required)

Download from HuggingFace and run locally with [Ollama](https://ollama.ai):

```bash
# Download and run the recommended 4B model (150 examples, sweet spot)
ollama run tfl35/ti-analyst-4b-150

# Example query
>>> Our vendor says 85% of companies are struggling to hire AI talent. Should we be worried?
```

### Reproducing the training pipeline

```bash
# Clone
git clone https://github.com/tfl35/tacit-knowledge-lora.git
cd tacit-knowledge-lora/pipeline

# Setup (builds Docker image, downloads models)
bash 0_setup.sh --models 4B

# Run training + evaluation (dataset included at ../dataset/ti_350_production.json)
bash run_all.sh --models 4B
```

**Requirements:** NVIDIA GPU with CUDA support, Docker, ~30 GB disk per model. Developed on NVIDIA DGX Spark (128 GB unified memory). See [`docs/hardware.md`](docs/hardware.md) for details. See [`pipeline/README.md`](pipeline/README.md) for full pipeline documentation.

## Future Work

This project is the first test case in a broader research program. Replication in acupuncture clinical reasoning is underway with a collaborating domain practitioner, testing whether the encoding process generalizes when the researcher is curating another expert's knowledge rather than their own. The 27B model scale remains untested (predicted: lower entropy delta reflecting more confident internalization, higher actionability). A dedicated compensation adapter would test whether the most robustly transferable subdomain performs even better as a focused training objective.

## Citation

If you use this work, please cite:

```bibtex
@misc{tacit-knowledge-lora-2026,
  title={Encoding Tacit Knowledge into Open-Weight Language Models via LoRA: A Cross-Scale Knowledge Fidelity Analysis},
  author={Cameron Lowry},
  year={2026},
  url={https://github.com/tfl35/tacit-knowledge-lora}
}
```

## Acknowledgments

This project was completed as the capstone for an M.S. in Artificial Intelligence. The fine-tuning pipeline was developed following the [ARM PyTorch Fine-Tuning on Spark](https://learn.arm.com/learning-paths/laptops-and-desktops/pytorch-finetuning-on-spark/) learning path. The talent intelligence evaluation framework draws on Culshaw's (2022) *Talent Intelligence* maturity model and competency frameworks. The theoretical framing builds on Nonaka and Takeuchi's (1995) SECI model of knowledge creation. The behavioral divergence methodology adapts the framework introduced by Zhu et al. (2025).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details. The Qwen3.5 models used in this project are also Apache 2.0 licensed.
