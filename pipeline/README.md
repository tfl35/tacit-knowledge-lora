# Pipeline

Cross-scale LoRA fine-tuning and evaluation pipeline for the Qwen3.5 model family (9B, 4B, 2B, 0.8B).

## Requirements

- NVIDIA GPU with CUDA support (developed on DGX Spark, 128 GB unified memory)
- Docker with NVIDIA Container Toolkit
- ~30 GB disk per model size for weights
- Base image: `nvcr.io/nvidia/pytorch:25.12-py3`

See [`../docs/hardware.md`](../docs/hardware.md) for detailed hardware notes including memory constraints and the 27B training attempt.

## Setup

```bash
# Build Docker image and download all four model sizes (~30 min)
bash 0_setup.sh

# Or download specific models only
bash 0_setup.sh --models 4B 9B
```

## Running the Full Pipeline

```bash
# Train + merge + evaluate all four models (~3-4 hours on DGX Spark)
bash run_all.sh

# Specific models only
bash run_all.sh --models 9B 4B

# Skip training (if adapters already exist)
bash run_all.sh --skip-train

# Re-run analysis on existing results
bash run_all.sh --eval-only
```

The pipeline processes models **sequentially** — even the 0.8B model requires most of the available memory during training.

## Running the Ablation Study

```bash
# Train 9B on stratified subsets of 50, 100, 150, 200, 250, 300 examples
bash run_ablation.sh
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train.py` | bf16 LoRA training with model-specific hyperparameters |
| `scripts/merge.py` | Merge LoRA adapter weights into base model |
| `scripts/evaluate.py` | 45-question evaluation suite across 7 knowledge subcategories and 12 behavioral categories |
| `scripts/rescore.py` | Enhanced fuzzy signal matching and visible signal density scoring |
| `scripts/judge.py` | LLM-as-judge evaluation (Claude Sonnet 4.6) across 5 qualitative dimensions |
| `scripts/analyze.py` | Cross-scale knowledge fidelity matrix and summary statistics |
| `scripts/analyze_ablation.py` | Ablation study analysis (signal density, response length, general knowledge by dataset size) |
| `scripts/divergence.py` | Logit-level behavioral divergence analysis (token agreement, entropy delta, domain token rank shifts) |
| `scripts/extract_ti_dataset.py` | Extract talent intelligence examples from combined datasets |
| `scripts/extract_ablation_subsets.py` | Generate stratified ablation subsets from the production dataset |

## Training Configuration

LoRA rank scales with model size to maintain a proportional trainable parameter ratio. All other hyperparameters are held constant to isolate parameter capacity.

| Parameter | 9B | 4B | 2B | 0.8B |
|---|---|---|---|---|
| LoRA rank (r) | 64 | 64 | 32 | 16 |
| LoRA alpha | 128 | 128 | 64 | 32 |
| Epochs | 3 | 3 | 3 | 3 |
| Learning rate | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| Effective batch size | 8 | 8 | 8 | 8 |
| Max sequence length | 4096 | 4096 | 4096 | 4096 |
| Precision | bf16 | bf16 | bf16 | bf16 |
| Target modules | q/k/v/o_proj, gate/up/down_proj | same | same | same |

**Why bf16, not QLoRA?** The Qwen3.5 architecture's hybrid linear attention layers (Gated DeltaNet) produce NaN loss values under 4-bit NF4 quantization. bf16 LoRA is the working configuration for this model family.

## Evaluation Framework

The 45-question evaluation suite maps to 7 knowledge subcategories derived from Culshaw (2022) and 12 behavioral categories. Each model is evaluated in both base and fine-tuned states.

**Eight evaluation methods:**

1. **Signal-based scoring** with enhanced fuzzy matching
2. **Visible signal density** (signals per 100 visible words) — normalizes for verbosity
3. **LLM-as-judge** scoring across 5 qualitative dimensions (reasoning strategy, analytical depth, epistemic calibration, actionability, delivery quality)
4. **Vocabulary shift analysis** — measures frequency changes in analytical vs. domain vocabulary
5. **Brevity calibration** — response length distribution analysis
6. **Culshaw maturity mapping** — maps outputs to the 5-level TI maturity model
7. **Dataset size ablation** — trains on 50–300 example subsets
8. **Behavioral divergence** — logit-level token agreement, entropy delta, domain token rank shifts

## Output Structure

```
output/
├── qwen3.5-9b/run_XXXX/       # LoRA adapter checkpoints
├── qwen3.5-4b/run_XXXX/
├── qwen3.5-2b/run_XXXX/
└── qwen3.5-0.8b/run_XXXX/

merged_qwen3.5-{9b,4b,2b,0.8b}/  # Merged full models

results/
├── eval_qwen3.5-{size}-{base,ft}.json   # Per-model evaluation
├── cross_scale_analysis.json              # Knowledge fidelity matrix
├── ablation_analysis.json                 # Dataset size study
└── divergence_9b_base_vs_9b_ft.json      # Token-level analysis
```

## GGUF Conversion and Deployment

After training and merging, convert your models to GGUF format for local deployment with Ollama or llama.cpp. The conversion kit is in [`gguf/`](gguf/):

```bash
cd gguf/

# Edit run_docker.sh with your merged model paths, then:
./run_docker.sh

# Inside the container:
bash /workspace/kit/setup_env.sh              # one-time setup
python /workspace/kit/convert_to_gguf.py \
    /workspace/models/qwen35-4b \
    --output-dir /workspace/output \
    --model-name ti-analyst-4b
```

The converter runs pre-flight validation, attempts Unsloth conversion (with llama.cpp fallback), generates an Ollama Modelfile, and validates the output. See [`gguf/README.md`](gguf/README.md) for full documentation.
