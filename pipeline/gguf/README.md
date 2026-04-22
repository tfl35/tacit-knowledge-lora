# Qwen3.5 GGUF Conversion Kit

Convert your fine-tuned merged Qwen3.5 models (4B / 9B, bf16) to GGUF Q4_K_M
for Ollama and Hugging Face deployment.

Built for **DGX Spark** using the [NVIDIA Unsloth cookbook](https://build.nvidia.com/spark/unsloth/overview) environment.

---

## What's In This Kit

```
qwen35-gguf-kit/
├── README.md            ← you are here
├── run_docker.sh        ← launches the container with your models mounted
├── setup_env.sh         ← installs deps inside the container (run once)
└── convert_to_gguf.py   ← the converter (pre-flight → convert → validate → artifacts)
```

## Quick Start (4 commands)

### 1. Edit `run_docker.sh`

Open it and set the two model paths to wherever your merged models live:

```bash
MODEL_DIR_4B="/home/you/models/qwen35-4b-merged"
MODEL_DIR_9B="/home/you/models/qwen35-9b-merged"
```

### 2. Launch the container

```bash
chmod +x run_docker.sh
./run_docker.sh
```

This drops you into the pytorch container with your models mounted read-only
at `/workspace/models/qwen35-4b` and `/workspace/models/qwen35-9b`.

### 3. Install dependencies (once)

```bash
bash /workspace/kit/setup_env.sh
```

### 4. Convert!

```bash
# Start with the 4B (faster, validates your pipeline)
python /workspace/kit/convert_to_gguf.py \
    /workspace/models/qwen35-4b \
    --output-dir /workspace/output \
    --model-name my-qwen35-4b

# Then the 9B
python /workspace/kit/convert_to_gguf.py \
    /workspace/models/qwen35-9b \
    --output-dir /workspace/output \
    --model-name my-qwen35-9b
```

Output lands in `/workspace/output/` (mapped to `~/gguf-output` on the host).

---

## After Conversion

### Register with Ollama

```bash
cd ~/gguf-output/my-qwen35-4b-GGUF/
ollama create my-qwen35-4b -f Modelfile
ollama run my-qwen35-4b
```

Repeat for the 9B. Both will appear in `ollama list`.

### Upload to Hugging Face

```bash
cd ~/gguf-output/my-qwen35-4b-GGUF/
# Edit upload_to_hf.sh — set your HF username
huggingface-cli login   # if not already logged in
./upload_to_hf.sh
```

---

## What the Converter Does

The script runs in order:

1. **Pre-flight checks** — validates your merged model directory has config.json,
   tokenizer files, and weight files. Flags LoRA artifacts, vision tensors,
   and tokenizer issues before wasting time on conversion.

2. **Unsloth conversion** (primary path) — loads the model in 16-bit via
   `FastLanguageModel.from_pretrained()` and calls `save_pretrained_gguf()`
   with `q4_k_m`. This handles tokenizer mapping internally.

3. **llama.cpp fallback** — if Unsloth fails, automatically clones latest
   llama.cpp, runs `convert_hf_to_gguf.py` → `llama-quantize`. Gives
   specific diagnostics for tokenizer hash mismatches.

4. **Generates deployment artifacts** — Ollama Modelfile, HF model card,
   and an upload helper script.

5. **Validates the GGUF** — checks magic bytes, reads metadata via the
   gguf Python package, and runs a quick inference test if llama-cli
   is available.

---

## Troubleshooting

### "Unrecognized tokenizer" / tokenizer hash mismatch

Your fine-tuning modified the tokenizer config. Fix: copy `tokenizer.json`,
`tokenizer_config.json`, and any `special_tokens_map.json` from the base
Qwen3.5 model (download from HF) into your merged model directory, replacing
the modified versions. Only do this if you did NOT add custom tokens.

### "Has vision encoder, but it will be ignored"

This is expected and harmless. Qwen3.5 is natively multimodal — the converter
skips vision tensors for text-only GGUF.

### Unsloth import fails

Make sure you ran `setup_env.sh` and are inside the pytorch container.
If Unsloth still fails, the script auto-falls back to raw llama.cpp.

### GGUF loads in llama.cpp but garbled output

Usually a tokenizer issue. Verify:
- The chat template in the Modelfile matches Qwen's `<|im_start|>/<|im_end|>` format
- You're using the `--jinja` flag with llama-cli
- The tokenizer wasn't corrupted during fine-tuning (test: load with `AutoTokenizer` and encode/decode a sentence)

### Ollama says "unsupported model architecture"

Update Ollama: `ollama update` or reinstall latest. Qwen3.5 support
requires a recent Ollama version.

---

## Flags

```
python convert_to_gguf.py --help

positional arguments:
  model_path           Path to merged bf16 model directory

options:
  --output-dir DIR     Output directory (default: ./output)
  --model-name NAME    Name for the output model (default: derived from path)
  --skip-unsloth       Skip Unsloth, go straight to llama.cpp fallback
```
