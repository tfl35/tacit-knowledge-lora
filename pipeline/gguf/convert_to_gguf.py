#!/usr/bin/env python3
"""
Qwen3.5 Fine-Tuned Model → GGUF Q4_K_M Converter
==================================================
Plug-and-play script for converting merged bf16 Qwen3.5 models (4B/9B)
to GGUF format with Q4_K_M quantization.

Designed for DGX Spark with the nvidia/pytorch:25.11-py3 container
and Unsloth, following the Spark Unsloth cookbook.

Usage:
    python convert_to_gguf.py /path/to/merged_model [--output-dir ./output] [--model-name my-qwen35]
"""

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path


# ──────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────
QUANT_METHOD = "q4_k_m"
OLLAMA_TEMPLATE = """<|im_start|>system
{{{{.System}}}}<|im_end|>
<|im_start|>user
{{{{.Prompt}}}}<|im_end|>
<|im_start|>assistant
"""
OLLAMA_STOP = "<|im_end|>"
TEST_PROMPT = "Hello! Can you tell me a brief fun fact about space?"


def log(msg, level="INFO"):
    colors = {"INFO": "\033[94m", "OK": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m", "STEP": "\033[95m"}
    reset = "\033[0m"
    prefix = colors.get(level, "")
    print(f"{prefix}[{level}]{reset} {msg}")


def run_cmd(cmd, desc="", check=True):
    """Run a shell command with logging."""
    if desc:
        log(desc, "STEP")
    log(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        log(f"Command failed (exit {result.returncode}):", "FAIL")
        log(f"  stdout: {result.stdout[-500:]}" if result.stdout else "  (no stdout)")
        log(f"  stderr: {result.stderr[-500:]}" if result.stderr else "  (no stderr)")
        return False, result
    return True, result


# ──────────────────────────────────────────────
# 1. PRE-FLIGHT CHECKS
# ──────────────────────────────────────────────
def preflight_check(model_path: Path):
    """Validate the merged model directory before attempting conversion."""
    log("=" * 60)
    log("PRE-FLIGHT CHECKS", "STEP")
    log("=" * 60)
    errors = []

    # Check directory exists
    if not model_path.is_dir():
        log(f"Model path does not exist: {model_path}", "FAIL")
        sys.exit(1)

    # Check required files
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for f in required_files:
        fp = model_path / f
        if fp.exists():
            log(f"  Found {f}", "OK")
        else:
            log(f"  MISSING {f}", "FAIL")
            errors.append(f)

    # Check for safetensors or bin files
    safetensors = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("*.bin"))
    if safetensors:
        total_size = sum(f.stat().st_size for f in safetensors)
        log(f"  Found {len(safetensors)} safetensors file(s) ({total_size / 1e9:.1f} GB)", "OK")
    elif bin_files:
        total_size = sum(f.stat().st_size for f in bin_files)
        log(f"  Found {len(bin_files)} .bin file(s) ({total_size / 1e9:.1f} GB)", "OK")
    else:
        log("  No model weight files found (.safetensors or .bin)", "FAIL")
        errors.append("model weights")

    # Read and validate config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "unknown")
        architectures = config.get("architectures", [])
        log(f"  model_type: {model_type}", "INFO")
        log(f"  architectures: {architectures}", "INFO")

        # Qwen3.5 may report as qwen3_5, qwen2, or qwen3
        known_types = ["qwen3_5", "qwen2", "qwen3", "qwen2_5"]
        if model_type not in known_types:
            log(f"  WARNING: model_type '{model_type}' not in expected {known_types}", "WARN")
            log(f"  This may still work if llama.cpp has architecture support.", "WARN")
    else:
        log("  Cannot validate config.json (missing)", "WARN")

    # Check tokenizer_config.json for chat_template
    tok_config_path = model_path / "tokenizer_config.json"
    if tok_config_path.exists():
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        if "chat_template" in tok_config:
            log("  chat_template present in tokenizer_config.json", "OK")
        else:
            log("  No chat_template in tokenizer_config.json (Ollama Modelfile will handle this)", "WARN")

    # Check for leftover LoRA artifacts
    if config_path.exists():
        with open(config_path) as f:
            config_text = f.read()
        if "lora" in config_text.lower() and "peft" not in str(model_path).lower():
            log("  WARNING: config.json references 'lora' — verify this is a fully merged model", "WARN")

    # Check for vision encoder weights that might cause issues
    if safetensors:
        try:
            # Quick check via safetensors header (first file)
            import struct
            with open(safetensors[0], "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))
            vision_keys = [k for k in header.keys() if "visual" in k.lower() or "vision" in k.lower()]
            if vision_keys:
                log(f"  Found {len(vision_keys)} vision-related tensor(s) — Unsloth will skip these", "WARN")
            else:
                log("  No vision encoder tensors detected", "OK")
        except Exception:
            log("  Could not inspect safetensors headers (non-critical)", "WARN")

    if errors:
        log(f"\nPre-flight failed: missing {errors}", "FAIL")
        log("Your merged model directory must contain config.json, tokenizer.json,", "FAIL")
        log("tokenizer_config.json, and model weight files.", "FAIL")
        sys.exit(1)

    log("\nPre-flight checks passed!\n", "OK")


# ──────────────────────────────────────────────
# 2. UNSLOTH CONVERSION (PRIMARY PATH)
# ──────────────────────────────────────────────
def convert_with_unsloth(model_path: Path, output_dir: Path, model_name: str):
    """
    Convert using Unsloth's save_pretrained_gguf.
    This handles tokenizer mapping and calls llama.cpp internally.
    """
    log("=" * 60)
    log("CONVERTING WITH UNSLOTH", "STEP")
    log("=" * 60)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log("Unsloth not installed. Run the setup script first.", "FAIL")
        return None

    log(f"Loading merged model from: {model_path}")
    log("Loading in 16-bit (bf16) since your model is already merged bf16...")
    log("(This avoids double-quantization that load_in_4bit would cause)\n")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            dtype=None,           # auto-detect bf16
            load_in_4bit=False,   # DO NOT 4-bit load a model we're about to quantize
            load_in_16bit=True,   # keep full bf16 precision as the quantization source
            local_files_only=True,
        )
    except Exception as e:
        log(f"Failed to load model: {e}", "FAIL")
        # Try with FastModel as fallback (newer Unsloth API)
        log("Trying FastModel API instead...", "WARN")
        try:
            from unsloth import FastModel
            model, tokenizer = FastModel.from_pretrained(
                model_name=str(model_path),
                max_seq_length=2048,
                load_in_4bit=False,
                load_in_16bit=True,
                full_finetuning=False,
            )
        except Exception as e2:
            log(f"FastModel also failed: {e2}", "FAIL")
            return None

    log("Model loaded successfully!", "OK")

    # Create output directory
    gguf_dir = output_dir / f"{model_name}-GGUF"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    log(f"\nQuantizing to {QUANT_METHOD} and saving GGUF...")
    log("(This calls llama.cpp internally — may take 5-20 min depending on model size)\n")

    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=QUANT_METHOD,
        )
    except Exception as e:
        log(f"GGUF conversion failed: {e}", "FAIL")
        log("\nFalling back to manual llama.cpp conversion...", "WARN")
        return None

    # Find the output GGUF file
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        log("No GGUF file produced!", "FAIL")
        return None

    gguf_file = gguf_files[0]
    size_gb = gguf_file.stat().st_size / 1e9
    log(f"\nGGUF file created: {gguf_file} ({size_gb:.2f} GB)", "OK")
    return gguf_file


# ──────────────────────────────────────────────
# 3. LLAMA.CPP FALLBACK (IF UNSLOTH FAILS)
# ──────────────────────────────────────────────
def convert_with_llamacpp(model_path: Path, output_dir: Path, model_name: str):
    """
    Direct llama.cpp conversion as fallback.
    Clones llama.cpp, runs convert_hf_to_gguf.py, then llama-quantize.
    """
    log("=" * 60)
    log("FALLBACK: CONVERTING WITH LLAMA.CPP DIRECTLY", "STEP")
    log("=" * 60)

    llamacpp_dir = Path("/tmp/llama.cpp")
    gguf_dir = output_dir / f"{model_name}-GGUF"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    bf16_gguf = gguf_dir / f"{model_name}-bf16.gguf"
    q4km_gguf = gguf_dir / f"{model_name}-Q4_K_M.gguf"

    # Clone llama.cpp
    if not llamacpp_dir.exists():
        ok, _ = run_cmd(
            f"git clone --depth 1 https://github.com/ggml-org/llama.cpp.git {llamacpp_dir}",
            "Cloning llama.cpp (latest)..."
        )
        if not ok:
            log("Failed to clone llama.cpp", "FAIL")
            return None

    # Install Python deps for converter
    run_cmd("pip install gguf sentencepiece protobuf --quiet", "Installing gguf Python package...")

    # Step 1: Convert HF → bf16 GGUF
    ok, result = run_cmd(
        f"python {llamacpp_dir}/convert_hf_to_gguf.py {model_path} "
        f"--outtype bf16 --outfile {bf16_gguf}",
        "Converting HF model → bf16 GGUF..."
    )
    if not ok:
        log("HF → GGUF conversion failed.", "FAIL")
        # Check for tokenizer hash warning
        if "chkhsh" in (result.stderr or ""):
            log("\nThis is likely a TOKENIZER HASH MISMATCH.", "WARN")
            log("Your fine-tuned model's tokenizer config differs from the base model.", "WARN")
            log("Options:", "WARN")
            log("  1. Copy tokenizer files from the original Qwen3.5 base model", "WARN")
            log("  2. Add --vocab-only flag to test tokenizer conversion alone", "WARN")
            log("  3. Update convert_hf_to_gguf_update.py with your tokenizer hash", "WARN")
        return None

    if not bf16_gguf.exists():
        log(f"Expected file not found: {bf16_gguf}", "FAIL")
        return None

    log(f"bf16 GGUF created: {bf16_gguf} ({bf16_gguf.stat().st_size / 1e9:.2f} GB)", "OK")

    # Step 2: Build llama-quantize
    log("Building llama-quantize...", "STEP")
    build_dir = llamacpp_dir / "build"
    build_dir.mkdir(exist_ok=True)
    ok, _ = run_cmd(
        f"cd {build_dir} && cmake .. -DGGML_CUDA=ON && cmake --build . --target llama-quantize -j$(nproc)",
        "Building with CUDA support..."
    )
    if not ok:
        # Try CPU-only build
        log("CUDA build failed, trying CPU-only...", "WARN")
        ok, _ = run_cmd(
            f"cd {build_dir} && cmake .. -DGGML_CUDA=OFF && cmake --build . --target llama-quantize -j$(nproc)",
            "Building CPU-only..."
        )
        if not ok:
            log("Failed to build llama-quantize", "FAIL")
            return None

    quantize_bin = build_dir / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        # Try alternative path
        quantize_bin = build_dir / "llama-quantize"
    if not quantize_bin.exists():
        log("Cannot find llama-quantize binary after build", "FAIL")
        return None

    # Step 3: Quantize bf16 → Q4_K_M
    ok, _ = run_cmd(
        f"{quantize_bin} {bf16_gguf} {q4km_gguf} Q4_K_M",
        "Quantizing bf16 → Q4_K_M..."
    )
    if not ok:
        log("Quantization failed", "FAIL")
        return None

    if q4km_gguf.exists():
        size_gb = q4km_gguf.stat().st_size / 1e9
        log(f"\nGGUF file created: {q4km_gguf} ({size_gb:.2f} GB)", "OK")
        # Clean up the large bf16 intermediate
        log(f"Cleaning up intermediate bf16 file...")
        bf16_gguf.unlink()
        return q4km_gguf

    log("Quantized file not found", "FAIL")
    return None


# ──────────────────────────────────────────────
# 4. POST-CONVERSION: OLLAMA + HF ARTIFACTS
# ──────────────────────────────────────────────
def create_ollama_modelfile(gguf_path: Path, model_name: str):
    """Generate an Ollama Modelfile next to the GGUF."""
    modelfile_path = gguf_path.parent / "Modelfile"
    content = f"""# Ollama Modelfile for {model_name}
# Usage: ollama create {model_name} -f Modelfile
FROM ./{gguf_path.name}

TEMPLATE \"\"\"{OLLAMA_TEMPLATE}\"\"\"

PARAMETER stop "{OLLAMA_STOP}"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
    with open(modelfile_path, "w") as f:
        f.write(content)
    log(f"Ollama Modelfile created: {modelfile_path}", "OK")
    return modelfile_path


def create_hf_readme(gguf_path: Path, model_name: str, source_model_path: Path):
    """Generate a basic HF model card for the GGUF upload."""
    readme_path = gguf_path.parent / "README.md"

    # Try to read base model info from config
    base_model = "Qwen3.5"
    config_path = source_model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        architectures = cfg.get("architectures", [])
        hidden_size = cfg.get("hidden_size", "?")
        num_layers = cfg.get("num_hidden_layers", "?")
        base_model = f"Qwen3.5 ({hidden_size}h, {num_layers}L)"

    content = f"""---
language:
- en
tags:
- gguf
- qwen3.5
- q4_k_m
---

# {model_name} (GGUF Q4_K_M)

This is a Q4_K_M GGUF quantization of a fine-tuned {base_model} model.

## Quantization Details

- **Source precision:** bf16 (merged fine-tune)
- **Quantization:** Q4_K_M via llama.cpp
- **Architecture:** {base_model}

## Usage with Ollama

```bash
# Download the GGUF and Modelfile, then:
ollama create {model_name} -f Modelfile
ollama run {model_name}
```

## Usage with llama.cpp

```bash
llama-cli -m {gguf_path.name} --jinja --color -ngl 99 -fa -c 4096
```
"""
    with open(readme_path, "w") as f:
        f.write(content)
    log(f"HF README.md created: {readme_path}", "OK")
    return readme_path


def create_hf_upload_script(gguf_path: Path, model_name: str):
    """Generate a convenience script for uploading to HF."""
    script_path = gguf_path.parent / "upload_to_hf.sh"
    content = f"""#!/bin/bash
# Upload GGUF model to Hugging Face
# Edit HF_USERNAME before running
set -e

HF_USERNAME="YOUR_HF_USERNAME"
REPO_NAME="{model_name}-GGUF"

echo "Uploading to $HF_USERNAME/$REPO_NAME ..."
echo "Make sure you're logged in: huggingface-cli login"

huggingface-cli repo create "$REPO_NAME" --type model || true
huggingface-cli upload "$HF_USERNAME/$REPO_NAME" "{gguf_path.name}" "{gguf_path.name}"
huggingface-cli upload "$HF_USERNAME/$REPO_NAME" "README.md" "README.md"
huggingface-cli upload "$HF_USERNAME/$REPO_NAME" "Modelfile" "Modelfile"

echo ""
echo "Done! Model available at: https://huggingface.co/$HF_USERNAME/$REPO_NAME"
"""
    with open(script_path, "w") as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    log(f"HF upload script created: {script_path}", "OK")


# ──────────────────────────────────────────────
# 5. VALIDATION
# ──────────────────────────────────────────────
def validate_gguf(gguf_path: Path):
    """Run basic validation on the produced GGUF file."""
    log("=" * 60)
    log("VALIDATION", "STEP")
    log("=" * 60)

    # Check file size sanity
    size_gb = gguf_path.stat().st_size / 1e9
    if size_gb < 0.5:
        log(f"GGUF file suspiciously small ({size_gb:.2f} GB) — conversion may have failed", "WARN")
    else:
        log(f"File size: {size_gb:.2f} GB — looks reasonable for Q4_K_M", "OK")

    # Check GGUF magic bytes
    with open(gguf_path, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
        log("GGUF magic header: valid", "OK")
    else:
        log(f"GGUF magic header: INVALID (got {magic!r})", "FAIL")
        return False

    # Try to read metadata with gguf Python package
    try:
        from gguf import GGUFReader
        reader = GGUFReader(str(gguf_path))
        arch = None
        for field in reader.fields.values():
            if field.name == "general.architecture":
                arch = str(bytes(field.parts[-1]), "utf-8")
                break
        if arch:
            log(f"Architecture in GGUF: {arch}", "OK")
        tensor_count = len(reader.tensors)
        log(f"Tensor count: {tensor_count}", "OK")
        if tensor_count < 10:
            log("Very few tensors — possible incomplete conversion", "WARN")
    except ImportError:
        log("gguf package not available — skipping metadata validation", "WARN")
    except Exception as e:
        log(f"Could not read GGUF metadata: {e}", "WARN")

    # Try llama.cpp inference test if available
    llamacpp_cli = Path("/tmp/llama.cpp/build/bin/llama-cli")
    if not llamacpp_cli.exists():
        llamacpp_cli = Path("/tmp/llama.cpp/build/llama-cli")
    if llamacpp_cli.exists():
        log("\nRunning quick inference test with llama-cli...", "STEP")
        ok, result = run_cmd(
            f'{llamacpp_cli} -m {gguf_path} -p "{TEST_PROMPT}" -n 50 -ngl 99 2>&1 | tail -20',
            desc="",
            check=False
        )
        if ok and result.stdout:
            # Show last few lines of output
            lines = result.stdout.strip().split("\n")[-5:]
            for line in lines:
                log(f"  > {line}", "INFO")
            log("Inference test completed — check output above for coherence", "OK")
        else:
            log("Inference test did not produce output (non-critical)", "WARN")
    else:
        log("llama-cli not built — skipping inference test", "WARN")
        log("You can test manually with: ollama run <model-name>", "INFO")

    log("\nValidation complete!", "OK")
    return True


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert a merged bf16 Qwen3.5 model to GGUF Q4_K_M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert the 4B model
  python convert_to_gguf.py /models/qwen35-4b-merged --model-name my-qwen35-4b

  # Convert the 9B model with custom output location
  python convert_to_gguf.py /models/qwen35-9b-merged --output-dir /output --model-name my-qwen35-9b

  # After conversion, register with Ollama:
  cd output/my-qwen35-4b-GGUF/
  ollama create my-qwen35-4b -f Modelfile
  ollama run my-qwen35-4b
        """
    )
    parser.add_argument("model_path", type=str, help="Path to merged bf16 model directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--model-name", type=str, default=None, help="Name for the output model (default: derived from path)")
    parser.add_argument("--skip-unsloth", action="store_true", help="Skip Unsloth, go straight to llama.cpp")

    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_name = args.model_name or model_path.name

    log("=" * 60)
    log(f"Qwen3.5 → GGUF Q4_K_M Converter")
    log(f"=" * 60)
    log(f"  Source:  {model_path}")
    log(f"  Output:  {output_dir}")
    log(f"  Name:    {model_name}")
    log(f"  Quant:   {QUANT_METHOD}")
    log("")

    # Pre-flight
    preflight_check(model_path)

    # Convert
    gguf_path = None

    if not args.skip_unsloth:
        gguf_path = convert_with_unsloth(model_path, output_dir, model_name)

    if gguf_path is None:
        log("\nUnsloth path unavailable or failed — using llama.cpp directly\n", "WARN")
        gguf_path = convert_with_llamacpp(model_path, output_dir, model_name)

    if gguf_path is None:
        log("\n" + "=" * 60, "FAIL")
        log("CONVERSION FAILED", "FAIL")
        log("=" * 60, "FAIL")
        log("Both Unsloth and llama.cpp conversion paths failed.", "FAIL")
        log("Check the error messages above. Common fixes:", "FAIL")
        log("  1. Update Unsloth: pip install --upgrade unsloth unsloth_zoo", "INFO")
        log("  2. Ensure transformers is latest: pip install --upgrade transformers", "INFO")
        log("  3. Check your model directory has all required files", "INFO")
        log("  4. Try copying tokenizer files from the base Qwen3.5 model", "INFO")
        sys.exit(1)

    # Post-conversion artifacts
    log("\n" + "=" * 60)
    log("CREATING DEPLOYMENT ARTIFACTS", "STEP")
    log("=" * 60)

    create_ollama_modelfile(gguf_path, model_name)
    create_hf_readme(gguf_path, model_name, model_path)
    create_hf_upload_script(gguf_path, model_name)

    # Validate
    validate_gguf(gguf_path)

    # Final summary
    gguf_dir = gguf_path.parent
    log("\n" + "=" * 60)
    log("ALL DONE!", "OK")
    log("=" * 60)
    log(f"\nOutput directory: {gguf_dir}")
    log(f"  {gguf_path.name}  — the quantized model")
    log(f"  Modelfile         — for Ollama registration")
    log(f"  README.md         — HF model card")
    log(f"  upload_to_hf.sh   — HF upload helper")
    log(f"\n--- NEXT STEPS ---\n")
    log(f"  Register with Ollama:")
    log(f"    cd {gguf_dir}")
    log(f"    ollama create {model_name} -f Modelfile")
    log(f"    ollama run {model_name}")
    log(f"")
    log(f"  Upload to Hugging Face:")
    log(f"    cd {gguf_dir}")
    log(f"    # Edit upload_to_hf.sh with your HF username first")
    log(f"    ./upload_to_hf.sh")


if __name__ == "__main__":
    main()
