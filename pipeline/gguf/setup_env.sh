#!/bin/bash
# ============================================================
# Qwen3.5 GGUF Conversion — Environment Setup for DGX Spark
# ============================================================
# Based on the NVIDIA DGX Spark Unsloth cookbook:
#   https://build.nvidia.com/spark/unsloth/instructions
#
# This script sets up the conversion environment INSIDE the
# nvidia/pytorch container. Run this ONCE after launching Docker.
#
# Usage:
#   1. Launch the container (see run_docker.sh)
#   2. Inside container: bash setup_env.sh
# ============================================================
set -e

echo "============================================================"
echo " Setting up Qwen3.5 GGUF conversion environment"
echo "============================================================"
echo ""

# ---- Step 1: Core deps (matches Spark cookbook pinning) ----
echo "[1/4] Installing core dependencies..."
pip install --quiet \
    transformers \
    peft \
    hf_transfer \
    "datasets==4.3.0" \
    "trl==0.26.1"

# ---- Step 2: Unsloth (no-deps to avoid conflicts) ----
echo "[2/4] Installing Unsloth..."
pip install --quiet --no-deps unsloth unsloth_zoo bitsandbytes

# ---- Step 3: GGUF support ----
echo "[3/4] Installing GGUF tooling..."
pip install --quiet gguf sentencepiece protobuf

# ---- Step 4: HF CLI for uploads ----
echo "[4/4] Installing Hugging Face CLI..."
pip install --quiet "huggingface_hub>=0.34.0"

echo ""
echo "============================================================"
echo " Environment ready!"
echo "============================================================"
echo ""
echo " Quick validation:"
python -c "from unsloth import FastLanguageModel; print('  Unsloth: OK')" 2>/dev/null || echo "  Unsloth: FAILED"
python -c "import gguf; print('  gguf: OK')" 2>/dev/null || echo "  gguf: FAILED"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>/dev/null || echo "  transformers: FAILED"
echo ""
echo "You're ready to convert. Run:"
echo "  python convert_to_gguf.py /path/to/merged_model --model-name my-qwen35"
echo ""
