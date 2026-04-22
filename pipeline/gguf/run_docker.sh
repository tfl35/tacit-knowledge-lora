#!/bin/bash
# ============================================================
# Launch Docker for Qwen3.5 GGUF Conversion on DGX Spark
# ============================================================
# This script launches the nvidia/pytorch container with your
# models and this conversion kit mounted.
#
# BEFORE RUNNING — edit the paths below:
#   MODEL_DIR_4B  = path to your merged Qwen3.5-4B model
#   MODEL_DIR_9B  = path to your merged Qwen3.5-9B model
#   KIT_DIR       = path to this kit (the thumb drive folder)
#   OUTPUT_DIR    = where you want GGUF outputs saved
# ============================================================

# ---- EDIT THESE PATHS ----
MODEL_DIR_4B="/path/to/your/qwen35-4b-merged"
MODEL_DIR_9B="/path/to/your/qwen35-9b-merged"
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"  # auto-detects this script's folder
OUTPUT_DIR="$HOME/gguf-output"
# --------------------------

echo "============================================================"
echo " Launching GGUF conversion container"
echo "============================================================"
echo "  Kit:     $KIT_DIR"
echo "  4B model: $MODEL_DIR_4B"
echo "  9B model: $MODEL_DIR_9B"
echo "  Output:   $OUTPUT_DIR"
echo ""

# Create output dir if needed
mkdir -p "$OUTPUT_DIR"

# Build mount flags (only mount models that exist)
MOUNTS="-v $KIT_DIR:/workspace/kit -v $OUTPUT_DIR:/workspace/output"

if [ -d "$MODEL_DIR_4B" ]; then
    MOUNTS="$MOUNTS -v $MODEL_DIR_4B:/workspace/models/qwen35-4b:ro"
    echo "  Mounting 4B model ✓"
else
    echo "  4B model path not found — skipping (edit MODEL_DIR_4B in this script)"
fi

if [ -d "$MODEL_DIR_9B" ]; then
    MOUNTS="$MOUNTS -v $MODEL_DIR_9B:/workspace/models/qwen35-9b:ro"
    echo "  Mounting 9B model ✓"
else
    echo "  9B model path not found — skipping (edit MODEL_DIR_9B in this script)"
fi

echo ""
echo "Starting container..."
echo "(Once inside, run: bash /workspace/kit/setup_env.sh)"
echo ""

docker run --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    $MOUNTS \
    -it --rm \
    --entrypoint /usr/bin/bash \
    nvcr.io/nvidia/pytorch:25.11-py3
