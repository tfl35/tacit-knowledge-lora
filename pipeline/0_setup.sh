#!/bin/bash
# =============================================================
# 0_setup.sh — One-time setup: build container + download models
# =============================================================
# Builds the Docker image with all dependencies
# and downloads all 4 Qwen3.5 model sizes.
#
# Usage:
#   bash 0_setup.sh                     # all 4 models
#   bash 0_setup.sh --models 9B 4B      # specific models only
#   bash 0_setup.sh --skip-download     # image only, no model download
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
CUSTOM_IMAGE="qwen35-finetune:latest"
HF_CACHE="${HOME}/.cache/huggingface"

# Models to download
ALL_MODELS=("Qwen/Qwen3.5-9B" "Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-0.8B")
SELECTED_MODELS=("${ALL_MODELS[@]}")
SKIP_DOWNLOAD=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            SELECTED_MODELS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]]; do
                case $1 in
                    9B)   SELECTED_MODELS+=("Qwen/Qwen3.5-9B") ;;
                    4B)   SELECTED_MODELS+=("Qwen/Qwen3.5-4B") ;;
                    2B)   SELECTED_MODELS+=("Qwen/Qwen3.5-2B") ;;
                    0.8B) SELECTED_MODELS+=("Qwen/Qwen3.5-0.8B") ;;
                    *)    SELECTED_MODELS+=("$1") ;;
                esac
                shift
            done
            ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        *) shift ;;
    esac
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  Cross-Scale Capstone — Setup"
echo "════════════════════════════════════════════════════"
echo "  Models: ${SELECTED_MODELS[*]}"
echo ""

# ── Step 1: Preflight ──
echo "[1/4] Preflight checks..."
command -v docker &>/dev/null || { echo "ERROR: Docker not found"; exit 1; }
nvidia-smi &>/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Docker: OK"

# ── Step 2: Check if image already exists ──
if docker image inspect "$CUSTOM_IMAGE" &>/dev/null; then
    echo ""
    echo "[2/4] Custom image already exists: $CUSTOM_IMAGE"
    echo "  Skipping build. Delete image to rebuild: docker rmi $CUSTOM_IMAGE"
else
    echo ""
    echo "[2/4] Pulling base image and building custom image..."
    docker pull "$BASE_IMAGE"

    CONTAINER_ID=$(docker run -d --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$SCRIPT_DIR":/workspace \
        -w /workspace \
        "$BASE_IMAGE" \
        bash -c '
            set -e
            export PIP_CONSTRAINT=""
            export CUDA_HOME=/usr/local/cuda
            export PATH=$CUDA_HOME/bin:$PATH

            echo "Installing core packages..."
            pip install --no-cache-dir --root-user-action=ignore \
                "transformers>=5.0.0" \
                "datasets>=3.0.0" \
                "accelerate>=1.0.0" \
                "trl>=0.29.0" \
                "peft>=0.18.0"

            echo ""
            echo "Installing Qwen3.5 CUDA kernels (~5 min)..."
            pip install --no-cache-dir --no-build-isolation --root-user-action=ignore causal-conv1d
            pip install --no-cache-dir --no-build-isolation --root-user-action=ignore flash-linear-attention

            echo ""
            echo "Verifying..."
            python3 -c "
import torch, transformers, peft, trl
print(f\"  PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}\")
print(f\"  Transformers {transformers.__version__}, TRL {trl.__version__}, PEFT {peft.__version__}\")
import fla, causal_conv1d
print(f\"  Qwen3.5 kernels: OK\")
"
            echo "SETUP_OK"
        ')

    echo "  Waiting for installation..."
    docker logs -f "$CONTAINER_ID"
    EXIT_CODE=$(docker wait "$CONTAINER_ID")

    if [ "$EXIT_CODE" -ne 0 ] || ! docker logs "$CONTAINER_ID" 2>&1 | grep -q "SETUP_OK"; then
        echo "ERROR: Setup failed"
        docker rm "$CONTAINER_ID" >/dev/null 2>&1
        exit 1
    fi

    docker commit "$CONTAINER_ID" "$CUSTOM_IMAGE" >/dev/null
    docker rm "$CONTAINER_ID" >/dev/null
    echo "  Custom image ready: $CUSTOM_IMAGE"
fi

# ── Step 3: Download models ──
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "[3/4] Downloading models..."
    for MODEL in "${SELECTED_MODELS[@]}"; do
        echo "  Downloading: $MODEL"
        docker run --gpus all --rm \
            --ipc=host \
            -v "$HF_CACHE":/root/.cache/huggingface \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            "$CUSTOM_IMAGE" \
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL')
print('  Downloaded: $MODEL')
"
    done
else
    echo ""
    echo "[3/4] Skipping model downloads."
fi

# ── Step 4: Verify ──
echo ""
echo "[4/4] Verifying setup..."
docker run --gpus all --rm \
    -v "$HF_CACHE":/root/.cache/huggingface \
    "$CUSTOM_IMAGE" \
    python3 -c "
import torch, transformers, peft, trl
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  Transformers {transformers.__version__}')
print(f'  PEFT {peft.__version__}')
print(f'  TRL {trl.__version__}')
import fla, causal_conv1d
print(f'  Qwen3.5 kernels: OK')
print('  All checks passed.')
"

echo ""
echo "════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "════════════════════════════════════════════════════"
echo "  Next steps:"
echo "    1. Place your training dataset at: dataset/ti_350_production.json"
echo "       See dataset/README.md for format specification"
echo "    2. Run: bash run_all.sh"
echo "════════════════════════════════════════════════════"
