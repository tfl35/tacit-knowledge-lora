#!/bin/bash
# =============================================================
# run_ablation.sh — Ablation Study + Behavioral Divergence
# =============================================================
# Runs two extension studies on the 9B model:
#
# 1. ABLATION STUDY: Train 9B on 50, 100, 150, 200, 250, 300 examples
#    (350 already exists). Evaluate each. Produce ablation curve.
#
# 2. BEHAVIORAL DIVERGENCE: Capture logits from 9B-base and 9B-ft,
#    compute agreement rate, entropy delta, domain token rank shifts.
#
# Prerequisites:
#   - Ablation subsets in dataset/ablation/ (run extract_ablation_subsets.py first)
#   - 9B-ft results already exist from the main pipeline run
#   - Docker image qwen35-finetune:latest available
#
# Usage:
#   bash run_ablation.sh                    # full study
#   bash run_ablation.sh --ablation-only    # skip divergence
#   bash run_ablation.sh --divergence-only  # skip ablation training
#
# Estimated time:
#   Ablation training: ~5 hours (5 runs × ~60 min each)
#   Ablation eval:     ~4 hours (5 × ~45 min per ft eval)
#   Divergence base:   ~90 min (15 questions with logit capture)
#   Divergence ft:     ~90 min
#   Divergence compare: <1 min
#   Total:             ~12 hours
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE="qwen35-finetune:latest"
HF_CACHE="${HOME}/.cache/huggingface"
MODEL_ID="Qwen/Qwen3.5-9B"
EPOCHS=3
ABLATION_SIZES=(50 100 150 200 250 300)

RUN_ABLATION=true
RUN_DIVERGENCE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --ablation-only)   RUN_DIVERGENCE=false; shift ;;
        --divergence-only) RUN_ABLATION=false; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

run_docker() {
    docker run --gpus all --rm \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$SCRIPT_DIR":/workspace \
        -v "$SCRIPT_DIR/../dataset":/workspace/dataset \
        -v "$HF_CACHE":/root/.cache/huggingface \
        -w /workspace \
        "$IMAGE" \
        "$@"
}

START_TIME=$(date +%s)
mkdir -p results

# ── Preflight ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Extension Studies: Ablation + Behavioral Divergence"
echo "════════════════════════════════════════════════════════════"
echo "  Ablation:   $RUN_ABLATION"
echo "  Divergence: $RUN_DIVERGENCE"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check Docker image
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "ERROR: Image '$IMAGE' not found. Run: bash 0_setup.sh"
    exit 1
fi

# Check dataset exists
if [ "$RUN_ABLATION" = true ] && [ ! -f "../dataset/ti_350_production.json" ]; then
    echo "ERROR: ../dataset/ti_350_production.json not found"
    exit 1
fi

# For analyze_ablation: need the 350-example eval as reference
if [ "$RUN_ABLATION" = true ] && [ ! -f "results/eval_qwen3.5-9b-ft.json" ]; then
    echo ""
    echo "WARNING: results/eval_qwen3.5-9b-ft.json not found."
    echo "The ablation analysis needs this as the 350-example reference."
    echo "Copy it from your main pipeline output, or the analysis step will"
    echo "run without the 350 baseline (ablation curve will be incomplete)."
    echo ""
fi

# ─────────────────────────────────────────────────────────────
# PART 1: ABLATION STUDY
# ─────────────────────────────────────────────────────────────
if [ "$RUN_ABLATION" = true ]; then
    # Step 0: Generate subsets if they don't exist
    if [ ! -f "../dataset/ablation/ti_50_ablation.json" ]; then
        echo "[ABLATION] Generating subset files..."
        run_docker python3 scripts/extract_ablation_subsets.py \
            dataset/ti_350_production.json dataset/ablation/
    fi

    for SIZE in "${ABLATION_SIZES[@]}"; do
        DATASET="dataset/ablation/ti_${SIZE}_ablation.json"
        TAG="qwen3.5-9b-abl${SIZE}"

        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  Ablation: 9B × ${SIZE} examples"
        echo "╚══════════════════════════════════════════════════════════╝"

        if [ ! -f "$DATASET" ]; then
            echo "  ERROR: $DATASET not found"
            continue
        fi

        # Train
        echo "[1/3] Training 9B on ${SIZE} examples (${EPOCHS} epochs)..."
        run_docker python3 scripts/train.py \
            --dataset "$DATASET" \
            --model "$MODEL_ID" \
            --epochs "$EPOCHS" \
            --output output

        RUN_DIR=$(ls -1d output/qwen3.5-9b/run_* 2>/dev/null | sort | tail -1)
        echo "  Training complete: $RUN_DIR"

        # Merge
        MERGED_DIR="merged_${TAG}"
        echo "[2/3] Merging → $MERGED_DIR..."
        run_docker python3 scripts/merge.py \
            --adapter "$RUN_DIR" \
            --base "$MODEL_ID" \
            --output "$MERGED_DIR"

        # Evaluate (fine-tuned only — base already exists)
        FT_LABEL="${TAG}-ft"
        echo "[3/3] Evaluating → $FT_LABEL..."
        run_docker python3 scripts/evaluate.py \
            --model "$MERGED_DIR" \
            --label "$FT_LABEL" \
            --output "results/eval_${FT_LABEL}.json"

        echo "  ✓ Ablation ${SIZE} complete."
    done

    # Run cross-scale analysis including ablation results
    echo ""
    echo "[ABLATION] Running analysis across all results..."
    run_docker python3 scripts/analyze.py results/
    run_docker python3 scripts/analyze_ablation.py results/

    echo ""
    echo "  ✓ Ablation study complete."
fi

# ─────────────────────────────────────────────────────────────
# PART 2: BEHAVIORAL DIVERGENCE
# ─────────────────────────────────────────────────────────────
if [ "$RUN_DIVERGENCE" = true ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Behavioral Divergence: Logit Capture"
    echo "╚══════════════════════════════════════════════════════════╝"

    # Capture base model logits
    if [ ! -f "results/div_9b-base.json" ]; then
        echo "[DIV 1/3] Capturing 9B base model logits..."
        run_docker python3 scripts/divergence.py \
            --model "$MODEL_ID" \
            --label "9b-base" \
            --output results/
    else
        echo "  Using existing: results/div_9b-base.json"
    fi

    # Capture fine-tuned model logits (use the 350-example FT)
    # First check for the main pipeline merged model, then fallback to latest ablation merge
    MERGED_FT="merged_qwen3.5-9b"
    if [ ! -d "$MERGED_FT" ]; then
        echo "  merged_qwen3.5-9b not found (main pipeline run)."
        echo "  Looking for latest ablation merged model..."
        MERGED_FT=$(ls -1d merged_qwen3.5-9b-abl* 2>/dev/null | sort | tail -1)
        if [ -z "$MERGED_FT" ]; then
            echo "  ERROR: No merged 9B model found. Run ablation first, or copy"
            echo "  merged_qwen3.5-9b from your main pipeline output directory."
            exit 1
        fi
        echo "  Using: $MERGED_FT"
    fi
    if [ ! -f "results/div_9b-ft.json" ]; then
        echo "[DIV 2/3] Capturing 9B fine-tuned model logits..."
        run_docker python3 scripts/divergence.py \
            --model "$MERGED_FT" \
            --label "9b-ft" \
            --output results/
    else
        echo "  Using existing: results/div_9b-ft.json"
    fi

    # Compare
    echo "[DIV 3/3] Computing divergence metrics..."
    run_docker python3 scripts/divergence.py \
        --compare results/div_9b-base.json results/div_9b-ft.json \
        --output results/

    echo ""
    echo "  ✓ Behavioral divergence analysis complete."
fi

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  EXTENSION STUDIES COMPLETE"
echo "  Total time: ${ELAPSED} minutes"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Ablation results:"
ls -la results/eval_*abl*.json 2>/dev/null
echo ""
echo "  Divergence results:"
ls -la results/div_*.json results/divergence_*.json 2>/dev/null
echo ""
