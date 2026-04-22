#!/bin/bash
# =============================================================
# run_all.sh — Full Cross-Scale Training + Evaluation Pipeline
# =============================================================
# Runs train → merge → eval (base) → eval (ft) for all 4 model sizes.
# Sequential execution — DGX Spark can only hold one model at a time.
#
# Usage:
#   bash run_all.sh                     # full pipeline, all 4 models
#   bash run_all.sh --models 9B 4B      # specific models only
#   bash run_all.sh --skip-train        # eval only (adapters must exist)
#   bash run_all.sh --skip-base-eval    # skip base model evaluation
#   bash run_all.sh --eval-only         # run analysis on existing results
#
# Estimated time (DGX Spark, 128GB, 350 examples, 3 epochs):
#   9B:  ~30 min train + ~15 min merge + ~30 min eval = ~75 min
#   4B:  ~20 min train + ~10 min merge + ~20 min eval = ~50 min
#   2B:  ~15 min train + ~8 min merge  + ~15 min eval = ~38 min
#   0.8B: ~8 min train + ~5 min merge  + ~10 min eval = ~23 min
#   Total: ~3-4 hours (with overhead)
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE="qwen35-finetune:latest"
HF_CACHE="${HOME}/.cache/huggingface"
DATASET="dataset/ti_350_production.json"
EPOCHS=3

# ── Model registry ──────────────────────────────────────────
declare -A MODELS
MODELS[9B]="Qwen/Qwen3.5-9B"
MODELS[4B]="Qwen/Qwen3.5-4B"
MODELS[2B]="Qwen/Qwen3.5-2B"
MODELS[0.8B]="Qwen/Qwen3.5-0.8B"

# Default: all models
SELECTED_MODELS=("9B" "4B" "2B" "0.8B")
SKIP_TRAIN=false
SKIP_BASE_EVAL=false
EVAL_ONLY=false

# ── Parse arguments ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            SELECTED_MODELS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]]; do
                SELECTED_MODELS+=("$1")
                shift
            done
            ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-base-eval) SKIP_BASE_EVAL=true; shift ;;
        --eval-only)  EVAL_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Preflight ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Cross-Scale Capstone Pipeline"
echo "════════════════════════════════════════════════════════════"
echo "  Models:  ${SELECTED_MODELS[*]}"
echo "  Dataset: $DATASET"
echo "  Epochs:  $EPOCHS"
echo "  Skip training:   $SKIP_TRAIN"
echo "  Skip base eval:  $SKIP_BASE_EVAL"
echo "  Eval only:       $EVAL_ONLY"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check Docker image
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "ERROR: Image '$IMAGE' not found. Run: bash 0_setup.sh"
    exit 1
fi

# Check dataset
if [ "$EVAL_ONLY" = false ] && [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

# ── Helper: run command in Docker ───────────────────────────
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

# ── Helper: find latest training run for a model tag ────────
find_latest_run() {
    local tag="$1"
    ls -1d "output/${tag}/run_"* 2>/dev/null | sort | tail -1
}

# ── Main pipeline ───────────────────────────────────────────
mkdir -p results

if [ "$EVAL_ONLY" = true ]; then
    echo "[ANALYSIS] Running cross-scale analysis on existing results..."
    run_docker python3 scripts/analyze.py results/
    exit 0
fi

START_TIME=$(date +%s)

for SIZE in "${SELECTED_MODELS[@]}"; do
    MODEL_ID="${MODELS[$SIZE]}"
    TAG=$(echo "$MODEL_ID" | sed 's|Qwen/||' | tr '[:upper:]' '[:lower:]')

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Processing: $MODEL_ID"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # ── Step 1: Train ──
    if [ "$SKIP_TRAIN" = false ]; then
        echo "[1/4] Training $MODEL_ID ($EPOCHS epochs)..."
        run_docker python3 scripts/train.py \
            --dataset "$DATASET" \
            --model "$MODEL_ID" \
            --epochs "$EPOCHS" \
            --output output

        RUN_DIR=$(find_latest_run "$TAG")
        echo "  Training complete: $RUN_DIR"
    else
        RUN_DIR=$(find_latest_run "$TAG")
        if [ -z "$RUN_DIR" ]; then
            echo "  SKIP: No training run found for $TAG, skipping..."
            continue
        fi
        echo "  Using existing run: $RUN_DIR"
    fi

    # ── Step 2: Merge ──
    MERGED_DIR="merged_${TAG}"
    if [ ! -f "$MERGED_DIR/MERGE_COMPLETE" ] || [ "$SKIP_TRAIN" = false ]; then
        echo "[2/4] Merging adapter → $MERGED_DIR..."
        run_docker python3 scripts/merge.py \
            --adapter "$RUN_DIR" \
            --base "$MODEL_ID" \
            --output "$MERGED_DIR"
        echo "  Merge complete."
    else
        echo "  Using existing merged model: $MERGED_DIR"
    fi

    # ── Step 3: Eval base model ──
    if [ "$SKIP_BASE_EVAL" = false ]; then
        BASE_LABEL="${TAG}-base"
        echo "[3/4] Evaluating base model: $MODEL_ID → $BASE_LABEL..."
        run_docker python3 scripts/evaluate.py \
            --model "$MODEL_ID" \
            --label "$BASE_LABEL" \
            --output "results/eval_${BASE_LABEL}.json"
        echo "  Base eval complete."
    else
        echo "  Skipping base model evaluation."
    fi

    # ── Step 4: Eval fine-tuned model ──
    FT_LABEL="${TAG}-ft"
    echo "[4/4] Evaluating fine-tuned model: $MERGED_DIR → $FT_LABEL..."
    run_docker python3 scripts/evaluate.py \
        --model "$MERGED_DIR" \
        --label "$FT_LABEL" \
        --output "results/eval_${FT_LABEL}.json"
    echo "  Fine-tuned eval complete."

    echo ""
    echo "  ✓ $MODEL_ID pipeline complete."
    echo ""
done

# ── Step 5: Cross-scale analysis ────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Running Cross-Scale Analysis"
echo "╚══════════════════════════════════════════════════════════╝"

run_docker python3 scripts/analyze.py results/

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "  Total time: ${ELAPSED} minutes"
echo "  Results:    results/"
echo "  Analysis:   results/cross_scale_analysis.json"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Output files:"
ls -la results/eval_*.json results/cross_scale_analysis.json 2>/dev/null
echo ""
