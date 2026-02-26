#!/bin/bash
# run_pipeline.sh — End-to-end pipeline: train → validate checkpoint → evaluate → plot
#
# Usage:
#   bash run_pipeline.sh
#   EXPERIMENT=stop_grad BATCH_SIZE=32 LR=1e-5 bash run_pipeline.sh
#
# Overridable environment variables (all have defaults):
#   EXPERIMENT        Hydra experiment config  (default: multi_loss)
#   BATCH_SIZE        Per-device training batch size  (default: 64)
#   LR                BERT learning rate  (default: 3e-5)
#   GNN_LR            GNN learning rate  (default: 1e-4)
#   LAMBDA_ALIGN      Alignment loss weight λ  (default: 0.1)
#   MU_GNN            GNN contrastive loss weight μ  (default: 0.05)
#   NUM_EPOCHS        Training epochs  (default: 3)
#   OUTPUT_DIR        Checkpoint output dir  (default: outputs/<EXPERIMENT>)
#   EVAL_RESULTS_DIR  Evaluation outputs dir  (default: eval_results)
#   PLOTS_DIR         Plot outputs dir  (default: plots)
#   LOG_FILE          Log file path  (default: logs/pipeline_<timestamp>.log)

set -euo pipefail

# ==============================================================================
# Configuration — all values are overridable via environment variables
# ==============================================================================

EXPERIMENT="${EXPERIMENT:-multi_loss}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-5}"
GNN_LR="${GNN_LR:-1e-4}"
LAMBDA_ALIGN="${LAMBDA_ALIGN:-0.1}"
MU_GNN="${MU_GNN:-0.05}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXPERIMENT}}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-eval_results}"
PLOTS_DIR="${PLOTS_DIR:-plots}"
LOG_FILE="${LOG_FILE:-logs/pipeline_${TIMESTAMP}.log}"

# Resolve the script's own directory so it can be called from any working dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Logging setup — tee all output to log file + terminal
# ==============================================================================

mkdir -p "$(dirname "${LOG_FILE}")" "${EVAL_RESULTS_DIR}" "${PLOTS_DIR}"

# Redirect all subsequent stdout and stderr through tee into the log file
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# On any error, print context before exiting
trap 'log "ERROR: Pipeline failed at line ${LINENO} (exit code $?). See ${LOG_FILE} for details."' ERR

log "================================================================"
log "GNN-Syntax-BERT Pipeline — ${TIMESTAMP}"
log "  EXPERIMENT    = ${EXPERIMENT}"
log "  BATCH_SIZE    = ${BATCH_SIZE}"
log "  LR            = ${LR}"
log "  GNN_LR        = ${GNN_LR}"
log "  LAMBDA_ALIGN  = ${LAMBDA_ALIGN}"
log "  MU_GNN        = ${MU_GNN}"
log "  NUM_EPOCHS    = ${NUM_EPOCHS}"
log "  OUTPUT_DIR    = ${OUTPUT_DIR}"
log "  LOG_FILE      = ${LOG_FILE}"
log "================================================================"

# ==============================================================================
# Step 0: Environment activation
# Replicates activate_env.sh: sets UV_PROJECT_ENVIRONMENT so uv run uses the
# centralized venv at /Data/edouard.rabasse/venvs/CS224N_project
# ==============================================================================

log "--- Step 0: Environment setup ---"

VENV_BASE_DIR="/Data/edouard.rabasse/venvs"
PROJECT_NAME="CS224N_project"
export UV_PROJECT_ENVIRONMENT="${VENV_BASE_DIR}/${PROJECT_NAME}"

log "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"

if ! command -v uv &>/dev/null; then
    log "ERROR: 'uv' not found on PATH. Add ~/.local/bin to PATH or install uv."
    exit 1
fi

log "uv found at: $(command -v uv)"
log "Environment ready (UV_PROJECT_ENVIRONMENT set; uv sync not re-run to save time)"

# ==============================================================================
# Step 1: Training
# ==============================================================================

log "--- Step 1: Training ---"
log "Launching: uv run python src/train.py"

cd "${SCRIPT_DIR}"

uv run python src/train.py \
    --config-name config \
    "experiment=${EXPERIMENT}" \
    "training.per_device_train_batch_size=${BATCH_SIZE}" \
    "training.learning_rate=${LR}" \
    "training.gnn_learning_rate=${GNN_LR}" \
    "training.num_train_epochs=${NUM_EPOCHS}" \
    "model.lambda_align=${LAMBDA_ALIGN}" \
    "model.mu_gnn=${MU_GNN}" \
    "training.output_dir=${OUTPUT_DIR}" \
    "training.load_best_model_at_end=true" \
    "training.metric_for_best_model=stsb_spearman"

log "Training complete."

# ==============================================================================
# Step 2: Checkpoint validation
# ==============================================================================

log "--- Step 2: Checkpoint validation ---"

# config.yaml is the definitive "training completed" marker written by train.py
CHECKPOINT_EXISTS=0
if [[ -f "${OUTPUT_DIR}/config.yaml" ]]; then
    CHECKPOINT_EXISTS=1
    log "Found training marker: ${OUTPUT_DIR}/config.yaml"
fi

# Also accept any checkpoint subdirectory
if compgen -G "${OUTPUT_DIR}/checkpoint-*" &>/dev/null; then
    CHECKPOINT_EXISTS=1
    log "Found checkpoint subdirectories under ${OUTPUT_DIR}/"
fi

if [[ "${CHECKPOINT_EXISTS}" -eq 0 ]]; then
    log "ERROR: No checkpoint found at '${OUTPUT_DIR}'. Training may have failed silently."
    exit 1
fi

# Determine the best checkpoint path:
# - When load_best_model_at_end=true, HF Trainer copies the best model files
#   directly into OUTPUT_DIR (pytorch_model.bin or model.safetensors).
# - Fall back to the numerically latest checkpoint-* subdirectory.
BEST_CHECKPOINT="${OUTPUT_DIR}"

if [[ ! -f "${OUTPUT_DIR}/pytorch_model.bin" && ! -f "${OUTPUT_DIR}/model.safetensors" ]]; then
    # Best model was not copied to OUTPUT_DIR — find the latest checkpoint subdir
    LATEST_CKPT=$(compgen -G "${OUTPUT_DIR}/checkpoint-*" | sort -V | tail -1 || true)
    if [[ -n "${LATEST_CKPT}" ]]; then
        BEST_CHECKPOINT="${LATEST_CKPT}"
        log "Best model not in ${OUTPUT_DIR} directly; using latest checkpoint: ${BEST_CHECKPOINT}"
    else
        log "WARNING: No model weights found in ${OUTPUT_DIR}. Proceeding with output dir as checkpoint path."
    fi
else
    log "Best model found directly in ${OUTPUT_DIR}"
fi

log "Checkpoint resolved: ${BEST_CHECKPOINT}"

# ==============================================================================
# Step 3: Evaluation
# ==============================================================================

log "--- Step 3: Evaluation ---"

EVAL_OUTPUT_FILE="${EVAL_RESULTS_DIR}/eval_${EXPERIMENT}_${TIMESTAMP}.json"

log "Evaluating checkpoint: ${BEST_CHECKPOINT}"
log "Evaluation output:     ${EVAL_OUTPUT_FILE}"
log "Tasks: STS + syntactic probing (TreeDepth, BShift, TopConst)"

uv run python scripts/evaluation.py \
    --model_name_or_path "${BEST_CHECKPOINT}" \
    --pooler cls_before_pooler \
    --task_set sts syntactic_probing \
    --probing_tasks TreeDepth BShift TopConst \
    --output_file "${EVAL_OUTPUT_FILE}"

if [[ ! -f "${EVAL_OUTPUT_FILE}" ]]; then
    log "ERROR: Evaluation did not produce output file: ${EVAL_OUTPUT_FILE}"
    exit 1
fi

log "Evaluation complete. Results: ${EVAL_OUTPUT_FILE}"

# ==============================================================================
# Step 4: Plot generation
# ==============================================================================

log "--- Step 4: Plotting ---"

uv run python scripts/plot_results.py \
    --results_file "${EVAL_OUTPUT_FILE}" \
    --output_dir "${PLOTS_DIR}" \
    --experiment_name "${EXPERIMENT}"

log "Plots saved to: ${PLOTS_DIR}/"

# ==============================================================================
# Summary
# ==============================================================================

log "================================================================"
log "Pipeline complete."
log "  Checkpoint : ${BEST_CHECKPOINT}"
log "  Eval JSON  : ${EVAL_OUTPUT_FILE}"
log "  Plots      : ${PLOTS_DIR}/"
log "  Full log   : ${LOG_FILE}"
log "================================================================"

#!/bin/bash
# run_pipeline.sh — End-to-end pipeline: train → validate checkpoint → evaluate → plot
#
# Usage:
#   bash run_pipeline.sh
#   EXPERIMENT=stop_grad BATCH_SIZE=32 LR=1e-5 bash run_pipeline.sh
#
# Overridable environment variables (all have defaults):
#   EXPERIMENT        Hydra experiment config  (default: multi_loss)
#   BATCH_SIZE        Per-device training batch size  (default: 64)
#   LR                BERT learning rate  (default: 3e-5)
#   GNN_LR            GNN learning rate  (default: 1e-4)
#   LAMBDA_ALIGN      Alignment loss weight λ  (default: 0.1)
#   MU_GNN            GNN contrastive loss weight μ  (default: 0.05)
#   NUM_EPOCHS        Training epochs  (default: 3)
#   OUTPUT_DIR        Checkpoint output dir  (default: outputs/<EXPERIMENT>)
#   EVAL_RESULTS_DIR  Evaluation outputs dir  (default: eval_results)
#   PLOTS_DIR         Plot outputs dir  (default: plots)
#   LOG_FILE          Log file path  (default: logs/pipeline_<timestamp>.log)

set -euo pipefail

# ==============================================================================
# Configuration — all values are overridable via environment variables
# ==============================================================================

EXPERIMENT="${EXPERIMENT:-multi_loss}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-5}"
GNN_LR="${GNN_LR:-1e-4}"
LAMBDA_ALIGN="${LAMBDA_ALIGN:-0.1}"
MU_GNN="${MU_GNN:-0.05}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXPERIMENT}}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-eval_results}"
PLOTS_DIR="${PLOTS_DIR:-plots}"
LOG_FILE="${LOG_FILE:-logs/pipeline_${TIMESTAMP}.log}"

# Resolve the script's own directory so it can be called from any working dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Logging setup — tee all output to log file + terminal
# ==============================================================================

mkdir -p "$(dirname "${LOG_FILE}")" "${EVAL_RESULTS_DIR}" "${PLOTS_DIR}"

# Redirect all subsequent stdout and stderr through tee into the log file
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# On any error, print context before exiting
trap 'log "ERROR: Pipeline failed at line ${LINENO} (exit code $?). See ${LOG_FILE} for details."' ERR

log "================================================================"
log "GNN-Syntax-BERT Pipeline — ${TIMESTAMP}"
log "  EXPERIMENT    = ${EXPERIMENT}"
log "  BATCH_SIZE    = ${BATCH_SIZE}"
log "  LR            = ${LR}"
log "  GNN_LR        = ${GNN_LR}"
log "  LAMBDA_ALIGN  = ${LAMBDA_ALIGN}"
log "  MU_GNN        = ${MU_GNN}"
log "  NUM_EPOCHS    = ${NUM_EPOCHS}"
log "  OUTPUT_DIR    = ${OUTPUT_DIR}"
log "  LOG_FILE      = ${LOG_FILE}"
log "================================================================"

# ==============================================================================
# Step 0: Environment activation
# Replicates activate_env.sh: sets UV_PROJECT_ENVIRONMENT so uv run uses the
# centralized venv at /Data/edouard.rabasse/venvs/CS224N_project
# ==============================================================================

log "--- Step 0: Environment setup ---"

VENV_BASE_DIR="/Data/edouard.rabasse/venvs"
PROJECT_NAME="CS224N_project"
export UV_PROJECT_ENVIRONMENT="${VENV_BASE_DIR}/${PROJECT_NAME}"

log "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"

if ! command -v uv &>/dev/null; then
    log "ERROR: 'uv' not found on PATH. Add ~/.local/bin to PATH or install uv."
    exit 1
fi

log "uv found at: $(command -v uv)"
log "Environment ready (UV_PROJECT_ENVIRONMENT set; uv sync not re-run to save time)"

# ==============================================================================
# Step 1: Training
# ==============================================================================

log "--- Step 1: Training ---"
log "Launching: uv run python src/train.py"

cd "${SCRIPT_DIR}"

uv run python src/train.py \
    --config-name config \
    "experiment=${EXPERIMENT}" \
    "training.per_device_train_batch_size=${BATCH_SIZE}" \
    "training.learning_rate=${LR}" \
    "training.gnn_learning_rate=${GNN_LR}" \
    "training.num_train_epochs=${NUM_EPOCHS}" \
    "model.alignment.lambda_align=${LAMBDA_ALIGN}" \
    "model.alignment.mu_gnn=${MU_GNN}" \
    "training.output_dir=${OUTPUT_DIR}" \
    "training.load_best_model_at_end=true" \
    "training.metric_for_best_model=stsb_spearman"

log "Training complete."

# ==============================================================================
# Step 2: Checkpoint validation
# ==============================================================================

log "--- Step 2: Checkpoint validation ---"

# config.yaml is the definitive "training completed" marker written by train.py
CHECKPOINT_EXISTS=0
if [[ -f "${OUTPUT_DIR}/config.yaml" ]]; then
    CHECKPOINT_EXISTS=1
    log "Found training marker: ${OUTPUT_DIR}/config.yaml"
fi

# Also accept any checkpoint subdirectory
if compgen -G "${OUTPUT_DIR}/checkpoint-*" &>/dev/null; then
    CHECKPOINT_EXISTS=1
    log "Found checkpoint subdirectories under ${OUTPUT_DIR}/"
fi

if [[ "${CHECKPOINT_EXISTS}" -eq 0 ]]; then
    log "ERROR: No checkpoint found at '${OUTPUT_DIR}'. Training may have failed silently."
    exit 1
fi

# Determine the best checkpoint path:
# - When load_best_model_at_end=true, HF Trainer copies the best model files
#   directly into OUTPUT_DIR (pytorch_model.bin or model.safetensors).
# - Fall back to the numerically latest checkpoint-* subdirectory.
BEST_CHECKPOINT="${OUTPUT_DIR}"

if [[ ! -f "${OUTPUT_DIR}/pytorch_model.bin" && ! -f "${OUTPUT_DIR}/model.safetensors" ]]; then
    # Best model was not copied to OUTPUT_DIR — find the latest checkpoint subdir
    LATEST_CKPT=$(compgen -G "${OUTPUT_DIR}/checkpoint-*" | sort -V | tail -1 || true)
    if [[ -n "${LATEST_CKPT}" ]]; then
        BEST_CHECKPOINT="${LATEST_CKPT}"
        log "Best model not in ${OUTPUT_DIR} directly; using latest checkpoint: ${BEST_CHECKPOINT}"
    else
        log "WARNING: No model weights found in ${OUTPUT_DIR}. Proceeding with output dir as checkpoint path."
    fi
else
    log "Best model found directly in ${OUTPUT_DIR}"
fi

log "Checkpoint resolved: ${BEST_CHECKPOINT}"

# ==============================================================================
# Step 3: Evaluation
# ==============================================================================

log "--- Step 3: Evaluation ---"

EVAL_OUTPUT_FILE="${EVAL_RESULTS_DIR}/eval_${EXPERIMENT}_${TIMESTAMP}.json"

log "Evaluating checkpoint: ${BEST_CHECKPOINT}"
log "Evaluation output:     ${EVAL_OUTPUT_FILE}"
log "Tasks: STS + syntactic probing (TreeDepth, BShift, TopConst)"

uv run python scripts/evaluation.py \
    --model_name_or_path "${BEST_CHECKPOINT}" \
    --pooler cls_before_pooler \
    --task_set sts syntactic_probing \
    --probing_tasks TreeDepth BShift TopConst \
    --output_file "${EVAL_OUTPUT_FILE}"

if [[ ! -f "${EVAL_OUTPUT_FILE}" ]]; then
    log "ERROR: Evaluation did not produce output file: ${EVAL_OUTPUT_FILE}"
    exit 1
fi

log "Evaluation complete. Results: ${EVAL_OUTPUT_FILE}"

# ==============================================================================
# Step 4: Plot generation
# ==============================================================================

log "--- Step 4: Plotting ---"

uv run python scripts/plot_results.py \
    --results_file "${EVAL_OUTPUT_FILE}" \
    --output_dir "${PLOTS_DIR}" \
    --experiment_name "${EXPERIMENT}"

log "Plots saved to: ${PLOTS_DIR}/"

# ==============================================================================
# Summary
# ==============================================================================

log "================================================================"
log "Pipeline complete."
log "  Checkpoint : ${BEST_CHECKPOINT}"
log "  Eval JSON  : ${EVAL_OUTPUT_FILE}"
log "  Plots      : ${PLOTS_DIR}/"
log "  Full log   : ${LOG_FILE}"
log "================================================================"
