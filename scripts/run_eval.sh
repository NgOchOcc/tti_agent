#!/bin/bash

##############################################################################
# TOS-RL Evaluation Script
##############################################################################

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
CHECKPOINT=""
OUTPUT_DIR=""
EXPERIMENT_ID="evaluation"
DATASET="webarena"
NUM_TASKS=100
COST_PREFERENCE="balanced"
CONFIG=""

# ============================================================================
# Functions
# ============================================================================

log_error() {
    echo "[ERROR] $*" >&2
    exit 1
}

log_info() {
    echo "[INFO] $*"
}

show_usage() {
    cat << 'USAGE'
Usage: ./run_eval.sh [OPTIONS]

Required:
  --checkpoint FILE         Path to model checkpoint
  --output-dir DIR          Output directory for evaluation results

Options:
  --dataset NAME            Dataset: webarena or webvoyager (default: webarena)
  --experiment ID           Experiment identifier
  --cost-preference PREF    Cost preference (default: balanced)
                            Options: high_efficiency, balanced, high_success
  --num-tasks N             Number of tasks to evaluate (default: 100)
  --help                    Show this help message

Examples:
  ./run_eval.sh --checkpoint model.pt --output-dir logs/eval
  ./run_eval.sh --checkpoint model.pt --output-dir logs/eval --dataset webvoyager
USAGE
}

validate_args() {
    [[ -z "$CHECKPOINT" ]] && log_error "Missing: --checkpoint"
    [[ -z "$OUTPUT_DIR" ]] && log_error "Missing: --output-dir"
    [[ ! -f "$CHECKPOINT" ]] && log_error "Checkpoint not found: $CHECKPOINT"
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --cost-preference)
            COST_PREFERENCE="$2"
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
done

validate_args

# Set config based on dataset if not provided
if [[ -z "$CONFIG" ]]; then
    CONFIG="config/main/${DATASET}.yaml"
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Execute Evaluation
# ============================================================================

log_info "Starting evaluation..."
log_info "Dataset: $DATASET"
log_info "Config: $CONFIG"
log_info "Checkpoint: $CHECKPOINT"
log_info "Cost preference: $COST_PREFERENCE"
log_info "Output directory: $OUTPUT_DIR"

if [[ ! -f "$SCRIPT_DIR/$CONFIG" ]]; then
    log_error "Config file not found: $CONFIG"
fi

cd "$SCRIPT_DIR"

python3 -u eval_tosrl_tti.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --experiment "$EXPERIMENT_ID" \
    --dataset "$DATASET" \
    --cost-preference "$COST_PREFERENCE" \
    --num-tasks "$NUM_TASKS"

log_info "Evaluation completed"
