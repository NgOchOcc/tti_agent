#!/bin/bash

##############################################################################
# TOS-RL Training Script with Real Data
##############################################################################

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
TRAJECTORY_FILE=""
EPOCHS=10
BATCH_SIZE=4
GROUP_SIZE=4
OUTPUT_DIR=""
EXPERIMENT_ID=""
DATASET="webarena"
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
Usage: ./run_train_real_data.sh [OPTIONS]

Required:
  --trajectory-file FILE    Path to JSONL trajectory file
  --output-dir DIR          Output directory for logs and checkpoints

Options:
  --dataset NAME            Dataset name: webarena or webvoyager (default: webarena)
  --epochs N                Number of training epochs (default: 10)
  --batch-size N            Batch size (default: 4)
  --group-size N            Group size (default: 4)
  --experiment ID           Experiment identifier
  --help                    Show this help message

Examples:
  ./run_train_real_data.sh --trajectory-file data.jsonl --output-dir logs/train
  ./run_train_real_data.sh --trajectory-file data.jsonl --output-dir logs/train --dataset webvoyager
USAGE
}

validate_args() {
    [[ -z "$TRAJECTORY_FILE" ]] && log_error "Missing: --trajectory-file"
    [[ -z "$OUTPUT_DIR" ]] && log_error "Missing: --output-dir"
    [[ ! -f "$TRAJECTORY_FILE" ]] && log_error "File not found: $TRAJECTORY_FILE"
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trajectory-file)
            TRAJECTORY_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
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
# Execute Training
# ============================================================================

log_info "Starting training..."
log_info "Dataset: $DATASET"
log_info "Config: $CONFIG"
log_info "Trajectory file: $TRAJECTORY_FILE"
log_info "Output directory: $OUTPUT_DIR"
log_info "Epochs: $EPOCHS"

if [[ ! -f "$SCRIPT_DIR/$CONFIG" ]]; then
    log_error "Config file not found: $CONFIG"
fi

cd "$SCRIPT_DIR"

python3 -u train_tosrl_tti_real_data.py \
    --config "$CONFIG" \
    --trajectory-file "$TRAJECTORY_FILE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --group-size "$GROUP_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --experiment "${EXPERIMENT_ID:-training}" \
    --dataset "$DATASET"

log_info "Training completed"
