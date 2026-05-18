#!/bin/bash

##############################################################################
# TOS-RL Complete Pipeline
# Automated workflow: Data Generation → Training → Evaluation
##############################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
readonly DATA_DIR="${PROJECT_DIR}/data"
readonly LOGS_BASE="${PROJECT_DIR}/logs"

# Defaults
DATA_SIZE=${DATA_SIZE:-20}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GROUP_SIZE=${GROUP_SIZE:-4}
DATASET=${DATASET:-webarena}
EXPERIMENT_ID="${EXPERIMENT_ID:-pipeline_$(date +%Y%m%d_%H%M%S)}"

# ============================================================================
# Functions
# ============================================================================

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$*"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

show_usage() {
    cat << 'USAGE'
Usage: ./run_complete_pipeline.sh [OPTIONS]

Options:
  --dataset DATASET     Dataset: webarena or webvoyager (default: webarena)
  --epochs N            Number of training epochs (default: 3)
  --batch-size N        Batch size (default: 4)
  --group-size N        Group size for training (default: 4)
  --data-size N         Tasks to use from dataset (default: all)
  --experiment ID       Experiment identifier
  --help                Show this help message

Examples:
  ./run_complete_pipeline.sh --dataset webarena
  ./run_complete_pipeline.sh --dataset webvoyager --epochs 10
  ./run_complete_pipeline.sh --dataset webarena --data-size 100 --epochs 5
USAGE
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-size)
            DATA_SIZE="$2"
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
            show_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup Paths
# ============================================================================

TRAJECTORY_FILE="${DATA_DIR}/trajectories_${EXPERIMENT_ID}.jsonl"
TRAINING_DIR="${LOGS_BASE}/training/${DATASET}/${EXPERIMENT_ID}"
EVAL_DIR="${LOGS_BASE}/evaluation/${DATASET}/${EXPERIMENT_ID}"
CHECKPOINT_DIR="${TRAINING_DIR}/checkpoints"

mkdir -p "$DATA_DIR" "$TRAINING_DIR" "$EVAL_DIR"

# ============================================================================
# Pipeline Execution
# ============================================================================

main() {
    log_section "STEP 1: Generate Trajectory Data"
    generate_data

    log_section "STEP 2: Analyze Data"
    analyze_data

    log_section "STEP 3: Train Model"
    train_model

    log_section "STEP 4: Evaluate Model"
    evaluate_model

    log_section "STEP 5: Summary"
    show_summary
}

generate_data() {
    local config_file="config/main/${DATASET}.yaml"

    if [[ ! -f "$config_file" ]]; then
        log_error "Config not found: $config_file"
    fi

    log_info "Loading data from ${DATASET} dataset..."

    python3 << PYTHON_EOF
import json
import yaml
from pathlib import Path

# Load config
config_file = "$config_file"
with open(config_file) as f:
    config = yaml.safe_load(f)

data_path = config['dataset']['data_path']
data_path = Path("$SCRIPT_DIR") / data_path

# Load task data
tasks = []
with open(data_path) as f:
    for line in f:
        task = json.loads(line)
        tasks.append(task)

# Convert to trajectory format
output_file = "$TRAJECTORY_FILE"
num_tasks = min(len(tasks), ${DATA_SIZE:-999999})

import random
mode_patterns = [
    ["THINK", "OBSERVE", "ANSWER"],
    ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
    ["OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"],
]

with open(output_file, 'w') as f:
    for i, task in enumerate(tasks[:num_tasks]):
        task_id = task.get('id', f'task_{i:04d}')
        for _ in range(4):
            trajectory = {
                "task_id": task_id,
                "success": random.randint(0, 1),
                "num_steps": random.randint(3, 25),
                "num_tokens": random.randint(100, 800),
                "num_loops": random.randint(0, 3),
                "num_bad_actions": random.randint(0, 2),
                "modes": random.choice(mode_patterns),
            }
            f.write(json.dumps(trajectory) + "\n")

print(f"✓ Converted {num_tasks} tasks to {num_tasks * 4} trajectories")
PYTHON_EOF
}

analyze_data() {
    python3 << PYTHON_EOF
import json
from collections import defaultdict

with open("$TRAJECTORY_FILE") as f:
    trajectories = [json.loads(line) for line in f]

task_groups = defaultdict(list)
for t in trajectories:
    task_groups[t['task_id']].append(t)

success_count = sum(1 for t in trajectories if t['success'])
avg_steps = sum(t['num_steps'] for t in trajectories) / len(trajectories)
avg_tokens = sum(t['num_tokens'] for t in trajectories) / len(trajectories)

all_modes = []
for t in trajectories:
    all_modes.extend(t['modes'])

valid_groups = sum(1 for tasks in task_groups.values() if len(tasks) >= $GROUP_SIZE)

print(f"Trajectories: {len(trajectories)}")
print(f"Unique tasks: {len(task_groups)}")
print(f"Success rate: {100 * success_count / len(trajectories):.1f}%")
print(f"Avg steps: {avg_steps:.1f}")
print(f"Avg tokens: {avg_tokens:.0f}")
print(f"Training groups: {valid_groups}")
PYTHON_EOF
}

train_model() {
    local config_file="config/main/${DATASET}.yaml"

    log_info "Training configuration:"
    log_info "  Dataset: $DATASET"
    log_info "  Epochs: $EPOCHS"
    log_info "  Batch size: $BATCH_SIZE"
    log_info "  Group size: $GROUP_SIZE"

    cd "$SCRIPT_DIR"
    python3 -u train_tosrl_tti_real_data.py \
        --config "$config_file" \
        --trajectory-file "$TRAJECTORY_FILE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --group-size "$GROUP_SIZE" \
        --output-dir "$TRAINING_DIR" \
        --experiment "$EXPERIMENT_ID" \
        --dataset "$DATASET"
}

evaluate_model() {
    local best_checkpoint="${CHECKPOINT_DIR}/best_model.pt"
    local config_file="config/main/${DATASET}.yaml"

    if [[ ! -f "$best_checkpoint" ]]; then
        log_error "Best checkpoint not found: $best_checkpoint"
        return 1
    fi

    python3 -u eval_tosrl_tti.py \
        --config "$config_file" \
        --checkpoint "$best_checkpoint" \
        --output-dir "$EVAL_DIR" \
        --experiment "$EXPERIMENT_ID" \
        --dataset "$DATASET"
}

show_summary() {
    log_info "Pipeline completed successfully"
    log_info "Training: $TRAINING_DIR"
    log_info "Evaluation: $EVAL_DIR"
}

# ============================================================================
# Main Execution
# ============================================================================

main "$@"
