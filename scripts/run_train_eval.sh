#!/bin/bash
#
# Combined Training + Evaluation script for TOS-RL with TTI
# Trains the model and then evaluates it periodically
#
# Usage:
#   ./run_train_eval.sh                          # Default config
#   ./run_train_eval.sh --epochs 10 --eval-every 2
#   ./run_train_eval.sh --dataset webarena --mode tosrl_train
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$SCRIPT_DIR/config/main"

# Default values
TRAIN_CONFIG="webarena_rl"
EVAL_CONFIG="webarena_eval"
DATASET="webarena"
EXPERIMENT_ID="tosrl_train_eval_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10
EVAL_EVERY=2
BATCH_SIZE=4
GRAD_ACCUM_STEPS=2
NUM_EVAL_TASKS=50
TRAIN_MODE="tosrl_train"
EVAL_MODE="tosrl_eval"
USE_BRANCHING=false
MODE_WEIGHT=2.0
GROUP_SIZE=4
VERBOSE=false
PARETO_AT_END=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --eval-config)
            EVAL_CONFIG="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --eval-every)
            EVAL_EVERY="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --num-eval-tasks)
            NUM_EVAL_TASKS="$2"
            shift 2
            ;;
        --train-mode)
            TRAIN_MODE="$2"
            shift 2
            ;;
        --eval-mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        --mode-weight)
            MODE_WEIGHT="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --branching)
            USE_BRANCHING=true
            shift
            ;;
        --pareto)
            PARETO_AT_END=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Validate configs
for cfg in "$TRAIN_CONFIG" "$EVAL_CONFIG"; do
    if [ ! -f "$CONFIG_DIR/${cfg}.yaml" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_DIR/${cfg}.yaml${NC}"
        exit 1
    fi
done

# Print configuration
print_header() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           TOS-RL TRAINING + EVALUATION${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo "  Experiment ID:     $EXPERIMENT_ID"
    echo "  Dataset:           $DATASET"
    echo "  Train Config:      $TRAIN_CONFIG"
    echo "  Eval Config:       $EVAL_CONFIG"
    echo "  Total Epochs:      $EPOCHS"
    echo "  Eval Every:        $EVAL_EVERY epochs"
    echo "  Batch Size:        $BATCH_SIZE"
    echo "  Grad Accum:        $GRAD_ACCUM_STEPS"
    echo "  Group Size:        $GROUP_SIZE"
    echo "  Mode Weight:       $MODE_WEIGHT"
    echo "  Branching:         $USE_BRANCHING"
    echo "  Eval Tasks:        $NUM_EVAL_TASKS per eval"
    echo "  Pareto at End:     $PARETO_AT_END"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}\n"
}

print_header
print_config

# Create main logs directory
MAIN_LOG_DIR="$PROJECT_DIR/logs/train_eval/${DATASET}/${EXPERIMENT_ID}"
mkdir -p "$MAIN_LOG_DIR"
MAIN_LOGFILE="$MAIN_LOG_DIR/train_eval.log"

echo "Logging to: $MAIN_LOGFILE" | tee "$MAIN_LOGFILE"

# Create subdirectories
TRAIN_LOG_DIR="$MAIN_LOG_DIR/training"
EVAL_LOG_DIR="$MAIN_LOG_DIR/evaluation"
CHECKPOINTS_DIR="$MAIN_LOG_DIR/checkpoints"
mkdir -p "$TRAIN_LOG_DIR" "$EVAL_LOG_DIR" "$CHECKPOINTS_DIR"

echo "" | tee -a "$MAIN_LOGFILE"
echo "Directories:" | tee -a "$MAIN_LOGFILE"
echo "  Training:    $TRAIN_LOG_DIR" | tee -a "$MAIN_LOGFILE"
echo "  Evaluation:  $EVAL_LOG_DIR" | tee -a "$MAIN_LOGFILE"
echo "  Checkpoints: $CHECKPOINTS_DIR" | tee -a "$MAIN_LOGFILE"
echo "" | tee -a "$MAIN_LOGFILE"

# Training phase
train_phase() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${CYAN}║ TRAINING PHASE${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n" | tee -a "$MAIN_LOGFILE"

    "$SCRIPT_DIR/run_train.sh" \
        --config "$TRAIN_CONFIG" \
        --dataset "$DATASET" \
        --experiment "$EXPERIMENT_ID" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM_STEPS" \
        --mode "$TRAIN_MODE" \
        --group-size "$GROUP_SIZE" \
        --mode-weight "$MODE_WEIGHT" \
        $([ "$USE_BRANCHING" = true ] && echo "--branching") \
        2>&1 | tee -a "$MAIN_LOGFILE"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Training phase failed${NC}" | tee -a "$MAIN_LOGFILE"
        return 1
    fi

    # Copy checkpoints
    if [ -d "$TRAIN_LOG_DIR/checkpoints" ]; then
        cp -r "$TRAIN_LOG_DIR/checkpoints/"* "$CHECKPOINTS_DIR/" 2>/dev/null || true
    fi

    echo -e "${GREEN}✓ Training phase completed${NC}\n" | tee -a "$MAIN_LOGFILE"
    return 0
}

# Evaluation phase (after training)
eval_phase() {
    local checkpoint="$1"
    local eval_name="$2"
    local cost_pref="$3"

    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${CYAN}║ EVALUATION: $eval_name${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n" | tee -a "$MAIN_LOGFILE"

    "$SCRIPT_DIR/run_eval.sh" \
        --config "$EVAL_CONFIG" \
        --checkpoint "$checkpoint" \
        --dataset "$DATASET" \
        --experiment "${EXPERIMENT_ID}_${eval_name}" \
        --num-tasks "$NUM_EVAL_TASKS" \
        --cost "$cost_pref" \
        --mode "$EVAL_MODE" \
        2>&1 | tee -a "$MAIN_LOGFILE"

    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠ Evaluation ${eval_name} had issues but continuing...${NC}" | tee -a "$MAIN_LOGFILE"
        return 1
    fi

    echo -e "${GREEN}✓ Evaluation ${eval_name} completed${NC}\n" | tee -a "$MAIN_LOGFILE"
    return 0
}

# Pareto analysis phase (at end)
pareto_phase() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${CYAN}║ PARETO ANALYSIS${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n" | tee -a "$MAIN_LOGFILE"

    local best_checkpoint="$CHECKPOINTS_DIR/best_model.pt"

    if [ -f "$best_checkpoint" ]; then
        "$SCRIPT_DIR/run_eval.sh" \
            --config "$EVAL_CONFIG" \
            --checkpoint "$best_checkpoint" \
            --dataset "$DATASET" \
            --experiment "${EXPERIMENT_ID}_pareto" \
            --num-tasks "$NUM_EVAL_TASKS" \
            --pareto \
            --mode "$EVAL_MODE" \
            2>&1 | tee -a "$MAIN_LOGFILE"

        echo -e "${GREEN}✓ Pareto analysis completed${NC}\n" | tee -a "$MAIN_LOGFILE"
    else
        echo -e "${YELLOW}⚠ Best checkpoint not found, skipping Pareto analysis${NC}" | tee -a "$MAIN_LOGFILE"
    fi
}

# Main execution
main() {
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Start time: $start_time" | tee -a "$MAIN_LOGFILE"

    # Run training
    if ! train_phase; then
        echo -e "${RED}✗ Training failed. Aborting.${NC}" | tee -a "$MAIN_LOGFILE"
        return 1
    fi

    # Find best checkpoint
    local best_checkpoint=""
    if [ -f "$CHECKPOINTS_DIR/best_model.pt" ]; then
        best_checkpoint="$CHECKPOINTS_DIR/best_model.pt"
    else
        # Use latest checkpoint
        best_checkpoint=$(ls -t "$CHECKPOINTS_DIR"/*.pt 2>/dev/null | head -1)
    fi

    if [ -z "$best_checkpoint" ]; then
        echo -e "${YELLOW}⚠ No checkpoint found. Evaluation skipped.${NC}" | tee -a "$MAIN_LOGFILE"
        return 1
    fi

    echo "Using checkpoint: $best_checkpoint" | tee -a "$MAIN_LOGFILE"

    # Run final evaluation with multiple cost preferences
    if [ "$PARETO_AT_END" = true ]; then
        pareto_phase
    else
        eval_phase "$best_checkpoint" "final" "balanced"
    fi

    # Print final summary
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "" | tee -a "$MAIN_LOGFILE"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${GREEN}✓ TRAINING + EVALUATION COMPLETE${NC}" | tee -a "$MAIN_LOGFILE"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}" | tee -a "$MAIN_LOGFILE"
    echo "Start time: $start_time" | tee -a "$MAIN_LOGFILE"
    echo "End time:   $end_time" | tee -a "$MAIN_LOGFILE"
    echo "" | tee -a "$MAIN_LOGFILE"
    echo "Results:" | tee -a "$MAIN_LOGFILE"
    echo "  Main log:   $MAIN_LOGFILE" | tee -a "$MAIN_LOGFILE"
    echo "  Checkpoints: $CHECKPOINTS_DIR" | tee -a "$MAIN_LOGFILE"
    echo "  Eval results: $EVAL_LOG_DIR" | tee -a "$MAIN_LOGFILE"
    echo "" | tee -a "$MAIN_LOGFILE"

    return 0
}

# Run main
main
MAIN_EXIT=$?

exit $MAIN_EXIT

print_help() {
    cat << EOF
TOS-RL Training + Evaluation Script

Usage: $0 [OPTIONS]

Options:
  --train-config CONFIG        Training config (default: webarena_rl)
  --eval-config CONFIG         Evaluation config (default: webarena_eval)

  --dataset DATASET            Dataset name (default: webarena)
                               Options: webarena, webvoyager

  --experiment EXP_ID          Experiment ID (default: auto-generated)

  --epochs N                   Number of training epochs (default: 10)

  --eval-every N               Evaluate every N epochs (default: 2)

  --batch-size N               Batch size (default: 4)

  --num-eval-tasks N           Tasks per evaluation (default: 50)

  --train-mode MODE            Training mode (default: tosrl_train)

  --eval-mode MODE             Evaluation mode (default: tosrl_eval)

  --mode-weight N              Mode token weight (default: 2.0)

  --group-size N               Group size for TOS-RL (default: 4)

  --branching                  Enable prefix branching

  --pareto                     Run Pareto analysis at end

  --verbose                    Enable verbose output

  --help, -h                   Show this help message

Examples:
  # Default training + evaluation
  $0

  # 20 epochs with evaluation every epoch
  $0 --epochs 20 --eval-every 1

  # WebVoyager training + Pareto analysis
  $0 --dataset webvoyager --pareto

  # Custom experiment
  $0 --experiment "tosrl_v2" --branching --epochs 15

  # With custom hyperparameters
  $0 --epochs 10 --batch-size 8 --mode-weight 3.0 --group-size 4

Environment Variables:
  CUDA_VISIBLE_DEVICES   GPU IDs to use (e.g., "0,1,2,3")

EOF
}
