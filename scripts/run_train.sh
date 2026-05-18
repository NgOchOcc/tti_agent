#!/bin/bash
#
# Training script for TOS-RL with TTI
# Reads configuration from scripts/config/main/webarena_rl.yaml
#
# Usage:
#   ./run_train.sh                                    # Use default config
#   ./run_train.sh --config webarena_rl              # Explicit config
#   ./run_train.sh --epochs 10 --batch-size 8        # Override params
#   ./run_train.sh --experiment my_exp --dataset webarena
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$SCRIPT_DIR/config/main"

# Default values
CONFIG="webarena_rl"
DATASET="webarena"
EXPERIMENT_ID="tosrl_train_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10
BATCH_SIZE=4
GRAD_ACCUM_STEPS=2
EVAL_FREQ=1
SAVE_FREQ=1
MODE="tosrl_train"  # or "tti_train"
USE_BRANCHING=false
MODE_WEIGHT=2.0
GROUP_SIZE=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
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
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --eval-freq)
            EVAL_FREQ="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --branching)
            USE_BRANCHING=true
            shift
            ;;
        --mode-weight)
            MODE_WEIGHT="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
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

# Check if config exists
CONFIG_FILE="$CONFIG_DIR/${CONFIG}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo "Available configs:"
    ls -1 "$CONFIG_DIR"/*.yaml | xargs -n1 basename
    exit 1
fi

# Print configuration
print_config() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}TOS-RL TRAINING${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Config File:     $CONFIG_FILE"
    echo "Experiment ID:   $EXPERIMENT_ID"
    echo "Dataset:         $DATASET"
    echo "Mode:            $MODE"
    echo "Epochs:          $EPOCHS"
    echo "Batch Size:      $BATCH_SIZE"
    echo "Grad Accum:      $GRAD_ACCUM_STEPS"
    echo "Eval Freq:       $EVAL_FREQ"
    echo "Group Size:      $GROUP_SIZE"
    echo "Mode Weight:     $MODE_WEIGHT"
    echo "Branching:       $USE_BRANCHING"
    echo -e "${BLUE}========================================${NC}"
}

print_config

# Create logs directory
LOG_DIR="$PROJECT_DIR/logs/training/${DATASET}/${EXPERIMENT_ID}"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/training.log"

echo -e "${GREEN}Logging to: $LOGFILE${NC}\n"

# Start training
echo "Starting TOS-RL training..."
echo "Start time: $(date)" | tee -a "$LOGFILE"

python -u "$SCRIPT_DIR/train_tosrl_tti.py" \
    --config "$CONFIG_FILE" \
    --experiment "$EXPERIMENT_ID" \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --eval-freq "$EVAL_FREQ" \
    --save-freq "$SAVE_FREQ" \
    --mode "$MODE" \
    --group-size "$GROUP_SIZE" \
    --mode-weight "$MODE_WEIGHT" \
    $([ "$USE_BRANCHING" = true ] && echo "--branching") \
    --output-dir "$LOG_DIR" \
    2>&1 | tee -a "$LOGFILE"

TRAIN_EXIT=$?

echo -e "\n${BLUE}========================================${NC}" | tee -a "$LOGFILE"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}" | tee -a "$LOGFILE"
    echo "Model saved to: $LOG_DIR" | tee -a "$LOGFILE"
else
    echo -e "${RED}✗ Training failed with exit code $TRAIN_EXIT${NC}" | tee -a "$LOGFILE"
    exit $TRAIN_EXIT
fi

echo "End time: $(date)" | tee -a "$LOGFILE"
echo -e "${BLUE}========================================${NC}\n" | tee -a "$LOGFILE"

# Print summary
echo -e "${GREEN}Training Summary:${NC}"
echo "  Logs:     $LOGFILE"
echo "  Checkpoints: $LOG_DIR/checkpoints"
echo "  Results:  $LOG_DIR/results.json"
