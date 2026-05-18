#!/bin/bash
#
# Real Data Training Script for TOS-RL with TTI
# Loads trajectories from JSONL file instead of mock data
# Reads configuration from scripts/config/main/webarena_rl.yaml
#
# Usage:
#   ./run_train_real_data.sh --trajectory-file data/trajectories.jsonl
#   ./run_train_real_data.sh --trajectory-file data/traj.jsonl --epochs 20
#   ./run_train_real_data.sh --trajectory-file data/traj.jsonl --experiment my_real_training
#   ./run_train_real_data.sh --help
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
EXPERIMENT_ID="tosrl_real_data_$(date +%Y%m%d_%H%M%S)"
TRAJECTORY_FILE=""
EPOCHS=10
BATCH_SIZE=4
GRAD_ACCUM_STEPS=2
EVAL_FREQ=1
SAVE_FREQ=1
MODE="tosrl_train"
USE_BRANCHING=false
MODE_WEIGHT=2.0
GROUP_SIZE=4

# Print help
print_help() {
    cat << EOF
${BLUE}TOS-RL Real Data Training${NC}

USAGE:
  ./run_train_real_data.sh --trajectory-file <path> [OPTIONS]

REQUIRED:
  --trajectory-file FILE    Path to JSONL trajectory file (required)

OPTIONS:
  --config NAME             Config file name (default: webarena_rl)
  --dataset NAME            Dataset name (default: webarena)
  --experiment ID           Experiment identifier
  --epochs N                Number of epochs (default: 10)
  --batch-size N            Batch size (default: 4)
  --grad-accum N            Gradient accumulation steps (default: 2)
  --group-size N            Group size (default: 4)
  --mode-weight N           Mode token weight (default: 2.0)
  --branching               Enable prefix branching
  --help, -h                Show this help message

EXAMPLES:
  # Basic training
  ./run_train_real_data.sh --trajectory-file data/webarena_trajectories.jsonl

  # With custom epochs and experiment name
  ./run_train_real_data.sh \\
    --trajectory-file data/trajectories.jsonl \\
    --epochs 20 \\
    --experiment my_training

  # Full customization
  ./run_train_real_data.sh \\
    --trajectory-file data/traj.jsonl \\
    --dataset webarena \\
    --epochs 15 \\
    --batch-size 8 \\
    --group-size 4 \\
    --branching

TRAJECTORY FILE FORMAT (JSONL):
  Each line must be valid JSON with these fields:
  {
    "task_id": "task_001",
    "success": 1,
    "num_steps": 5,
    "num_tokens": 250,
    "num_loops": 0,
    "num_bad_actions": 0,
    "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]
  }

See TOSRL_REAL_DATA_GUIDE.md for complete documentation.

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trajectory-file)
            TRAJECTORY_FILE="$2"
            shift 2
            ;;
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
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --mode-weight)
            MODE_WEIGHT="$2"
            shift 2
            ;;
        --branching)
            USE_BRANCHING=true
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TRAJECTORY_FILE" ]; then
    echo -e "${RED}Error: --trajectory-file is required${NC}"
    echo ""
    print_help
    exit 1
fi

# Check trajectory file exists
if [ ! -f "$TRAJECTORY_FILE" ]; then
    echo -e "${RED}Error: Trajectory file not found: $TRAJECTORY_FILE${NC}"
    echo ""
    echo "Make sure to:"
    echo "  1. Collect trajectories from your TTI agent"
    echo "  2. Format as JSONL (one JSON per line)"
    echo "  3. Include required fields: task_id, success, num_steps, num_tokens, modes"
    echo ""
    echo "See TOSRL_REAL_DATA_GUIDE.md for details"
    exit 1
fi

# Get absolute path for trajectory file
TRAJECTORY_FILE="$(cd "$(dirname "$TRAJECTORY_FILE")" && pwd)/$(basename "$TRAJECTORY_FILE")"

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
    echo -e "${BLUE}TOS-RL REAL DATA TRAINING${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Config File:     $CONFIG_FILE"
    echo "Trajectory File: $TRAJECTORY_FILE"
    echo "Experiment ID:   $EXPERIMENT_ID"
    echo "Dataset:         $DATASET"
    echo "Epochs:          $EPOCHS"
    echo "Batch Size:      $BATCH_SIZE"
    echo "Grad Accum:      $GRAD_ACCUM_STEPS"
    echo "Group Size:      $GROUP_SIZE"
    echo "Mode Weight:     $MODE_WEIGHT"
    echo "Branching:       $USE_BRANCHING"
    echo -e "${BLUE}========================================${NC}"
}

print_config

# Validate trajectory file format
echo -e "${YELLOW}Validating trajectory file...${NC}"
python3 << EOF
import json
import sys

trajectory_file = "$TRAJECTORY_FILE"
try:
    with open(trajectory_file, 'r') as f:
        count = 0
        for line_num, line in enumerate(f, 1):
            try:
                traj = json.loads(line)
                required_fields = {'task_id', 'success', 'num_steps', 'num_tokens', 'modes'}
                if not all(field in traj for field in required_fields):
                    missing = required_fields - set(traj.keys())
                    print(f"Error on line {line_num}: Missing fields {missing}")
                    sys.exit(1)
                count += 1
            except json.JSONDecodeError as e:
                print(f"Error on line {line_num}: Invalid JSON - {e}")
                sys.exit(1)

    print(f"✓ Valid JSONL file with {count} trajectories")
except Exception as e:
    print(f"Error reading trajectory file: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Trajectory file validation failed${NC}"
    exit 1
fi

# Create logs directory
LOG_DIR="$PROJECT_DIR/logs/training/${DATASET}/${EXPERIMENT_ID}"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/training.log"

echo -e "${GREEN}Logging to: $LOGFILE${NC}\n"

# Start training
echo "Starting TOS-RL real data training..."
echo "Start time: $(date)" | tee -a "$LOGFILE"
echo "Trajectory file: $TRAJECTORY_FILE" | tee -a "$LOGFILE"

python3 -u "$SCRIPT_DIR/train_tosrl_tti_real_data.py" \
    --config "$CONFIG_FILE" \
    --experiment "$EXPERIMENT_ID" \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --data-source file \
    --trajectory-file "$TRAJECTORY_FILE" \
    --group-size "$GROUP_SIZE" \
    --mode-weight "$MODE_WEIGHT" \
    $([ "$USE_BRANCHING" = true ] && echo "--branching") \
    --output-dir "$LOG_DIR" \
    2>&1 | tee -a "$LOGFILE"

TRAIN_EXIT=$?

echo -e "\n${BLUE}========================================${NC}" | tee -a "$LOGFILE"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Real data training completed successfully!${NC}" | tee -a "$LOGFILE"
    echo "Model saved to: $LOG_DIR" | tee -a "$LOGFILE"
else
    echo -e "${RED}✗ Training failed with exit code $TRAIN_EXIT${NC}" | tee -a "$LOGFILE"
    exit $TRAIN_EXIT
fi

echo "End time: $(date)" | tee -a "$LOGFILE"
echo -e "${BLUE}========================================${NC}\n" | tee -a "$LOGFILE"

# Print summary
echo -e "${GREEN}Training Summary:${NC}"
echo "  Logs:         $LOGFILE"
echo "  Checkpoints:  $LOG_DIR/checkpoints"
echo "  Results:      $LOG_DIR/results.json"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Monitor training: tail -f $LOGFILE"
echo "  2. Evaluate: ./run_eval.sh --checkpoint $LOG_DIR/checkpoints/best_model.pt"
echo "  3. Compare: python3 scripts/analyze_pareto.py --results $LOG_DIR"
echo ""
