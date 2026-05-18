#!/bin/bash
#
# Evaluation script for TOS-RL with TTI
# Reads configuration from scripts/config/main/webarena_eval.yaml
#
# Usage:
#   ./run_eval.sh --checkpoint best_model.pt
#   ./run_eval.sh --checkpoint checkpoints/tosrl_epoch_5.pt --cost balanced
#   ./run_eval.sh --config webarena_eval --dataset webarena
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
CONFIG="webarena_eval"
DATASET="webarena"
CHECKPOINT=""
EXPERIMENT_ID="tosrl_eval_$(date +%Y%m%d_%H%M%S)"
COST_PREFERENCE="balanced"  # high_efficiency, balanced, high_success
NUM_TASKS=100
BATCH_SIZE=4
GRAD_ACCUM_STEPS=2
MODE="tosrl_eval"  # or "tti_eval", "cvi_eval"
VERBOSE=false
SAVE_RESULTS=true
PARETO_ANALYSIS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
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
        --cost)
            COST_PREFERENCE="$2"
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --pareto)
            PARETO_ANALYSIS=true
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

# Validate checkpoint
if [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}Error: --checkpoint is required${NC}"
    print_help
    exit 1
fi

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
    echo -e "${BLUE}TOS-RL EVALUATION${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Config File:       $CONFIG_FILE"
    echo "Checkpoint:        $CHECKPOINT"
    echo "Experiment ID:     $EXPERIMENT_ID"
    echo "Dataset:           $DATASET"
    echo "Mode:              $MODE"
    echo "Cost Preference:   $COST_PREFERENCE"
    echo "Num Tasks:         $NUM_TASKS"
    echo "Batch Size:        $BATCH_SIZE"
    echo "Pareto Analysis:   $PARETO_ANALYSIS"
    echo "Verbose:           $VERBOSE"
    echo -e "${BLUE}========================================${NC}"
}

print_config

# Create logs directory
LOG_DIR="$PROJECT_DIR/logs/evaluation/${DATASET}/${EXPERIMENT_ID}"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/evaluation.log"

echo -e "${GREEN}Logging to: $LOGFILE${NC}\n"

# Start evaluation
echo "Starting TOS-RL evaluation..."
echo "Start time: $(date)" | tee -a "$LOGFILE"

# Run standard evaluation
if [ "$PARETO_ANALYSIS" = false ]; then
    python -u "$SCRIPT_DIR/eval_tosrl_tti.py" \
        --config "$CONFIG_FILE" \
        --checkpoint "$CHECKPOINT" \
        --experiment "$EXPERIMENT_ID" \
        --dataset "$DATASET" \
        --cost-preference "$COST_PREFERENCE" \
        --num-tasks "$NUM_TASKS" \
        --batch-size "$BATCH_SIZE" \
        --mode "$MODE" \
        --output-dir "$LOG_DIR" \
        $([ "$VERBOSE" = true ] && echo "--verbose") \
        2>&1 | tee -a "$LOGFILE"
else
    # Run Pareto analysis (multiple cost preferences)
    echo -e "${YELLOW}Running Pareto analysis with multiple cost preferences...${NC}"

    for cost in high_efficiency balanced high_success; do
        echo -e "\n${BLUE}Evaluating with cost preference: $cost${NC}" | tee -a "$LOGFILE"

        python -u "$SCRIPT_DIR/eval_tosrl_tti.py" \
            --config "$CONFIG_FILE" \
            --checkpoint "$CHECKPOINT" \
            --experiment "${EXPERIMENT_ID}_${cost}" \
            --dataset "$DATASET" \
            --cost-preference "$cost" \
            --num-tasks "$NUM_TASKS" \
            --batch-size "$BATCH_SIZE" \
            --mode "$MODE" \
            --output-dir "$LOG_DIR" \
            $([ "$VERBOSE" = true ] && echo "--verbose") \
            2>&1 | tee -a "$LOGFILE"
    done

    # Run comparison analysis
    python -u "$SCRIPT_DIR/analyze_pareto.py" \
        --results-dir "$LOG_DIR" \
        --experiment-id "$EXPERIMENT_ID" \
        2>&1 | tee -a "$LOGFILE"
fi

EVAL_EXIT=$?

echo -e "\n${BLUE}========================================${NC}" | tee -a "$LOGFILE"
if [ $EVAL_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation completed successfully!${NC}" | tee -a "$LOGFILE"
    echo "Results saved to: $LOG_DIR" | tee -a "$LOGFILE"
else
    echo -e "${RED}✗ Evaluation failed with exit code $EVAL_EXIT${NC}" | tee -a "$LOGFILE"
    exit $EVAL_EXIT
fi

echo "End time: $(date)" | tee -a "$LOGFILE"
echo -e "${BLUE}========================================${NC}\n" | tee -a "$LOGFILE"

# Print summary
echo -e "${GREEN}Evaluation Summary:${NC}"
echo "  Logs:       $LOGFILE"
echo "  Results:    $LOG_DIR/results.json"
echo "  Metrics:    $LOG_DIR/metrics.json"
if [ "$PARETO_ANALYSIS" = true ]; then
    echo "  Pareto:     $LOG_DIR/pareto_analysis.json"
fi

print_help() {
    cat << EOF
TOS-RL Evaluation Script

Usage: $0 [OPTIONS]

Required:
  --checkpoint PATH            Path to model checkpoint (e.g., best_model.pt)

Options:
  --config CONFIG_NAME         Config file name (default: webarena_eval)
                               Options: webarena_eval, webvoyager_eval, default

  --dataset DATASET            Dataset name (default: webarena)
                               Options: webarena, webvoyager

  --experiment EXP_ID          Experiment ID (default: auto-generated timestamp)

  --cost PREFERENCE            Cost preference (default: balanced)
                               Options: high_efficiency, balanced, high_success

  --num-tasks N                Number of tasks to evaluate (default: 100)

  --batch-size N               Evaluation batch size (default: 4)

  --mode MODE                  Evaluation mode (default: tosrl_eval)
                               Options: tosrl_eval, tti_eval, cvi_eval

  --pareto                     Run Pareto analysis (test all cost preferences)

  --verbose                    Enable verbose output

  --help, -h                   Show this help message

Examples:
  # Basic evaluation with single cost preference
  $0 --checkpoint best_model.pt

  # Evaluation with custom parameters
  $0 --checkpoint best_model.pt --cost high_success --num-tasks 200

  # Full Pareto analysis
  $0 --checkpoint best_model.pt --pareto

  # WebVoyager evaluation
  $0 --checkpoint checkpoints/webarena_to_webvoyager.pt --dataset webvoyager

  # Verbose evaluation
  $0 --checkpoint best_model.pt --verbose --cost balanced

Environment Variables:
  CUDA_VISIBLE_DEVICES   GPU IDs to use (e.g., "0,1,2,3")
  PYTHONPATH             Python path for imports

Cost Preferences:
  high_efficiency        Minimize steps & tokens (λ_env=0.1, λ_tok=0.01)
  balanced               Balance success & cost (λ_env=0.01, λ_tok=0.001)
  high_success           Maximize success (λ_env=0.001, λ_tok=0.0001)

EOF
}
