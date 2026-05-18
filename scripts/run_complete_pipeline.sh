#
# Complete TOS-RL Pipeline: Data → Training → Evaluation
#
# Usage:
#   ./run_complete_pipeline.sh                    # Full pipeline with sample data
#   ./run_complete_pipeline.sh --data-size 100    # Generate 100 tasks worth of data
#   ./run_complete_pipeline.sh --help              # Show help
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
LOGS_DIR="$PROJECT_DIR/logs"

# Default parameters
DATA_SIZE=20  # 20 tasks × 4 trajectories = 80 trajectories
EPOCHS=3
BATCH_SIZE=4
GROUP_SIZE=4
DATASET="webarena"
EXPERIMENT_ID="complete_pipeline_$(date +%Y%m%d_%H%M%S)"

print_help() {
    cat << EOF
${BLUE}TOS-RL Complete Pipeline (Data → Training → Evaluation)${NC}

USAGE:
  ./run_complete_pipeline.sh [OPTIONS]

OPTIONS:
  --data-size N       Number of tasks to generate (default: 20)
  --epochs N          Training epochs (default: 3)
  --batch-size N      Batch size (default: 4)
  --group-size N      Group size (default: 4)
  --experiment ID     Experiment identifier
  --dataset NAME      Dataset name (default: webarena)
  --help, -h          Show this help

PIPELINE STEPS:
  1. Generate/validate trajectory data (JSONL)
  2. Analyze data statistics
  3. Train TOS-RL on real data
  4. Monitor training
  5. Evaluate trained model
  6. Show results comparison

EXAMPLES:
  # Default (20 tasks, 3 epochs)
  ./run_complete_pipeline.sh

  # Larger dataset
  ./run_complete_pipeline.sh --data-size 100 --epochs 10

  # Custom experiment
  ./run_complete_pipeline.sh --experiment my_experiment --dataset webarena

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Paths
TRAJECTORY_FILE="$DATA_DIR/trajectories_${EXPERIMENT_ID}.jsonl"
TRAINING_LOG_DIR="$LOGS_DIR/training/${DATASET}/${EXPERIMENT_ID}"
CHECKPOINT_DIR="$TRAINING_LOG_DIR/checkpoints"
EVAL_LOG_DIR="$LOGS_DIR/evaluation/${DATASET}/${EXPERIMENT_ID}"

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  TOS-RL COMPLETE PIPELINE: Data → Training → Evaluation       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# STEP 1: Generate/Validate Data
# ============================================================================

echo -e "\n${BLUE}STEP 1: Prepare Trajectory Data${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$DATA_DIR"

if [ -f "$TRAJECTORY_FILE" ]; then
    echo -e "${YELLOW}Trajectory file already exists: $TRAJECTORY_FILE${NC}"
    EXISTING_LINES=$(wc -l < "$TRAJECTORY_FILE")
    echo "  Trajectories: $EXISTING_LINES"
else
    echo "Generating $DATA_SIZE tasks × 4 trajectories = $((DATA_SIZE * 4)) trajectories..."

    python3 << PYTHON_EOF
import json
import random
from pathlib import Path

num_tasks = $DATA_SIZE
num_traj_per_task = 4
output_file = "$TRAJECTORY_FILE"

# Mode patterns
mode_patterns = [
    ["THINK", "OBSERVE", "ANSWER"],
    ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
    ["OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"],
    ["THINK", "OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"],
    ["THINK", "THINK", "OBSERVE", "OBSERVE", "ANSWER"],
]

with open(output_file, 'w') as f:
    for task_id in range(1, num_tasks + 1):
        for traj_idx in range(num_traj_per_task):
            trajectory = {
                "task_id": f"task_{task_id:04d}",
                "success": random.randint(0, 1),
                "num_steps": random.randint(3, 25),
                "num_tokens": random.randint(100, 800),
                "num_loops": random.randint(0, 3),
                "num_bad_actions": random.randint(0, 2),
                "modes": random.choice(mode_patterns),
            }
            f.write(json.dumps(trajectory) + "\n")

print(f"✓ Generated {num_tasks * num_traj_per_task} trajectories")
PYTHON_EOF
fi

echo -e "${GREEN}✓ Data ready${NC}"

# ============================================================================
# STEP 2: Analyze Data
# ============================================================================

echo -e "\n${BLUE}STEP 2: Analyze Trajectory Data${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << PYTHON_EOF
import json
from collections import defaultdict

trajectory_file = "$TRAJECTORY_FILE"

trajectories = []
task_groups = defaultdict(list)

with open(trajectory_file, 'r') as f:
    for line in f:
        traj = json.loads(line)
        trajectories.append(traj)
        task_groups[traj['task_id']].append(traj)

# Statistics
success_count = sum(1 for t in trajectories if t['success'])
total_tokens = sum(t['num_tokens'] for t in trajectories)
total_steps = sum(t['num_steps'] for t in trajectories)

print(f"Total trajectories: {len(trajectories)}")
print(f"Unique tasks: {len(task_groups)}")
print(f"Avg per task: {len(trajectories) / len(task_groups):.1f}")
print(f"\nSuccess rate: {100 * success_count / len(trajectories):.1f}%")
print(f"Avg steps: {total_steps / len(trajectories):.1f}")
print(f"Avg tokens: {total_tokens / len(trajectories):.0f}")

# Mode distribution
all_modes = []
for t in trajectories:
    all_modes.extend(t['modes'])

print(f"\nMode distribution:")
print(f"  THINK:   {100 * all_modes.count('THINK') / len(all_modes):5.1f}%")
print(f"  OBSERVE: {100 * all_modes.count('OBSERVE') / len(all_modes):5.1f}%")
print(f"  ANSWER:  {100 * all_modes.count('ANSWER') / len(all_modes):5.1f}%")

# Training info
valid_groups = sum(1 for tasks in task_groups.values() if len(tasks) >= $GROUP_SIZE)
print(f"\nTraining groups (>=$GROUP_SIZE traj): {valid_groups}")
print(f"Trajectories per epoch: {valid_groups * $GROUP_SIZE}")
PYTHON_EOF

echo -e "${GREEN}✓ Analysis complete${NC}"

# ============================================================================
# STEP 3: Train Model
# ============================================================================

echo -e "\n${BLUE}STEP 3: Train TOS-RL on Real Data${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$TRAINING_LOG_DIR"

echo "Training configuration:"
echo "  Epochs:       $EPOCHS"
echo "  Batch size:   $BATCH_SIZE"
echo "  Group size:   $GROUP_SIZE"
echo "  Output dir:   $TRAINING_LOG_DIR"
echo ""
echo "Starting training..."

# Run training
cd "$SCRIPT_DIR"
./run_train_real_data.sh \
    --trajectory-file "$TRAJECTORY_FILE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --group-size "$GROUP_SIZE" \
    --dataset "$DATASET" \
    --experiment "$EXPERIMENT_ID"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training complete${NC}"

# ============================================================================
# STEP 4: Evaluate Model
# ============================================================================

echo -e "\n${BLUE}STEP 4: Evaluate Trained Model${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Find best checkpoint
BEST_CHECKPOINT="$CHECKPOINT_DIR/best_model.pt"

if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo -e "${RED}Error: Best checkpoint not found${NC}"
    exit 1
fi

echo "Evaluating on 3 cost preferences..."
mkdir -p "$EVAL_LOG_DIR"

for preference in high_efficiency balanced high_success; do
    echo -e "\n  Evaluating: ${CYAN}$preference${NC}"
    ./run_eval.sh \
        --checkpoint "$BEST_CHECKPOINT" \
        --cost-preference "$preference" \
        --output-dir "$EVAL_LOG_DIR" \
        --experiment "$EXPERIMENT_ID" \
        --dataset "$DATASET" \
        2>&1 | grep -E "(Results|Success|Steps|Mode)" | head -10
done

echo -e "${GREEN}✓ Evaluation complete${NC}"

# ============================================================================
# STEP 5: Show Results
# ============================================================================

echo -e "\n${BLUE}STEP 5: Summary${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "${GREEN}Pipeline completed successfully!${NC}\n"

echo "📊 Training Results:"
echo "  Logs:         ${YELLOW}$TRAINING_LOG_DIR/training.log${NC}"
echo "  Checkpoints:  ${YELLOW}$CHECKPOINT_DIR${NC}"
echo "  Results:      ${YELLOW}$TRAINING_LOG_DIR/results.json${NC}"

echo ""
echo "📈 Evaluation Results:"
echo "  Directory:    ${YELLOW}$EVAL_LOG_DIR${NC}"

echo ""
echo "🔍 View Results:"
echo "  Training logs:"
echo "    ${CYAN}tail -f $TRAINING_LOG_DIR/training.log${NC}"
echo ""
echo "  Training metrics:"
echo "    ${CYAN}cat $TRAINING_LOG_DIR/results.json | python3 -m json.tool | head -50${NC}"
echo ""
echo "  Evaluation metrics:"
echo "    ${CYAN}ls -lh $EVAL_LOG_DIR/metrics_*.json${NC}"
echo ""

# Show final metrics
echo "📌 Final Training Metrics:"
python3 << PYTHON_EOF
import json
import os

results_file = "$TRAINING_LOG_DIR/results.json"
if os.path.exists(results_file):
    with open(results_file) as f:
        results = json.load(f)

    if results:
        final = results[-1]
        print(f"  Epoch: {final.get('epoch', 'N/A')}")
        if 'metrics' in final:
            metrics = final['metrics']
            print(f"  Success rate: {metrics.get('success_rate', 'N/A'):.2%}" if isinstance(metrics.get('success_rate'), float) else f"  Success rate: {metrics.get('success_rate', 'N/A')}")
            print(f"  Total loss: {metrics.get('total_loss', 'N/A'):.4f}" if isinstance(metrics.get('total_loss'), float) else f"  Total loss: {metrics.get('total_loss', 'N/A')}")
PYTHON_EOF

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Complete pipeline finished successfully!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  1. Review training logs:"
echo "     tail -f $TRAINING_LOG_DIR/training.log"
echo ""
echo "  2. Visualize results:"
echo "     python3 scripts/analyze_pareto.py --results $EVAL_LOG_DIR"
echo ""
echo "  3. Run inference:"
echo "     python3 scripts/inference_tosrl.py --checkpoint $BEST_CHECKPOINT"
echo ""
