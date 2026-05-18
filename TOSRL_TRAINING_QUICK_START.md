# TOS-RL Training Quick Start Guide

**Status**: Real data training ready
**Date**: 2026-05-18

---

## 🎯 Two Training Modes

### Mode 1: Mock Data Training (Demo/Testing)
- Uses randomly generated fake trajectories
- 16 tasks per epoch
- No real trajectory data needed
- Good for: Testing setup, debugging, development

### Mode 2: Real Data Training (Production)
- Uses real trajectories from your TTI agent
- Loads from JSONL trajectory file
- Groups by task_id
- Good for: Training with real agent performance data

---

## Quick Start: 30 Seconds

### For Mock Data (Testing)
```bash
cd /Users/luungoc/Project/TTI/scripts

# Train with defaults (10 epochs)
./run_train.sh

# Or customize
./run_train.sh --epochs 20 --batch-size 8
```

### For Real Data (Production)
```bash
cd /Users/luungoc/Project/TTI/scripts

# Train on real trajectories
./run_train_real_data.sh --trajectory-file /path/to/trajectories.jsonl

# With customization
./run_train_real_data.sh \
  --trajectory-file /path/to/trajectories.jsonl \
  --epochs 20 \
  --group-size 4
```

---

## Setup Real Data (5 Minutes)

### Step 1: Collect Trajectories from Your Agent

Your TTI agent must track and save these metrics for each task:

```python
import json

# After each task, save trajectory
trajectory = {
    "task_id": "task_001",              # Unique task identifier
    "success": 1,                        # 1 if task succeeded, 0 otherwise
    "num_steps": 5,                      # Number of OBSERVE actions
    "num_tokens": 250,                   # Total tokens generated
    "num_loops": 0,                      # Number of repeated actions
    "num_bad_actions": 0,                # Number of invalid actions
    "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]  # Mode sequence
}

# Save to JSONL file (one JSON per line)
with open("trajectories.jsonl", "a") as f:
    f.write(json.dumps(trajectory) + "\n")
```

### Step 2: Format as JSONL

Your file should look like:
```jsonl
{"task_id": "task_001", "success": 1, "num_steps": 5, "num_tokens": 250, "num_loops": 0, "num_bad_actions": 0, "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]}
{"task_id": "task_002", "success": 0, "num_steps": 15, "num_tokens": 800, "num_loops": 2, "num_bad_actions": 1, "modes": ["OBSERVE", "OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"]}
{"task_id": "task_003", "success": 1, "num_steps": 8, "num_tokens": 400, "num_loops": 1, "num_bad_actions": 0, "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]}
```

### Step 3: Validate Format

```bash
# Check file exists and is valid
python3 -c "
import json
with open('trajectories.jsonl') as f:
    lines = f.readlines()
    print(f'Total trajectories: {len(lines)}')
    for i, line in enumerate(lines[:3]):
        traj = json.loads(line)
        print(f'  Task {i}: {traj[\"task_id\"]}, success={traj[\"success\"]}')
    if len(lines) > 3:
        print(f'  ... and {len(lines) - 3} more')
"
```

### Step 4: Run Training

```bash
cd /Users/luungoc/Project/TTI/scripts
./run_train_real_data.sh --trajectory-file ../data/trajectories.jsonl --epochs 20
```

---

## Complete Examples

### Example 1: Quick Test with Mock Data
```bash
cd /Users/luungoc/Project/TTI/scripts

# Train for 2 epochs to test setup
./run_train.sh --epochs 2

# Output: logs/training/webarena/tosrl_train_20260518_103045/
#         ├── training.log
#         ├── checkpoints/
#         │   ├── epoch_1.pt
#         │   ├── epoch_2.pt
#         │   └── best_model.pt
#         └── results.json
```

### Example 2: Production Training with Real Data
```bash
# First, create trajectories.jsonl from your TTI agent
python3 << 'EOF'
import json
import random

# Simulate collecting 100 trajectories from your agent
trajectories = []
for task_id in range(1, 101):
    # In real scenario, these come from your agent
    trajectory = {
        "task_id": f"webarena_task_{task_id:04d}",
        "success": random.choice([0, 1]),
        "num_steps": random.randint(3, 20),
        "num_tokens": random.randint(100, 500),
        "num_loops": random.randint(0, 3),
        "num_bad_actions": random.randint(0, 2),
        "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]
    }
    trajectories.append(trajectory)

# Save to JSONL
with open("../data/trajectories.jsonl", "w") as f:
    for traj in trajectories:
        f.write(json.dumps(traj) + "\n")

print(f"✓ Saved {len(trajectories)} trajectories")
EOF

# Now train on real data
./run_train_real_data.sh \
  --trajectory-file ../data/trajectories.jsonl \
  --epochs 20 \
  --batch-size 8 \
  --group-size 4 \
  --experiment "real_data_training_v1"

# Monitor in another terminal
tail -f ../logs/training/webarena/real_data_training_v1/training.log
```

### Example 3: Compare Mock vs Real Data
```bash
# Train with mock data
./run_train.sh \
  --epochs 5 \
  --experiment mock_baseline

# Create real data (see Example 2 code)
python3 create_sample_trajectories.py

# Train with real data
./run_train_real_data.sh \
  --trajectory-file ../data/trajectories.jsonl \
  --epochs 5 \
  --experiment real_data_baseline

# Compare results
python3 << 'EOF'
import json

# Load mock results
with open("../logs/training/webarena/mock_baseline/results.json") as f:
    mock = json.load(f)

# Load real data results
with open("../logs/training/webarena/real_data_baseline/results.json") as f:
    real = json.load(f)

print("Mock Data Results:")
print(f"  Final success: {mock[-1]['metrics']['success_rate']:.2%}")

print("\nReal Data Results:")
print(f"  Final success: {real[-1]['metrics']['success_rate']:.2%}")
EOF
```

---

## Common Commands

### Training
```bash
# Mock data: Quick test
./run_train.sh --epochs 2

# Mock data: Full training
./run_train.sh --epochs 20 --batch-size 8 --experiment my_test

# Real data: Basic
./run_train_real_data.sh --trajectory-file data/traj.jsonl

# Real data: Full customization
./run_train_real_data.sh \
  --trajectory-file data/traj.jsonl \
  --epochs 20 \
  --batch-size 8 \
  --group-size 4 \
  --branching \
  --experiment my_real_training
```

### Monitoring
```bash
# Watch training logs in real-time
tail -f logs/training/webarena/EXPERIMENT_ID/training.log

# Check available checkpoints
ls -lh logs/training/webarena/EXPERIMENT_ID/checkpoints/

# View results
cat logs/training/webarena/EXPERIMENT_ID/results.json | python3 -m json.tool
```

### Evaluation
```bash
# Evaluate after training
./run_eval.sh \
  --checkpoint logs/training/webarena/EXPERIMENT_ID/checkpoints/best_model.pt \
  --cost-preference balanced

# Pareto analysis (3 cost preferences)
./run_eval.sh \
  --checkpoint logs/training/webarena/EXPERIMENT_ID/checkpoints/best_model.pt \
  --pareto
```

---

## Trajectory File Format Reference

### Required Fields (MUST have)
```json
{
  "task_id": "string",           // Unique identifier for task
  "success": 0|1,                // Whether task succeeded
  "num_steps": 5,                // Number of OBSERVE steps
  "num_tokens": 250,             // Total tokens in trajectory
  "modes": ["THINK", "OBSERVE"]  // Sequence of modes used
}
```

### Optional Fields (can add)
```json
{
  "task_id": "task_001",
  "success": 1,
  "num_steps": 5,
  "num_tokens": 250,
  "num_loops": 0,                // (optional) Repeated actions
  "num_bad_actions": 0,          // (optional) Invalid actions
  "modes": ["THINK", "OBSERVE", "ANSWER"],

  "description": "Find price",   // (optional) Task description
  "domain": "shopping",          // (optional) Task domain
  "difficulty": "medium"         // (optional) Difficulty level
}
```

---

## FAQ

### Q: What's the difference between mock and real data?
**A**: Mock data is randomly generated (16 fake tasks/epoch). Real data comes from your actual agent (however many tasks you collected).

### Q: How many trajectories do I need?
**A**: Minimum: 4 trajectories per task (with default group_size=4). Recommended: 4+ trajectories per 100+ unique tasks.

### Q: Can I mix mock and real data?
**A**: No, use one at a time. But you can train mock→real→mock to compare.

### Q: What's this "group_size" parameter?
**A**: Groups K trajectories from same task for training. With K=4, you get one training group per 4 same-task trajectories. Groups with <K trajectories are skipped.

### Q: How do I create trajectories?
**A**: Modify your TTI agent to log: task_id, success, num_steps, num_tokens, modes. Save as JSONL (one JSON per line).

### Q: Can I run multiple trainings in parallel?
**A**: Yes, use different --experiment IDs. They'll create separate log directories.

### Q: Where are the results saved?
**A**: `logs/training/DATASET/EXPERIMENT_ID/`
- `training.log` - Full training logs
- `checkpoints/` - Model checkpoints per epoch + best_model.pt
- `results.json` - Training metrics per epoch

---

## Troubleshooting

### Error: Trajectory file not found
```bash
# Make sure file exists and path is correct
ls -lh /path/to/trajectories.jsonl

# Use absolute path if relative path fails
./run_train_real_data.sh --trajectory-file /absolute/path/to/traj.jsonl
```

### Error: Invalid JSON in trajectory file
```bash
# Validate file format
python3 -c "
import json
with open('trajectories.jsonl') as f:
    for i, line in enumerate(f, 1):
        json.loads(line)
        if i % 100 == 0: print(f'✓ Line {i}')
print('✓ All lines valid')
"
```

### Error: No training groups available
```bash
# This means no task has >= group_size trajectories
# Either:
# 1. Collect more trajectories per task
# 2. Lower group-size: ./run_train_real_data.sh --group-size 2
```

### Error: PyTorch not installed
```bash
pip install torch
```

### Error: Module not found (yaml, numpy, etc.)
```bash
pip install pyyaml numpy
```

---

## Next Steps

1. **For Testing**: Run `./run_train.sh --epochs 2` to verify setup works
2. **For Real Training**:
   - Collect trajectories from your TTI agent
   - Save as JSONL file
   - Run `./run_train_real_data.sh --trajectory-file <file>`
3. **After Training**: Evaluate with `./run_eval.sh --checkpoint <model>`

---

## File Reference

| File | Purpose |
|------|---------|
| `run_train.sh` | Mock data training (testing) |
| `run_train_real_data.sh` | Real data training (production) |
| `train_tosrl_tti.py` | Mock data training implementation |
| `train_tosrl_tti_real_data.py` | Real data training implementation |
| `TOSRL_REAL_DATA_GUIDE.md` | Detailed real data guide |
| `TOSRL_IMPLEMENTATION_GUIDE.md` | TOS-RL concepts & architecture |
| `TOSRL_SETUP_GUIDE.md` | Environment setup & dependencies |

---

**Last Updated**: 2026-05-18
**Status**: Ready for use with both mock and real data
**Next**: See TOSRL_REAL_DATA_GUIDE.md for detailed real data setup
