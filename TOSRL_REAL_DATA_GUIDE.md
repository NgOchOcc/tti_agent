# TOS-RL Real Data Training Guide

**Status**: Real data training now supported
**Date**: 2026-05-18

---

## 📊 Current Status

### ❌ Previous Training (Mock Data Only)
```bash
./run_train.sh
# Uses: 16 FAKE tasks per epoch
# Status: Demo/proof-of-concept only
```

### ✅ New Training (Real Data Support)
```bash
./train_tosrl_tti_real_data.py --data-source file --trajectory-file data.jsonl
# Uses: REAL trajectories from file
# Status: Production-ready
```

---

## 📂 Real Data Format

### JSONL File Format

Your trajectory file should be **JSONL** (JSON Lines) - one trajectory per line.

**Example** (`webarena_trajectories.jsonl`):
```json
{"task_id": "task_001", "success": 1, "num_steps": 5, "num_tokens": 250, "num_loops": 0, "num_bad_actions": 0, "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]}
{"task_id": "task_002", "success": 0, "num_steps": 15, "num_tokens": 800, "num_loops": 2, "num_bad_actions": 1, "modes": ["OBSERVE", "OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"]}
{"task_id": "task_003", "success": 1, "num_steps": 8, "num_tokens": 400, "num_loops": 1, "num_bad_actions": 0, "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]}
```

### Required Fields for Each Trajectory

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `task_id` | string | Unique task identifier | `"task_001"` |
| `success` | int (0-1) | Whether task was successful | `1` |
| `num_steps` | int | Number of [OBSERVE] actions | `5` |
| `num_tokens` | int | Total tokens generated | `250` |
| `num_loops` | int | Number of repeated actions | `0` |
| `num_bad_actions` | int | Number of invalid actions | `0` |
| `modes` | list of strings | Sequence of mode tokens used | `["THINK", "OBSERVE", "ANSWER"]` |

### Optional Fields

```json
{
  "task_id": "task_001",
  "success": 1,
  "num_steps": 5,
  "num_tokens": 250,
  "num_loops": 0,
  "num_bad_actions": 0,
  "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],

  "description": "Find product price",           # (optional) Task description
  "trajectory_length": 4,                        # (optional) Total steps
  "domain": "shopping",                          # (optional) Task domain
  "difficulty": "medium"                         # (optional) Difficulty level
}
```

---

## 🚀 How to Use Real Data

### Option 1: Train from JSONL File

**Create your trajectory file** (`data/webarena_trajectories.jsonl`):

```bash
# Example: Create dummy trajectory file
cat > data/webarena_trajectories.jsonl << 'EOF'
{"task_id": "task_001", "success": 1, "num_steps": 5, "num_tokens": 250, "num_loops": 0, "num_bad_actions": 0, "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]}
{"task_id": "task_002", "success": 0, "num_steps": 15, "num_tokens": 800, "num_loops": 2, "num_bad_actions": 1, "modes": ["OBSERVE", "OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"]}
{"task_id": "task_003", "success": 1, "num_steps": 8, "num_tokens": 400, "num_loops": 1, "num_bad_actions": 0, "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]}
EOF
```

**Run training**:

```bash
cd /Users/luungoc/Project/TTI/scripts

python3 train_tosrl_tti_real_data.py \
  --config config/main/webarena_rl.yaml \
  --experiment "tosrl_real_data" \
  --output-dir ../logs/training/real_data \
  --data-source file \
  --trajectory-file ../data/webarena_trajectories.jsonl \
  --epochs 10
```

**Or use the shortcut**:
```bash
chmod +x train_tosrl_tti_real_data.py
./train_tosrl_tti_real_data.py \
  --config config/main/webarena_rl.yaml \
  --experiment "tosrl_real_data" \
  --output-dir ../logs/training/real_data \
  --data-source file \
  --trajectory-file ../data/webarena_trajectories.jsonl
```

### Option 2: Use WebArena Data

If you have collected trajectories from WebArena, organize them in JSONL format and use Option 1.

### Option 3: Real-Time TTI Agent Collection (Future)

```bash
# Not yet implemented, but coming soon:
python3 train_tosrl_tti_real_data.py \
  --config config/main/webarena_rl.yaml \
  --data-source tti \
  --epochs 10
```

---

## 📊 How Data Grouping Works

### Process

```
1. Load trajectories from file
   Input: JSONL with 1000 trajectories
   ↓
2. Group by task_id
   task_001: [traj1, traj2, traj3, traj4]
   task_002: [traj1, traj2, traj3]
   task_003: [traj1, traj2]
   ↓
3. Create K-sized groups
   For K=4 (group_size), keep full groups:
   Group1: [task_001_traj1, task_001_traj2, task_001_traj3, task_001_traj4]
   Group2: [task_002_traj1, task_002_traj2, task_002_traj3, ...] ← Skip (incomplete)
   ↓
4. Train on each group
   For each group:
   - Compute utilities
   - Compute group-relative advantages
   - Run GRPO training step
```

### Example

If you have:
- 100 tasks
- 4 trajectories per task
- Group size = 4

You get:
- 100 training groups per epoch
- 100 × 4 = 400 trajectories used per epoch

---

## 🔧 Creating Trajectory Files from Your TTI Agent

### Step 1: Collect Trajectories

Modify your TTI agent to save trajectories:

```python
import json

def collect_and_save_trajectories(agent, tasks, output_file):
    """Collect trajectories and save to JSONL."""

    with open(output_file, 'a') as f:
        for task in tasks:
            # Run agent on task
            success = agent.run(task)

            # Build trajectory dict
            trajectory = {
                "task_id": task['id'],
                "success": int(success),
                "num_steps": agent.num_steps,
                "num_tokens": agent.num_tokens,
                "num_loops": agent.num_repeated_actions,
                "num_bad_actions": agent.num_invalid_actions,
                "modes": agent.modes_used,  # List of ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]
            }

            # Save to JSONL
            f.write(json.dumps(trajectory) + '\n')

# Usage:
tasks = load_webarena_tasks()
collect_and_save_trajectories(agent, tasks, "webarena_trajectories.jsonl")
```

### Step 2: Verify Format

```bash
# Check JSONL format
head -5 webarena_trajectories.jsonl

# Count lines (trajectories)
wc -l webarena_trajectories.jsonl

# Validate JSON
python3 -c "
import json
with open('webarena_trajectories.jsonl') as f:
    count = 0
    for line in f:
        json.loads(line)
        count += 1
    print(f'✓ {count} valid trajectories')
"
```

### Step 3: Train

```bash
python3 train_tosrl_tti_real_data.py \
  --config config/main/webarena_rl.yaml \
  --experiment "tosrl_from_tti_agent" \
  --output-dir ../logs/training/from_tti \
  --data-source file \
  --trajectory-file ../webarena_trajectories.jsonl \
  --epochs 20
```

---

## 📈 Comparison: Mock vs Real Data

| Aspect | Mock Data | Real Data |
|--------|-----------|-----------|
| **Trajectories** | Random generated | From agent/file |
| **Realism** | Not realistic | Real performance data |
| **Tasks** | 16 per epoch | Actual number in file |
| **Success rate** | Random ~50% | Actual agent success |
| **Modes** | Fixed pattern | Real agent choices |
| **Training value** | Demo only | Production quality |

---

## 🎯 Quick Examples

### Example 1: Load From File

```bash
cd /Users/luungoc/Project/TTI/scripts

# Train on real trajectories
python3 train_tosrl_tti_real_data.py \
  --config config/main/webarena_rl.yaml \
  --experiment "real_data_training" \
  --output-dir ../logs/training \
  --data-source file \
  --trajectory-file ../data/webarena_trajectories.jsonl \
  --epochs 20 \
  --batch-size 8 \
  --branching
```

### Example 2: Compare Mock vs Real

```bash
# Mock data training
./run_train.sh --epochs 5 --experiment "mock_data"

# Real data training
python3 train_tosrl_tti_real_data.py \
  --config config/main/webarena_rl.yaml \
  --experiment "real_data" \
  --output-dir ../logs/training \
  --data-source file \
  --trajectory-file ../data/webarena_trajectories.jsonl \
  --epochs 5

# Compare results
python3 << 'EOF'
import json

# Load mock results
with open("../logs/training/webarena/mock_data/results.json") as f:
    mock = json.load(f)

# Load real results
with open("../logs/training/real_data/results.json") as f:
    real = json.load(f)

print("Mock Data Results:")
print(f"  Final success: {mock[-1]['metrics']['success_rate']:.2%}")

print("\nReal Data Results:")
print(f"  Final success: {real[-1]['metrics']['success_rate']:.2%}")
EOF
```

---

## 📊 Verifying Data Quality

### Check Your Trajectory File

```bash
python3 << 'EOF'
import json

file = "data/webarena_trajectories.jsonl"

with open(file) as f:
    trajectories = [json.loads(line) for line in f]

print(f"Total trajectories: {len(trajectories)}")

# Analyze
success_count = sum(1 for t in trajectories if t['success'])
print(f"Success rate: {100 * success_count / len(trajectories):.1f}%")

# Mode analysis
all_modes = []
for t in trajectories:
    all_modes.extend(t['modes'])

print(f"Mode distribution:")
print(f"  THINK: {100 * all_modes.count('THINK') / len(all_modes):.1f}%")
print(f"  OBSERVE: {100 * all_modes.count('OBSERVE') / len(all_modes):.1f}%")
print(f"  ANSWER: {100 * all_modes.count('ANSWER') / len(all_modes):.1f}%")

# Stats
print(f"\nStatistics:")
print(f"  Avg steps: {sum(t['num_steps'] for t in trajectories) / len(trajectories):.1f}")
print(f"  Avg tokens: {sum(t['num_tokens'] for t in trajectories) / len(trajectories):.1f}")
EOF
```

---

## ✅ Checklist for Real Data Training

- [ ] Create JSONL file with trajectory data
- [ ] Verify JSONL format (one JSON per line)
- [ ] Check all required fields are present
- [ ] Validate file has multiple trajectories (200+)
- [ ] Test data quality (check success rate, modes)
- [ ] Copy file to accessible location
- [ ] Run training with `--data-source file`
- [ ] Monitor logs during training
- [ ] Check results.json for metrics

---

## 🔗 Integration with Your TTI Agent

To integrate real trajectory collection:

```python
# In your TTI agent code

class TTIAgentWithTOSRLTracking:
    def __init__(self, trajectory_file):
        self.trajectory_file = trajectory_file
        self.modes = []
        self.num_steps = 0
        self.num_tokens = 0
        self.num_repeated_actions = 0
        self.num_invalid_actions = 0

    def run_and_save_trajectory(self, task):
        # ... run your agent ...
        success = self.check_success()

        # Save trajectory
        trajectory = {
            "task_id": task['id'],
            "success": int(success),
            "num_steps": self.num_steps,
            "num_tokens": self.num_tokens,
            "num_loops": self.num_repeated_actions,
            "num_bad_actions": self.num_invalid_actions,
            "modes": self.modes,
        }

        with open(self.trajectory_file, 'a') as f:
            f.write(json.dumps(trajectory) + '\n')

        return success

# Usage
agent = TTIAgentWithTOSRLTracking("webarena_trajectories.jsonl")

for task in webarenatasks:
    agent.run_and_save_trajectory(task)

# Then train
# python3 train_tosrl_tti_real_data.py --trajectory-file webarena_trajectories.jsonl
```

---

## 📞 Troubleshooting

### Error: "No trajectory file specified"
Make sure to pass `--trajectory-file <path>`

### Error: "Invalid JSON"
Check that each line is valid JSON. Use:
```bash
python3 -c "import json; [json.loads(line) for line in open('file.jsonl')]"
```

### Error: "No training groups available"
- Check that trajectories are grouped by task_id
- Make sure you have at least `group_size` trajectories per task
- Default `group_size=4`, so need 4+ trajectories per task

### Error: "Trajectory file not found"
- Use absolute path or correct relative path
- Check file exists: `ls -la <path>`

---

## 🎉 Next Steps

1. **Collect trajectories** from your TTI agent
2. **Format as JSONL** file
3. **Run training** with real data:
   ```bash
   python3 train_tosrl_tti_real_data.py \
     --config config/main/webarena_rl.yaml \
     --data-source file \
     --trajectory-file <your_file.jsonl>
   ```
4. **Monitor training** in logs
5. **Compare results** with mock data baseline

---

**Status**: Real data training ready to use
**Files**: `train_tosrl_tti_real_data.py` (new)
**Documentation**: This guide

