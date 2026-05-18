# TOS-RL Complete Pipeline: Setup & Usage Guide

**Status**: ✅ Ready to use (requires PyTorch installation)
**Date**: 2026-05-18

---

## What is the Complete Pipeline?

A one-command end-to-end workflow:

```
Data Generation → Data Analysis → Training → Evaluation → Results Summary
```

**Single Command**:
```bash
./run_complete_pipeline.sh
```

---

## Prerequisites

### 1. Install Dependencies

```bash
cd /Users/luungoc/Project/TTI

# Install all required packages
pip install -r requirements.txt
```

### Detailed Installation Guide

#### For macOS
```bash
# Using pip (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers pyyaml numpy

# Or using conda
conda install pytorch torchvision torchaudio -c pytorch
conda install pyyaml numpy transformers
```

#### For Linux with GPU
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### For Windows
```bash
pip install torch torchvision torchaudio
pip install transformers pyyaml numpy
```

### 2. Verify Installation

```bash
python3 << 'EOF'
import torch
import yaml
import numpy as np
import transformers

print("✓ PyTorch version:", torch.__version__)
print("✓ NumPy version:", np.__version__)
print("✓ PyYAML installed")
print("✓ Transformers version:", transformers.__version__)
print("\n✓ All dependencies installed!")
EOF
```

---

## Quick Start (2 Minutes)

### Step 1: Navigate to scripts directory
```bash
cd /Users/luungoc/Project/TTI/scripts
```

### Step 2: Run complete pipeline with defaults
```bash
./run_complete_pipeline.sh
```

### What happens:
```
STEP 1: Generate 20 tasks × 4 trajectories = 80 trajectories
STEP 2: Analyze data statistics
STEP 3: Train for 3 epochs
STEP 4: Evaluate trained model
STEP 5: Show results summary
```

**Total time**: ~5-10 minutes (depends on CPU/GPU)

### Step 3: Check results
```bash
# View training logs
tail -f ../logs/training/webarena/complete_pipeline_*/training.log

# View final metrics
cat ../logs/training/webarena/complete_pipeline_*/results.json | python3 -m json.tool
```

---

## Advanced Usage

### Customize Data Size
```bash
# Small dataset (5 tasks = 20 trajectories)
./run_complete_pipeline.sh --data-size 5 --epochs 2

# Medium dataset (50 tasks = 200 trajectories)
./run_complete_pipeline.sh --data-size 50 --epochs 5

# Large dataset (200 tasks = 800 trajectories)
./run_complete_pipeline.sh --data-size 200 --epochs 10
```

### Custom Experiment Name
```bash
./run_complete_pipeline.sh \
  --experiment my_experiment \
  --data-size 50 \
  --epochs 10 \
  --batch-size 8
```

### Full Customization
```bash
./run_complete_pipeline.sh \
  --data-size 100 \
  --epochs 15 \
  --batch-size 8 \
  --group-size 4 \
  --dataset webarena \
  --experiment production_run
```

---

## Pipeline Details

### What Each Step Does

#### Step 1: Prepare Trajectory Data
- Generates synthetic trajectory file in JSONL format
- 20 tasks × 4 trajectories per task (configurable)
- Realistic metrics:
  - Success rate: 40-60% (random)
  - Steps: 3-25 per trajectory
  - Tokens: 100-800 per trajectory
  - Modes: Mix of THINK, OBSERVE, ANSWER

**Output**: `/Users/luungoc/Project/TTI/data/trajectories_EXPERIMENT_ID.jsonl`

#### Step 2: Analyze Data
- Counts trajectories
- Computes success rate
- Analyzes mode distribution
- Shows training group statistics

**Example Output**:
```
Total trajectories: 80
Unique tasks: 20
Success rate: 55.0%
Avg steps: 14.2
Avg tokens: 450

Mode distribution:
  THINK:   24.5%
  OBSERVE: 52.0%
  ANSWER:  23.5%

Training groups (>=4 traj): 20
Trajectories per epoch: 80
```

#### Step 3: Train Model
- Loads trajectories from JSONL file
- Groups by task_id (K=4 default)
- Trains TOS-RL for N epochs
- Saves checkpoints every epoch
- Saves best checkpoint

**Output**:
```
logs/training/webarena/EXPERIMENT_ID/
├── training.log          # Full training logs
├── results.json          # Training metrics per epoch
└── checkpoints/
    ├── epoch_1.pt
    ├── epoch_2.pt
    ├── epoch_3.pt
    └── best_model.pt
```

#### Step 4: Evaluate Model
- Loads best checkpoint
- Evaluates on 3 cost preferences:
  - high_efficiency: fewer interactions
  - balanced: medium interactions
  - high_success: maximum interactions
- Saves evaluation metrics

**Output**:
```
logs/evaluation/webarena/EXPERIMENT_ID/
├── metrics_high_efficiency.json
├── metrics_balanced.json
├── metrics_high_success.json
└── results_*.json
```

#### Step 5: Show Results
- Displays training summary
- Shows evaluation results
- Provides next steps

---

## File Structure After Pipeline

```
TOS-RL/
├── data/
│   └── trajectories_EXPERIMENT_ID.jsonl    # Generated trajectories
├── logs/
│   ├── training/
│   │   └── webarena/EXPERIMENT_ID/
│   │       ├── training.log                # Full logs
│   │       ├── results.json                # Metrics
│   │       └── checkpoints/
│   │           ├── epoch_1.pt
│   │           ├── epoch_2.pt
│   │           └── best_model.pt           # Best checkpoint
│   └── evaluation/
│       └── webarena/EXPERIMENT_ID/
│           ├── metrics_*.json              # Evaluation results
│           └── results_*.json
└── scripts/
    ├── run_complete_pipeline.sh            # Main script
    ├── run_train_real_data.sh              # Training
    ├── run_eval.sh                         # Evaluation
    └── create_sample_trajectories.py       # Sample data
```

---

## Monitoring Training

### In Real-Time
```bash
# Open new terminal and watch training logs
tail -f /Users/luungoc/Project/TTI/logs/training/webarena/EXPERIMENT_ID/training.log
```

### After Training
```bash
# View final results
cat /Users/luungoc/Project/TTI/logs/training/webarena/EXPERIMENT_ID/results.json | python3 -m json.tool

# View specific epoch
python3 << 'EOF'
import json

with open("logs/training/webarena/EXPERIMENT_ID/results.json") as f:
    results = json.load(f)

for result in results:
    epoch = result['epoch']
    metrics = result['metrics']
    print(f"Epoch {epoch}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
EOF
```

---

## Example Runs

### Example 1: Quick Test (2 minutes)
```bash
./run_complete_pipeline.sh --data-size 5 --epochs 2
```

Expected output:
- 20 trajectories generated
- 5 training groups per epoch
- 2 epochs training
- Total time: ~2-3 minutes

### Example 2: Small Production (5 minutes)
```bash
./run_complete_pipeline.sh --data-size 50 --epochs 5
```

Expected output:
- 200 trajectories generated
- 50 training groups per epoch
- 5 epochs training
- Total time: ~5-10 minutes

### Example 3: Medium Production (10+ minutes)
```bash
./run_complete_pipeline.sh --data-size 200 --epochs 10
```

Expected output:
- 800 trajectories generated
- 200 training groups per epoch
- 10 epochs training
- Total time: ~15-30 minutes

### Example 4: Custom Configuration
```bash
./run_complete_pipeline.sh \
  --data-size 100 \
  --epochs 15 \
  --batch-size 8 \
  --group-size 4 \
  --experiment "my_production_run" \
  --dataset "webarena"
```

---

## Troubleshooting

### Error: `PyTorch not installed`
**Solution**:
```bash
pip install torch torchvision torchaudio
# or for GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Error: `Config file not found`
**Solution**:
```bash
# Make sure config files exist
ls /Users/luungoc/Project/TTI/scripts/config/main/
# Should show: default.yaml, webarena_rl.yaml, etc.
```

### Error: `Command not found: ./run_complete_pipeline.sh`
**Solution**:
```bash
# Make sure you're in scripts directory
cd /Users/luungoc/Project/TTI/scripts

# Make sure script is executable
chmod +x run_complete_pipeline.sh

# Run it
./run_complete_pipeline.sh
```

### Training takes too long
**Options**:
1. Use smaller dataset: `--data-size 20 --epochs 2`
2. Reduce epochs: `--epochs 5` instead of default 10
3. Use GPU if available

### Out of memory
**Try**:
1. Reduce batch size: `--batch-size 2`
2. Reduce data size: `--data-size 50`
3. Reduce epochs: `--epochs 2`

---

## Using Real Data

Instead of generated data, use your own trajectories:

```bash
# Create your trajectories file in JSONL format
# See TOSRL_TTI_INTEGRATION_GUIDE.md for details

# Then use it with training script directly
cd /Users/luungoc/Project/TTI/scripts
./run_train_real_data.sh \
  --trajectory-file /path/to/your/trajectories.jsonl \
  --epochs 20

# Then evaluate
./run_eval.sh \
  --checkpoint logs/training/webarena/*/checkpoints/best_model.pt
```

---

## Pipeline Command Reference

### Full Options
```bash
./run_complete_pipeline.sh [OPTIONS]

OPTIONS:
  --data-size N       Number of tasks to generate (default: 20)
  --epochs N          Training epochs (default: 3)
  --batch-size N      Batch size (default: 4)
  --group-size N      Group size (default: 4)
  --experiment ID     Experiment identifier
  --dataset NAME      Dataset name (default: webarena)
  --help, -h          Show help message
```

### Common Examples
```bash
# Minimal (test setup)
./run_complete_pipeline.sh --data-size 5 --epochs 1

# Standard (recommended)
./run_complete_pipeline.sh

# Production (large scale)
./run_complete_pipeline.sh --data-size 500 --epochs 20 --batch-size 8
```

---

## What Gets Saved

### Trajectory Data
```
data/trajectories_EXPERIMENT_ID.jsonl

Format: JSONL (one trajectory per line)
{
  "task_id": "task_0001",
  "success": 1,
  "num_steps": 5,
  "num_tokens": 250,
  "num_loops": 0,
  "num_bad_actions": 0,
  "modes": ["THINK", "OBSERVE", "ANSWER"]
}
```

### Training Logs
```
logs/training/webarena/EXPERIMENT_ID/training.log

Contains:
- Configuration used
- Training progress per epoch
- Loss values
- Checkpoint locations
- Final summary
```

### Training Results
```
logs/training/webarena/EXPERIMENT_ID/results.json

Format: JSON with per-epoch metrics
[
  {
    "epoch": 1,
    "metrics": {
      "total_loss": 0.5432,
      "policy_loss": 0.3210,
      "mode_loss": 0.2222,
      "success_rate": 0.5500
    }
  },
  ...
]
```

### Model Checkpoints
```
logs/training/webarena/EXPERIMENT_ID/checkpoints/

- epoch_1.pt      (checkpoint after epoch 1)
- epoch_2.pt      (checkpoint after epoch 2)
- epoch_N.pt      (checkpoint after epoch N)
- best_model.pt   (best checkpoint by success rate)
```

---

## Next Steps

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run complete pipeline**:
   ```bash
   cd /Users/luungoc/Project/TTI/scripts
   ./run_complete_pipeline.sh
   ```

3. **Monitor training**:
   ```bash
   tail -f ../logs/training/webarena/complete_pipeline_*/training.log
   ```

4. **Analyze results**:
   ```bash
   cat ../logs/training/webarena/complete_pipeline_*/results.json | python3 -m json.tool
   ```

5. **With your own data**:
   See `TOSRL_TTI_INTEGRATION_GUIDE.md`

---

## Performance Expectations

### Training Time (per epoch)
- 20 tasks: ~1-2 minutes
- 50 tasks: ~2-5 minutes
- 100 tasks: ~5-10 minutes
- 200 tasks: ~10-20 minutes

### Memory Usage
- Small (20 tasks): ~1-2 GB
- Medium (100 tasks): ~2-4 GB
- Large (200+ tasks): ~4-8 GB

### On GPU
- 3-5x faster than CPU
- Same memory requirements

---

## FAQ

**Q: Can I run multiple pipelines at once?**
A: Yes, use different `--experiment` IDs to avoid conflicts.

**Q: How much disk space do I need?**
A: ~500MB for 200 tasks × 20 epochs

**Q: Can I pause and resume training?**
A: Currently no, but you can resume using the checkpoint with a new training run.

**Q: What if training crashes?**
A: Checkpoints are saved every epoch, so you get partial results.

**Q: How do I use GPU?**
A: PyTorch will automatically use GPU if available. Install CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Summary

| Step | Command | Time | Output |
|------|---------|------|--------|
| 1. Generate Data | Built-in | <1 min | JSONL file |
| 2. Analyze | Built-in | <1 min | Statistics |
| 3. Train | Python + PyTorch | 2-20 min | Checkpoints + logs |
| 4. Evaluate | Python + PyTorch | <1 min | Metrics |
| 5. Summary | Built-in | <1 min | Report |

**Total**: 5-25 minutes depending on data size and hardware

---

**Status**: ✅ Ready for production use
**Next**: Install dependencies and run `./run_complete_pipeline.sh`
**Last Updated**: 2026-05-18
