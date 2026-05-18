# TOS-RL Run Scripts Guide

Complete guide to using the training and evaluation shell scripts with YAML configs.

**Location**: `/Users/luungoc/Project/TTI/scripts/`

---

## 📋 Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_train.sh` | Training only | `./run_train.sh --epochs 10` |
| `run_eval.sh` | Evaluation only | `./run_eval.sh --checkpoint best_model.pt` |
| `run_train_eval.sh` | Training + Evaluation | `./run_train_eval.sh --epochs 10 --eval-every 2` |

---

## 🎓 Configuration Files

Located in: `/Users/luungoc/Project/TTI/scripts/config/main/`

```
config/main/
├── default.yaml              # Base configuration
├── webarena_rl.yaml          # WebArena training
├── webarena_eval.yaml        # WebArena evaluation
├── webvoyager_rl.yaml        # WebVoyager training
└── webvoyager_eval.yaml      # WebVoyager evaluation
```

### Config Hierarchy

Configs use Hydra's defaults system:

```yaml
defaults:
  - default          # Base settings
  - _self_           # Override with this file

# Then add/override values
epochs: 10
batch_size: 4
```

---

## 🚀 Quick Start Examples

### 1. Training Only

```bash
# Basic training (uses webarena_rl.yaml)
./run_train.sh

# Training 20 epochs
./run_train.sh --epochs 20

# Training with branching enabled
./run_train.sh --epochs 10 --branching

# Custom experiment name
./run_train.sh --experiment "tosrl_baseline" --epochs 5
```

### 2. Evaluation Only

```bash
# Evaluate a checkpoint (required)
./run_eval.sh --checkpoint checkpoints/best_model.pt

# Evaluate with different cost preference
./run_eval.sh --checkpoint best_model.pt --cost high_success

# Full Pareto analysis (test all cost regimes)
./run_eval.sh --checkpoint best_model.pt --pareto

# Evaluate more tasks
./run_eval.sh --checkpoint best_model.pt --num-tasks 200
```

### 3. Training + Evaluation

```bash
# Train and evaluate every 2 epochs
./run_train_eval.sh --epochs 10 --eval-every 2

# Train and do full Pareto at end
./run_train_eval.sh --epochs 20 --pareto

# WebVoyager training + eval
./run_train_eval.sh --dataset webvoyager --epochs 10
```

---

## 📝 Detailed Usage

### Training Script (`run_train.sh`)

```bash
# View help
./run_train.sh --help

# Training with custom parameters
./run_train.sh \
    --epochs 15 \
    --batch-size 8 \
    --grad-accum 2 \
    --mode-weight 2.5 \
    --group-size 4 \
    --branching \
    --experiment "tosrl_v2_with_branching"

# Training with different dataset
./run_train.sh \
    --config webvoyager_rl \
    --dataset webvoyager \
    --epochs 10
```

**Available Options:**

```
--config CONFIG           Config file (default: webarena_rl)
--dataset DATASET         Dataset name (default: webarena)
--experiment EXP_ID       Experiment name
--epochs N                Number of epochs
--batch-size N            Training batch size
--grad-accum N            Gradient accumulation steps
--eval-freq N             Evaluate every N epochs
--mode MODE               Training mode (tosrl_train, tti_train)
--group-size N            Group size for TOS-RL
--mode-weight N           Mode token weight factor
--branching              Enable prefix-level branching
--help                   Show help message
```

**Output Structure:**

```
logs/training/webarena/EXPERIMENT_ID/
├── training.log              # Full training log
├── checkpoints/
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── best_model.pt
└── results.json              # Training metrics
```

---

### Evaluation Script (`run_eval.sh`)

```bash
# View help
./run_eval.sh --help

# Basic evaluation
./run_eval.sh --checkpoint best_model.pt

# Evaluation with custom parameters
./run_eval.sh \
    --checkpoint checkpoints/best_model.pt \
    --cost balanced \
    --num-tasks 150 \
    --batch-size 8

# Pareto analysis (tests all cost regimes)
./run_eval.sh \
    --checkpoint best_model.pt \
    --pareto \
    --num-tasks 100

# Verbose evaluation with logging
./run_eval.sh \
    --checkpoint best_model.pt \
    --verbose \
    --cost high_efficiency
```

**Available Options:**

```
--checkpoint PATH         Path to checkpoint (required!)
--config CONFIG           Config file (default: webarena_eval)
--dataset DATASET         Dataset name (default: webarena)
--experiment EXP_ID       Experiment name
--cost PREFERENCE         Cost preference (balanced, high_efficiency, high_success)
--num-tasks N             Number of evaluation tasks
--batch-size N            Evaluation batch size
--mode MODE               Evaluation mode (tosrl_eval, tti_eval)
--pareto                  Run Pareto analysis
--verbose                 Verbose output
--help                    Show help message
```

**Output Structure:**

```
logs/evaluation/webarena/EXPERIMENT_ID/
├── evaluation.log            # Full evaluation log
├── results.json              # Evaluation results
├── metrics.json              # Success rate, steps, tokens
└── pareto_analysis.json      # (if --pareto enabled)
```

**Cost Preferences:**

| Preference | Environment Cost | Token Cost | Use Case |
|------------|------------------|-----------|----------|
| `high_efficiency` | 0.1 | 0.01 | Minimize compute |
| `balanced` | 0.01 | 0.001 | Standard use |
| `high_success` | 0.001 | 0.0001 | Maximize accuracy |

---

### Combined Script (`run_train_eval.sh`)

```bash
# View help
./run_train_eval.sh --help

# Basic training + evaluation
./run_train_eval.sh

# Training 10 epochs, evaluate every 2 epochs
./run_train_eval.sh --epochs 10 --eval-every 2

# Training with Pareto analysis at end
./run_train_eval.sh --epochs 15 --pareto

# Full experiment with all options
./run_train_eval.sh \
    --epochs 20 \
    --eval-every 1 \
    --batch-size 8 \
    --branching \
    --pareto \
    --num-eval-tasks 100 \
    --experiment "tosrl_full_exp"

# WebVoyager training + evaluation
./run_train_eval.sh \
    --dataset webvoyager \
    --train-config webvoyager_rl \
    --eval-config webvoyager_eval \
    --epochs 10
```

**Available Options:**

```
--train-config CONFIG         Training config file
--eval-config CONFIG          Evaluation config file
--dataset DATASET             Dataset name
--experiment EXP_ID           Experiment name
--epochs N                    Number of training epochs
--eval-every N                Evaluate every N epochs
--batch-size N                Batch size
--grad-accum N                Gradient accumulation steps
--num-eval-tasks N            Tasks per evaluation
--train-mode MODE             Training mode
--eval-mode MODE              Evaluation mode
--mode-weight N               Mode token weight
--group-size N                Group size
--branching                   Enable branching
--pareto                      Pareto analysis at end
--verbose                     Verbose output
--help                        Show help message
```

**Output Structure:**

```
logs/train_eval/webarena/EXPERIMENT_ID/
├── train_eval.log            # Combined log
├── training/
│   ├── training.log
│   └── checkpoints/
├── evaluation/
│   ├── evaluation.log
│   ├── results.json
│   └── metrics.json
└── checkpoints/
    ├── best_model.pt
    └── [all training checkpoints]
```

---

## 🎯 Common Workflows

### Workflow 1: Full Baseline Training

```bash
# Step 1: Train for 10 epochs
./run_train.sh \
    --experiment "tosrl_baseline" \
    --epochs 10

# Step 2: Evaluate best model
./run_eval.sh \
    --checkpoint logs/training/webarena/tosrl_baseline/checkpoints/best_model.pt \
    --pareto

# Step 3: Check results
cat logs/evaluation/webarena/tosrl_baseline_pareto/pareto_analysis.json
```

### Workflow 2: Hyperparameter Tuning

```bash
# Test different mode weights
for weight in 1.0 2.0 3.0; do
    ./run_train.sh \
        --epochs 5 \
        --mode-weight $weight \
        --experiment "tosrl_weight_${weight}"
done

# Evaluate all variants
for weight in 1.0 2.0 3.0; do
    ./run_eval.sh \
        --checkpoint logs/training/webarena/tosrl_weight_${weight}/checkpoints/best_model.pt \
        --experiment "eval_weight_${weight}"
done
```

### Workflow 3: Incremental Improvement

```bash
# Initial training + evaluation
./run_train_eval.sh \
    --epochs 5 \
    --eval-every 1 \
    --experiment "tosrl_v1"

# Enable branching and continue
./run_train.sh \
    --epochs 10 \
    --branching \
    --experiment "tosrl_v2_with_branching" \
    --checkpoint logs/training/webarena/tosrl_v1/checkpoints/best_model.pt

# Final evaluation
./run_eval.sh \
    --checkpoint logs/training/webarena/tosrl_v2_with_branching/checkpoints/best_model.pt \
    --pareto
```

### Workflow 4: Quick Experiment

```bash
# Train for 2 epochs, evaluate every epoch
./run_train_eval.sh \
    --epochs 2 \
    --eval-every 1 \
    --num-eval-tasks 20 \
    --batch-size 4 \
    --experiment "quick_test"
```

---

## 🔧 Advanced Usage

### Using Custom Configs

Create custom config file: `scripts/config/main/my_config.yaml`

```yaml
defaults:
  - default
  - _self_

experiment_id: "my_experiment"
epochs: 15
batch_size: 8
learning_rate: 1e-5
# ... other settings
```

Then use it:

```bash
./run_train.sh --config my_config
```

### Environment Variables

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Add to Python path
export PYTHONPATH="/Users/luungoc/Project/TTI:$PYTHONPATH"

# Then run scripts
./run_train.sh --epochs 10
```

### Running on Cluster

```bash
# sbatch submission
cat > submit_training.slurm << EOF
#!/bin/bash
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --output=training.log

cd /Users/luungoc/Project/TTI/scripts
./run_train.sh --epochs 20 --batch-size 16
EOF

sbatch submit_training.slurm
```

---

## 📊 Results Analysis

### Training Results

```bash
# View training metrics
cat logs/training/webarena/EXPERIMENT_ID/results.json | python -m json.tool

# Plot learning curves
python -c "
import json
with open('logs/training/webarena/EXPERIMENT_ID/results.json') as f:
    results = json.load(f)
    print('Training metrics:', results.keys())
"
```

### Evaluation Results

```bash
# View evaluation metrics
cat logs/evaluation/webarena/EXPERIMENT_ID/metrics.json | python -m json.tool

# Pareto analysis
cat logs/evaluation/webarena/EXPERIMENT_ID/pareto_analysis.json | python -m json.tool
```

### Comparison

```bash
# Compare multiple experiments
python << EOF
import json
import os

exp_ids = ["tosrl_baseline", "tosrl_v2", "tosrl_with_branching"]

for exp_id in exp_ids:
    metrics_file = f"logs/evaluation/webarena/{exp_id}/metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
            print(f"{exp_id}:")
            print(f"  Success: {metrics['success_rate']:.2%}")
            print(f"  Avg Steps: {metrics['mean_steps']:.1f}")
EOF
```

---

## 🐛 Debugging

### View Script Help

```bash
# Get help for any script
./run_train.sh --help
./run_eval.sh --help
./run_train_eval.sh --help
```

### Check Configs

```bash
# List available configs
ls -la scripts/config/main/

# View a config
cat scripts/config/main/webarena_rl.yaml
```

### Monitor Training

```bash
# Watch log in real-time
tail -f logs/training/webarena/EXPERIMENT_ID/training.log

# Search for errors
grep -i error logs/training/webarena/EXPERIMENT_ID/training.log

# Count lines (progress)
wc -l logs/training/webarena/EXPERIMENT_ID/training.log
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Config not found" | Check `ls scripts/config/main/*.yaml` |
| "Checkpoint not found" | Use full path or check logs/ directory |
| "GPU out of memory" | Reduce `--batch-size` or `--grad-accum` |
| "Slow training" | Use larger batch size or fewer gradient accum steps |
| Permission denied | Run `chmod +x scripts/run_*.sh` |

---

## 📈 Expected Behavior

### Training Output

```
Starting TOS-RL training...
========================================
TOS-RL TRAINING
========================================
Config File:     scripts/config/main/webarena_rl.yaml
Experiment ID:   tosrl_exp_20260518_120000
...

Epoch 1/10
  Loss: 0.3245
  Success: 45.23%

Epoch 2/10
  Loss: 0.2891
  Success: 52.10%

✓ Training completed successfully!
Model saved to: logs/training/webarena/tosrl_exp_20260518_120000
```

### Evaluation Output

```
Starting TOS-RL evaluation...
========================================
TOS-RL EVALUATION
========================================
Cost preference = balanced

Success Rate: 52.30%
Avg Steps: 18.2
Avg Tokens: 2156

Mode Distribution:
  THINK: 25.3%
  OBSERVE: 62.1%
  ANSWER: 12.6%

✓ Evaluation completed successfully!
Results saved to: logs/evaluation/webarena/tosrl_eval_20260518_120000
```

---

## 💾 Checkpoint Management

### Automatic Checkpoints

Checkpoints are saved automatically:
- Every epoch (`save_freq: 1`)
- Best model based on validation

### Manual Checkpointing

```bash
# Find best checkpoint
ls -lt logs/training/webarena/EXPERIMENT_ID/checkpoints/ | head -5

# Use specific checkpoint for evaluation
./run_eval.sh --checkpoint logs/training/webarena/EXPERIMENT_ID/checkpoints/epoch_5.pt
```

### Restore Training

```bash
# Continue training from checkpoint
./run_train.sh \
    --checkpoint logs/training/webarena/exp1/checkpoints/best_model.pt \
    --epochs 20
```

---

## 🎯 Best Practices

1. **Use Descriptive Names**
   ```bash
   ./run_train.sh --experiment "tosrl_baseline_webarena"
   ```

2. **Track Configurations**
   - Save config overrides in experiment name
   - Document why you changed parameters

3. **Evaluate Regularly**
   ```bash
   ./run_train_eval.sh --eval-every 1  # Evaluate every epoch
   ```

4. **Use Pareto Analysis**
   ```bash
   ./run_eval.sh --checkpoint best.pt --pareto
   ```

5. **Monitor GPU/Memory**
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```

6. **Backup Important Results**
   ```bash
   tar -czf results_backup.tar.gz logs/
   ```

---

## 📞 Quick Reference

### Most Common Commands

```bash
# Train
./run_train.sh

# Evaluate
./run_eval.sh --checkpoint best_model.pt

# Train + eval with Pareto
./run_train_eval.sh --pareto

# Get help
./run_train.sh --help
```

### Key Directories

```
Project Root: /Users/luungoc/Project/TTI
Scripts: /Users/luungoc/Project/TTI/scripts
Configs: /Users/luungoc/Project/TTI/scripts/config/main
Logs: /Users/luungoc/Project/TTI/logs
```

---

**Next**: Check out specific experiment results in `logs/` directory!
