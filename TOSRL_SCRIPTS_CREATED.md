# TOS-RL Train vs Eval Scripts - Creation Summary

**Date Created**: 2026-05-18
**Status**: ✅ **COMPLETE**

---

## 📋 Summary

Created **3 production-ready shell scripts** that read from your existing YAML configs in `scripts/config/main/` and integrate with TOS-RL training/evaluation.

---

## 📂 Files Created

### Shell Scripts (Executable)

Located in: `/Users/luungoc/Project/TTI/scripts/`

1. **`run_train.sh`** (6.1KB, executable)
   - Purpose: Training-only mode
   - Reads: `scripts/config/main/webarena_rl.yaml` (+ others)
   - Usage: `./run_train.sh --epochs 10 --branching`

2. **`run_eval.sh`** (7.6KB, executable)
   - Purpose: Evaluation-only mode
   - Reads: `scripts/config/main/webarena_eval.yaml` (+ others)
   - Usage: `./run_eval.sh --checkpoint best_model.pt --pareto`

3. **`run_train_eval.sh`** (13KB, executable)
   - Purpose: Combined training + evaluation
   - Reads: Both train and eval configs
   - Usage: `./run_train_eval.sh --epochs 20 --eval-every 2`

### Documentation

Located in: `/Users/luungoc/Project/TTI/`

1. **`TOSRL_RUN_SCRIPTS_GUIDE.md`** (14KB)
   - Comprehensive guide to all three scripts
   - All options and parameters explained
   - Workflows and examples
   - Advanced usage and debugging

2. **`TOSRL_SCRIPTS_QUICK_REFERENCE.md`** (5.5KB)
   - Quick command reference
   - Common workflows
   - Quick commands (print this!)
   - Cheatsheet format

3. **`TOSRL_TRAIN_VS_EVALUATE.md`** (17KB)
   - Training vs evaluation concepts
   - Complete code examples
   - Metrics to track
   - Common patterns

---

## ⚡ Quick Start

```bash
cd /Users/luungoc/Project/TTI/scripts

# Training
./run_train.sh --epochs 10

# Evaluation
./run_eval.sh --checkpoint best_model.pt

# Combined
./run_train_eval.sh --epochs 20 --eval-every 2 --pareto
```

---

## 🔧 How It Works

### Config Integration

Scripts automatically read YAML configs:

```
scripts/config/main/
├── default.yaml              ← Base settings
├── webarena_rl.yaml          ← Used by run_train.sh
├── webarena_eval.yaml        ← Used by run_eval.sh
├── webvoyager_rl.yaml
└── webvoyager_eval.yaml
```

### Command-Line Overrides

All config values can be overridden from command line:

```bash
# Config has: batch_size: 4
# Override it:
./run_train.sh --batch-size 8 --epochs 20
```

### Automatic Logging

All output goes to:
```
logs/
├── training/webarena/EXPERIMENT_ID/
│   ├── training.log
│   ├── checkpoints/
│   └── results.json
│
├── evaluation/webarena/EXPERIMENT_ID/
│   ├── evaluation.log
│   ├── metrics.json
│   └── pareto_analysis.json
│
└── train_eval/webarena/EXPERIMENT_ID/
    ├── train_eval.log
    ├── training/
    ├── evaluation/
    └── checkpoints/
```

---

## 🎯 Key Features

✅ **Reads YAML configs** from `scripts/config/main/`
✅ **Automatic logging** to `logs/` directory
✅ **Colored output** for easy reading (BLUE, GREEN, RED, YELLOW)
✅ **Error handling** and validation
✅ **Train mode** - Training only
✅ **Eval mode** - Evaluation only
✅ **Combined mode** - Training + evaluation together
✅ **Pareto analysis** - Test all cost regimes
✅ **Hyperparameter overrides** from CLI
✅ **Checkpoint management** - Save and restore
✅ **Multiple datasets** - WebArena, WebVoyager
✅ **Comprehensive help** - `./script.sh --help`

---

## 📊 Common Commands

### Training Only
```bash
./run_train.sh                              # Default
./run_train.sh --epochs 20                  # Custom epochs
./run_train.sh --epochs 15 --branching      # With branching
./run_train.sh --mode-weight 3.0            # Custom mode weight
```

### Evaluation Only
```bash
./run_eval.sh --checkpoint best_model.pt    # Required!
./run_eval.sh --checkpoint best.pt --pareto # Pareto analysis
./run_eval.sh --checkpoint best.pt --cost high_success
```

### Combined
```bash
./run_train_eval.sh                         # Default
./run_train_eval.sh --epochs 10 --eval-every 2
./run_train_eval.sh --pareto                # With Pareto
```

---

## 🎓 Configuration Parameters

### Available in all scripts

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--config` | webarena_rl | YAML config file |
| `--dataset` | webarena | Dataset name (webarena/webvoyager) |
| `--experiment` | timestamp | Experiment ID for logging |
| `--help` | - | Show help message |

### Training only (`run_train.sh`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 4 | Training batch size |
| `--grad-accum` | 2 | Gradient accumulation steps |
| `--eval-freq` | 1 | Evaluation frequency |
| `--mode-weight` | 2.0 | Mode token emphasis factor |
| `--group-size` | 4 | TOS-RL group size |
| `--branching` | off | Enable prefix branching |

### Evaluation only (`run_eval.sh`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--checkpoint` | required | Model checkpoint path |
| `--cost` | balanced | Cost preference (balanced/high_efficiency/high_success) |
| `--num-tasks` | 100 | Number of evaluation tasks |
| `--pareto` | off | Run Pareto analysis |

### Combined (`run_train_eval.sh`)

Supports both train and eval parameters:
- `--epochs` - training epochs
- `--eval-every` - evaluate every N epochs
- `--checkpoint` - resume from checkpoint
- `--pareto` - Pareto analysis at end
- All other parameters from train and eval scripts

---

## 📈 Output Examples

### Training Log
```
Starting TOS-RL training...
========================================
TOS-RL TRAINING
========================================
Config File:     scripts/config/main/webarena_rl.yaml
Experiment ID:   tosrl_exp_20260518_120000
Dataset:         webarena
Mode:            tosrl_train
Epochs:          10
...

Epoch 1/10
  Loss: 0.3245
  Success: 45.23%

Epoch 2/10
  Loss: 0.2891
  Success: 52.10%

✓ Training completed successfully!
```

### Evaluation Results
```
Success Rate: 52.30%
Avg Steps: 18.2
Avg Tokens: 2156

Mode Distribution:
  THINK: 25.3%
  OBSERVE: 62.1%
  ANSWER: 12.6%
```

---

## 🔄 Typical Workflows

### Workflow 1: Quick Test (5 minutes)
```bash
./run_train_eval.sh --epochs 2 --eval-every 1 --num-eval-tasks 20
```

### Workflow 2: Full Training (2-3 days)
```bash
./run_train.sh --epochs 20 --branching --experiment "exp_v1"
./run_eval.sh --checkpoint logs/training/webarena/exp_v1/checkpoints/best_model.pt --pareto
```

### Workflow 3: Hyperparameter Tuning (1 day)
```bash
for weight in 1.0 2.0 3.0; do
  ./run_train.sh --epochs 10 --mode-weight $weight --experiment "weight_${weight}"
done
```

### Workflow 4: Compare Datasets (1-2 days)
```bash
./run_train.sh --config webarena_rl --dataset webarena --epochs 10
./run_train.sh --config webvoyager_rl --dataset webvoyager --epochs 10
```

---

## 💾 What Gets Saved

### Training Outputs
- `training.log` - Full training log
- `checkpoints/epoch_N.pt` - Checkpoint after each epoch
- `checkpoints/best_model.pt` - Best checkpoint
- `results.json` - Training metrics

### Evaluation Outputs
- `evaluation.log` - Full evaluation log
- `metrics.json` - Success rate, steps, tokens
- `pareto_analysis.json` - (if --pareto enabled)

### Combined Outputs
- Both training and evaluation outputs
- `train_eval.log` - Combined log

---

## 🚀 Getting Started

1. **Read the quick reference** (5 min)
   ```bash
   cat TOSRL_SCRIPTS_QUICK_REFERENCE.md
   ```

2. **Try a quick test** (5 min)
   ```bash
   cd /Users/luungoc/Project/TTI/scripts
   ./run_train.sh --epochs 2
   ```

3. **Monitor training** (in another terminal)
   ```bash
   tail -f logs/training/webarena/*/training.log
   ```

4. **Evaluate** (after training)
   ```bash
   ./run_eval.sh --checkpoint logs/training/webarena/*/checkpoints/best_model.pt
   ```

5. **Check results**
   ```bash
   cat logs/evaluation/webarena/*/metrics.json | python -m json.tool
   ```

---

## 📚 Documentation Files

| File | Purpose | Size |
|------|---------|------|
| `TOSRL_SCRIPTS_QUICK_REFERENCE.md` | Daily reference (print this!) | 5.5KB |
| `TOSRL_RUN_SCRIPTS_GUIDE.md` | Comprehensive guide | 14KB |
| `TOSRL_TRAIN_VS_EVALUATE.md` | Train vs eval concepts | 17KB |

---

## ✅ Quality Checklist

- [x] Scripts created and tested
- [x] Executable permissions set
- [x] Reads YAML configs correctly
- [x] Logging works properly
- [x] Error handling included
- [x] Help messages comprehensive
- [x] Colored output implemented
- [x] Config overrides work
- [x] Checkpoint management works
- [x] Documentation complete

---

## 🎯 Next Steps

1. ✅ Read `TOSRL_SCRIPTS_QUICK_REFERENCE.md` (this page)
2. ✅ Try `./run_train.sh --epochs 2` (quick test)
3. ⏳ Read `TOSRL_RUN_SCRIPTS_GUIDE.md` (detailed guide)
4. ⏳ Run your first real experiment
5. ⏳ Evaluate using `./run_eval.sh`
6. ⏳ Analyze results and iterate

---

## 📞 Questions?

- **How do I run training?** → See Quick Start section
- **What configs are available?** → `ls scripts/config/main/`
- **How do I evaluate?** → See `./run_eval.sh --help`
- **Where are the results?** → Check `logs/` directory
- **How do I monitor?** → Use `tail -f logs/.../training.log`

---

## 🎉 Ready to Go!

All three scripts are ready to use. Start with:

```bash
cd /Users/luungoc/Project/TTI/scripts
./run_train.sh --epochs 10
```

The scripts will handle:
- Reading configs
- Creating log directories
- Saving checkpoints
- Logging results
- Error handling

**Enjoy!** 🚀

---

**Status**: ✅ Complete and production-ready
**Location**: `/Users/luungoc/Project/TTI/scripts/`
**Date**: 2026-05-18
