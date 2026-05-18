# TOS-RL Scripts Quick Reference

**Location**: `/Users/luungoc/Project/TTI/scripts/`

---

## 🎯 Three Main Scripts

```bash
run_train.sh           # Training only
run_eval.sh            # Evaluation only
run_train_eval.sh      # Training + Evaluation
```

---

## ⚡ Quick Commands

### Training

```bash
# Default training
./run_train.sh

# 10 epochs with custom name
./run_train.sh --epochs 10 --experiment "my_exp"

# With branching
./run_train.sh --epochs 20 --branching

# WebVoyager
./run_train.sh --config webvoyager_rl --dataset webvoyager
```

### Evaluation

```bash
# Single evaluation
./run_eval.sh --checkpoint best_model.pt

# Different cost
./run_eval.sh --checkpoint best.pt --cost high_success

# Pareto analysis (all costs)
./run_eval.sh --checkpoint best.pt --pareto
```

### Train + Eval

```bash
# Train and eval every 2 epochs
./run_train_eval.sh --epochs 10 --eval-every 2

# With Pareto at end
./run_train_eval.sh --epochs 20 --pareto
```

---

## 📋 Key Parameters

### Training (`run_train.sh`)

| Parameter | Default | Examples |
|-----------|---------|----------|
| `--epochs` | 10 | `--epochs 20` |
| `--batch-size` | 4 | `--batch-size 8` |
| `--experiment` | timestamp | `--experiment "exp_v1"` |
| `--branching` | off | `--branching` |
| `--mode-weight` | 2.0 | `--mode-weight 3.0` |
| `--group-size` | 4 | `--group-size 8` |

### Evaluation (`run_eval.sh`)

| Parameter | Default | Examples |
|-----------|---------|----------|
| `--checkpoint` | required | `--checkpoint model.pt` |
| `--cost` | balanced | `--cost high_efficiency` |
| `--num-tasks` | 100 | `--num-tasks 200` |
| `--pareto` | off | `--pareto` |

---

## 🔄 Common Workflows

### Workflow 1: Quick Test
```bash
./run_train_eval.sh --epochs 2 --eval-every 1 --num-eval-tasks 20
```

### Workflow 2: Full Experiment
```bash
./run_train.sh --epochs 20 --branching --experiment "exp_v1"
./run_eval.sh --checkpoint logs/training/webarena/exp_v1/checkpoints/best_model.pt --pareto
```

### Workflow 3: Hyperparameter Sweep
```bash
for weight in 1.0 2.0 3.0; do
  ./run_train.sh --epochs 10 --mode-weight $weight --experiment "weight_${weight}"
done
```

### Workflow 4: Compare Configs
```bash
# WebArena
./run_train.sh --config webarena_rl --experiment "webarena_v1"

# WebVoyager
./run_train.sh --config webvoyager_rl --dataset webvoyager --experiment "webvoyager_v1"
```

---

## 📂 Output Directories

```
logs/
├── training/
│   └── webarena/
│       └── EXPERIMENT_ID/
│           ├── training.log
│           ├── checkpoints/
│           │   ├── epoch_1.pt
│           │   └── best_model.pt
│           └── results.json
│
├── evaluation/
│   └── webarena/
│       └── EXPERIMENT_ID/
│           ├── evaluation.log
│           ├── results.json
│           └── metrics.json
│
└── train_eval/
    └── webarena/
        └── EXPERIMENT_ID/
            ├── train_eval.log
            ├── training/
            ├── evaluation/
            └── checkpoints/
```

---

## 💻 Command Cheatsheet

### Setup
```bash
cd /Users/luungoc/Project/TTI/scripts
chmod +x run_*.sh
```

### List Configs
```bash
ls config/main/*.yaml
```

### View Config
```bash
cat config/main/webarena_rl.yaml
```

### Train with All Options
```bash
./run_train.sh \
  --epochs 15 \
  --batch-size 8 \
  --grad-accum 2 \
  --branching \
  --mode-weight 2.5 \
  --experiment "full_exp"
```

### Full Evaluation Pipeline
```bash
./run_eval.sh \
  --checkpoint checkpoints/best.pt \
  --num-tasks 200 \
  --pareto \
  --verbose
```

### Monitor Training
```bash
tail -f logs/training/webarena/*/training.log
```

### Check Results
```bash
cat logs/evaluation/webarena/*/metrics.json | python -m json.tool
```

---

## 🎯 Cost Preferences (Eval)

| Preference | When to Use |
|-----------|-----------|
| `high_efficiency` | Minimize steps/tokens |
| `balanced` | Standard use |
| `high_success` | Maximize accuracy |

---

## ⚙️ Environment Setup

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Python path (if needed)
export PYTHONPATH="/Users/luungoc/Project/TTI:$PYTHONPATH"

# Then run scripts
./run_train.sh --epochs 10
```

---

## 🔍 Debugging

| Problem | Solution |
|---------|----------|
| Config not found | Check `ls config/main/` |
| Script not found | Run from `/scripts` dir |
| Permission denied | Run `chmod +x run_*.sh` |
| CUDA error | Check GPU: `nvidia-smi` |
| Memory error | Reduce batch size |
| Slow training | Increase batch size |

---

## 📊 View Results

```bash
# Training loss
grep "Loss:" logs/training/webarena/*/training.log

# Evaluation metrics
python -c "import json; print(json.dumps(json.load(open('logs/evaluation/webarena/*/metrics.json')), indent=2))"

# Pareto curve
cat logs/evaluation/webarena/*/pareto_analysis.json | python -m json.tool
```

---

## 🚀 Typical Day

```bash
# Morning: Start training
./run_train.sh --epochs 20 --experiment "daily_exp"

# Afternoon: Check training, start eval
tail -f logs/training/webarena/daily_exp/training.log
./run_eval.sh --checkpoint logs/training/webarena/daily_exp/checkpoints/best_model.pt

# Evening: Analyze and compare results
cat logs/evaluation/webarena/daily_exp/metrics.json | python -m json.tool
```

---

## 📌 Print This Page!

This is your daily reference. Pin it to your monitor! 📌

---

**See Also**:
- `TOSRL_RUN_SCRIPTS_GUIDE.md` - Detailed guide
- `TOSRL_TRAIN_VS_EVALUATE.md` - Training vs evaluation concepts
- `TOSRL_IMPLEMENTATION_GUIDE.md` - TOS-RL concepts

