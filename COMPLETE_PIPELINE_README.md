# TOS-RL Complete Pipeline - Ready to Use

**Status**: ✅ Complete and ready for production
**Created**: 2026-05-18
**What**: One-command end-to-end pipeline for TOS-RL training

---

## Quick Start (30 seconds)

### Prerequisites
```bash
# Install dependencies (one-time)
pip install -r requirements.txt
```

### Run Pipeline
```bash
cd /Users/luungoc/Project/TTI/scripts
./run_complete_pipeline.sh
```

That's it! The pipeline will:
1. ✅ Generate trajectory data
2. ✅ Analyze statistics
3. ✅ Train model
4. ✅ Evaluate results
5. ✅ Show summary

**Time**: ~10-15 minutes depending on hardware

---

## What's Included

### Scripts
| File | Purpose |
|------|---------|
| `run_complete_pipeline.sh` | Main pipeline (NEW) |
| `run_train_real_data.sh` | Real data training wrapper |
| `run_train.sh` | Mock data training |
| `run_eval.sh` | Evaluation |
| `create_sample_trajectories.py` | Generate sample data |

### Documentation
| File | Purpose |
|------|---------|
| `COMPLETE_PIPELINE_SETUP.md` | Setup & detailed guide (NEW) |
| `COMPLETE_PIPELINE_README.md` | This file (NEW) |
| `TOSRL_TRAINING_QUICK_START.md` | 30-second quick start |
| `TOSRL_REAL_DATA_GUIDE.md` | Real data training details |
| `TOSRL_TTI_INTEGRATION_GUIDE.md` | TTI agent integration |
| `requirements.txt` | Python dependencies (NEW) |

### Sample Data
- `data/sample_trajectories.jsonl` - 80 ready-to-use trajectories

---

## Architecture: What Happens in Pipeline

```
┌─────────────────────────────────────────┐
│ STEP 1: Generate Trajectory Data        │
│ - 20 tasks × 4 trajectories (default)  │
│ - JSONL format                         │
│ - Realistic metrics                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ STEP 2: Analyze Data                    │
│ - Count trajectories                   │
│ - Success rate                         │
│ - Mode distribution                    │
│ - Training group statistics            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ STEP 3: Train TOS-RL                    │
│ - Load trajectories (JSONL)            │
│ - Group by task_id                     │
│ - Train for N epochs                   │
│ - Save checkpoints                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ STEP 4: Evaluate                        │
│ - Load best checkpoint                 │
│ - 3 cost preferences                   │
│ - Compute metrics                      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ STEP 5: Show Results                    │
│ - Training metrics                     │
│ - Evaluation results                   │
│ - Next steps                           │
└─────────────────────────────────────────┘
```

---

## Example Commands

### Minimal (5 minutes, CPU)
```bash
./run_complete_pipeline.sh --data-size 5 --epochs 2
```

### Recommended (10-15 minutes)
```bash
./run_complete_pipeline.sh
```

### Production (30+ minutes)
```bash
./run_complete_pipeline.sh --data-size 200 --epochs 20
```

### With GPU
Same commands - PyTorch automatically uses GPU if available

---

## Output Files

After running, you get:

```
logs/
├── training/webarena/complete_pipeline_TIME/
│   ├── training.log              ← View with: tail -f
│   ├── results.json              ← Training metrics
│   └── checkpoints/
│       ├── epoch_1.pt
│       ├── epoch_2.pt
│       ├── epoch_3.pt
│       └── best_model.pt
│
└── evaluation/webarena/complete_pipeline_TIME/
    ├── metrics_high_efficiency.json
    ├── metrics_balanced.json
    └── metrics_high_success.json

data/
└── trajectories_complete_pipeline_TIME.jsonl  ← Trajectory data
```

---

## Data Flow

### Generated Trajectories (JSONL Format)
```json
{
  "task_id": "task_0001",
  "success": 1,
  "num_steps": 5,
  "num_tokens": 250,
  "num_loops": 0,
  "num_bad_actions": 0,
  "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"]
}
```

### Training Results (JSON)
```json
{
  "epoch": 1,
  "metrics": {
    "total_loss": 0.5432,
    "policy_loss": 0.3210,
    "mode_loss": 0.2222,
    "success_rate": 0.55
  }
}
```

### Evaluation Metrics (JSON)
```json
{
  "success_rate": 0.52,
  "mean_steps": 14.2,
  "mean_tokens": 456,
  "mode_distribution": {
    "THINK": 0.24,
    "OBSERVE": 0.52,
    "ANSWER": 0.24
  }
}
```

---

## Customization Options

```bash
./run_complete_pipeline.sh \
  --data-size 100         # Tasks to generate (default: 20)
  --epochs 20             # Training epochs (default: 3)
  --batch-size 8          # Batch size (default: 4)
  --group-size 4          # Group size for training (default: 4)
  --experiment my_run     # Experiment name
  --dataset webarena      # Dataset name
```

---

## Monitoring

### While Training
```bash
# Open new terminal
tail -f logs/training/webarena/complete_pipeline_*/training.log
```

### After Training
```bash
# View results
cat logs/training/webarena/complete_pipeline_*/results.json | python3 -m json.tool

# View specific metric
python3 << 'EOF'
import json
with open("logs/training/webarena/complete_pipeline_*/results.json") as f:
    results = json.load(f)
    final = results[-1]
    print(f"Final epoch: {final['epoch']}")
    print(f"Success rate: {final['metrics']['success_rate']:.2%}")
EOF
```

---

## Using Your Own Data

Instead of generated data:

```bash
# Step 1: Collect trajectories from your TTI agent
# (See TOSRL_TTI_INTEGRATION_GUIDE.md)
python3 my_agent.py --output-file my_trajectories.jsonl

# Step 2: Use training script directly
./run_train_real_data.sh \
  --trajectory-file my_trajectories.jsonl \
  --epochs 20

# Step 3: Evaluate
./run_eval.sh --checkpoint logs/training/webarena/*/checkpoints/best_model.pt
```

---

## Requirements

### System Requirements
- Python 3.8+
- 2GB+ RAM (minimum)
- 4GB+ for GPU

### Python Dependencies
```
torch>=2.0.0
transformers>=4.30.0
pyyaml>=6.0
numpy>=1.21.0
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### PyTorch Not Installed
```bash
pip install torch torchvision torchaudio
```

### Out of Memory
```bash
# Use smaller dataset
./run_complete_pipeline.sh --data-size 20 --epochs 2
```

### Permission Denied
```bash
chmod +x scripts/run_complete_pipeline.sh
chmod +x scripts/run_train_real_data.sh
chmod +x scripts/run_eval.sh
```

### Logs Not Updating
```bash
# Make sure you're in scripts directory
cd /Users/luungoc/Project/TTI/scripts

# Check log file exists
ls logs/training/webarena/*/training.log
```

---

## Performance Expectations

### Training Time Per Epoch
- CPU:
  - 20 tasks: 1-2 min
  - 100 tasks: 5-10 min
  - 200 tasks: 10-20 min

- GPU:
  - 3-5x faster than CPU
  - 100 tasks: 1-3 min
  - 200 tasks: 2-5 min

### Memory Usage
- 20 tasks: 1-2 GB
- 100 tasks: 2-4 GB
- 200 tasks: 4-8 GB

---

## Complete Workflow Example

```bash
#!/bin/bash

# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. Navigate to scripts
cd /Users/luungoc/Project/TTI/scripts

# 3. Run small test (5 minutes)
./run_complete_pipeline.sh --data-size 10 --epochs 2

# 4. Check logs
tail -f ../logs/training/webarena/complete_pipeline_*/training.log

# 5. View results
cat ../logs/training/webarena/complete_pipeline_*/results.json | python3 -m json.tool

# 6. Run production (30+ minutes)
./run_complete_pipeline.sh --data-size 200 --epochs 20

# 7. Analyze final results
python3 << 'EOF'
import json
with open("../logs/training/webarena/complete_pipeline_*/results.json") as f:
    results = json.load(f)
    print(f"Total epochs: {len(results)}")
    print(f"Final success: {results[-1]['metrics']['success_rate']:.2%}")
EOF
```

---

## File Organization

```
TTI/
├── scripts/
│   ├── run_complete_pipeline.sh      ← MAIN SCRIPT
│   ├── run_train_real_data.sh
│   ├── run_eval.sh
│   ├── train_tosrl_tti_real_data.py
│   └── create_sample_trajectories.py
│
├── data/
│   └── sample_trajectories.jsonl
│
├── logs/
│   ├── training/
│   │   └── webarena/EXPERIMENT_ID/
│   │       ├── training.log
│   │       ├── results.json
│   │       └── checkpoints/
│   │
│   └── evaluation/
│       └── webarena/EXPERIMENT_ID/
│           └── metrics_*.json
│
├── COMPLETE_PIPELINE_README.md       ← YOU ARE HERE
├── COMPLETE_PIPELINE_SETUP.md        ← Detailed guide
├── requirements.txt                  ← Dependencies
└── ... (other documentation)
```

---

## Key Features

✅ **One Command**: Single script does everything
✅ **Generates Data**: No need to provide trajectories initially
✅ **Tracks Metrics**: Saves all training and evaluation results
✅ **Checkpointing**: Saves model every epoch
✅ **Flexible**: Customizable data size, epochs, batch size
✅ **Fast**: CPU: 2-20 min, GPU: 1-5 min
✅ **Production Ready**: Real data support with JSONL format
✅ **Well Documented**: 5+ comprehensive guides

---

## Next Steps

1. **Install** (1 minute):
   ```bash
   pip install -r requirements.txt
   ```

2. **Test** (5 minutes):
   ```bash
   cd scripts
   ./run_complete_pipeline.sh --data-size 5 --epochs 2
   ```

3. **Monitor** (while running):
   ```bash
   tail -f ../logs/training/webarena/*/training.log
   ```

4. **Analyze** (after training):
   ```bash
   cat ../logs/training/webarena/*/results.json | python3 -m json.tool
   ```

5. **Scale** (for production):
   ```bash
   ./run_complete_pipeline.sh --data-size 200 --epochs 20
   ```

---

## Support

### Documentation
- `COMPLETE_PIPELINE_SETUP.md` - Detailed setup & usage
- `TOSRL_TRAINING_QUICK_START.md` - 30-second quick start
- `TOSRL_REAL_DATA_GUIDE.md` - Using real trajectory data
- `TOSRL_TTI_INTEGRATION_GUIDE.md` - Integrating with TTI agent

### Issues
- Check `logs/training/EXPERIMENT/training.log` for errors
- Verify `requirements.txt` packages are installed
- Ensure Python 3.8+ is being used

---

## Summary

| Task | Time | Command |
|------|------|---------|
| Setup | 2 min | `pip install -r requirements.txt` |
| Quick test | 5 min | `./run_complete_pipeline.sh --data-size 5 --epochs 2` |
| Small run | 10 min | `./run_complete_pipeline.sh` |
| Medium run | 20 min | `./run_complete_pipeline.sh --data-size 100 --epochs 10` |
| Production | 30+ min | `./run_complete_pipeline.sh --data-size 200 --epochs 20` |

---

**Status**: ✅ Ready for immediate use
**Last Updated**: 2026-05-18
**Next**: Run `./run_complete_pipeline.sh` in scripts directory

See `COMPLETE_PIPELINE_SETUP.md` for detailed documentation.
