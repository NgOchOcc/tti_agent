# TOS-RL Setup & Environment Guide

## ⚠️ Dependencies Required

### Core Requirements

```bash
pip install torch>=2.0.0      # Deep learning framework
pip install pyyaml            # Config file handling
pip install hydra-core        # Config management (optional)
pip install numpy>=1.21.0     # Numerical computing
```

### Installation Command

```bash
# All at once
pip install torch pyyaml hydra-core numpy

# Or if you already have PyTorch installed
pip install pyyaml hydra-core
```

---

## 🔧 Verify Installation

```bash
# Check PyTorch
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

# Check YAML
python3 -c "import yaml; print('✓ PyYAML installed')"

# Check TOS-RL imports
cd /Users/luungoc/Project/TTI
python3 -c "from tti.tos_rl import TOSRLTrainer; print('✓ TOS-RL imports work')"
```

---

## 🚀 Quick Setup

```bash
# 1. Install dependencies
pip install torch pyyaml

# 2. Go to scripts directory
cd /Users/luungoc/Project/TTI/scripts

# 3. Test training (2 epochs)
./run_train.sh --epochs 2

# 4. Monitor in another terminal
tail -f ../logs/training/webarena/*/training.log

# 5. Evaluate
./run_eval.sh --checkpoint ../logs/training/webarena/*/checkpoints/best_model.pt
```

---

## 📋 What If Installation Fails?

### PyTorch Installation Issues

**Problem**: `No module named 'torch'`

**Solution 1: CPU-only (fast installation)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Solution 2: GPU (CUDA 11.8)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Solution 3: GPU (CUDA 12.1)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### YAML Issues

**Problem**: `No module named 'yaml'`

**Solution**:
```bash
pip install pyyaml
```

---

## 🐍 Python Version Requirements

- **Python 3.8+** recommended
- **Python 3.10+** preferred

Check your version:
```bash
python3 --version
```

---

## 🎯 Environment Setup for Different Scenarios

### Scenario 1: Local Development

```bash
# Create virtual environment
python3 -m venv ~/venv_tosrl

# Activate
source ~/venv_tosrl/bin/activate

# Install dependencies
pip install torch pyyaml numpy

# Test
cd /Users/luungoc/Project/TTI/scripts
./run_train.sh --epochs 2
```

### Scenario 2: GPU Training (NVIDIA)

```bash
# Check GPU
nvidia-smi

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other deps
pip install pyyaml numpy

# Run with GPU
cd /Users/luungoc/Project/TTI/scripts
export CUDA_VISIBLE_DEVICES="0,1"  # Use GPUs 0 and 1
./run_train.sh --epochs 10 --batch-size 16
```

### Scenario 3: HPC/Cluster

```bash
# Load modules (example for SLURM cluster)
module load python/3.10
module load cuda/11.8

# Create virtual environment
python3 -m venv ~/venv_tosrl
source ~/venv_tosrl/bin/activate

# Install
pip install torch pyyaml numpy

# Create SLURM job script
cat > train_job.sh << 'EOF'
#!/bin/bash
#SBATCH --gpus=4
#SBATCH --time=24:00:00

cd /Users/luungoc/Project/TTI/scripts
./run_train.sh --epochs 20 --batch-size 16
EOF

# Submit
sbatch train_job.sh
```

---

## 🔍 Troubleshooting

### Script Fails Immediately

**Problem**: `train_tosrl_tti.py: No such file or directory`

**Cause**: Scripts are not in `/Users/luungoc/Project/TTI/scripts/`

**Solution**:
```bash
ls -la /Users/luungoc/Project/TTI/scripts/*.py
# Should show: train_tosrl_tti.py, eval_tosrl_tti.py, analyze_pareto.py
```

### Import Errors

**Problem**: `No module named 'tti.tos_rl'`

**Cause**: Project directory not in Python path

**Solution**:
```bash
# Set Python path
export PYTHONPATH=/Users/luungoc/Project/TTI:$PYTHONPATH

# Then run scripts
./run_train.sh --epochs 5
```

### PyTorch Not Found During Training

**Problem**: Training starts but crashes with `No module named 'torch'`

**Cause**: Dependencies installed in wrong environment

**Solution**:
```bash
# Verify Python being used
which python3

# Verify torch is installed
python3 -c "import torch; print(torch.__version__)"

# If not installed, install it
pip install torch
```

### YAML Config Not Found

**Problem**: `Error: Config file not found`

**Cause**: Config files not in expected location

**Solution**:
```bash
# Check configs exist
ls /Users/luungoc/Project/TTI/scripts/config/main/

# Should show: default.yaml, webarena_rl.yaml, webarena_eval.yaml, etc.
```

---

## 📊 Verify Complete Setup

Run this verification script:

```bash
#!/bin/bash
echo "=== TOS-RL Setup Verification ==="

# 1. Check Python
echo -n "Python: "
python3 --version

# 2. Check PyTorch
echo -n "PyTorch: "
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED"

# 3. Check PyYAML
echo -n "PyYAML: "
python3 -c "import yaml; print('✓ installed')" 2>/dev/null || echo "NOT INSTALLED"

# 4. Check project structure
echo ""
echo "Project Structure:"
test -d /Users/luungoc/Project/TTI/tti/tos_rl && echo "  ✓ tti/tos_rl/" || echo "  ✗ tti/tos_rl/ missing"
test -f /Users/luungoc/Project/TTI/scripts/run_train.sh && echo "  ✓ scripts/run_train.sh" || echo "  ✗ scripts/run_train.sh missing"
test -f /Users/luungoc/Project/TTI/scripts/train_tosrl_tti.py && echo "  ✓ scripts/train_tosrl_tti.py" || echo "  ✗ scripts/train_tosrl_tti.py missing"
test -d /Users/luungoc/Project/TTI/scripts/config/main && echo "  ✓ scripts/config/main/" || echo "  ✗ scripts/config/main/ missing"

# 5. Check imports
echo ""
echo "Imports:"
python3 -c "from tti.tos_rl import TOSRLTrainer; print('  ✓ TOS-RL imports work')" 2>/dev/null || echo "  ✗ TOS-RL import failed"

echo ""
echo "=== Setup Verification Complete ==="
```

Save as `verify_setup.sh` and run:

```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

---

## 💡 Common Tips

### Tip 1: Use Virtual Environment

```bash
# Create
python3 -m venv ~/tosrl_env

# Activate
source ~/tosrl_env/bin/activate

# Install
pip install torch pyyaml

# Deactivate when done
deactivate
```

### Tip 2: Install Requirements File

Create `requirements.txt`:
```
torch>=2.0.0
pyyaml>=5.4
numpy>=1.21.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Tip 3: Check CUDA Support

```bash
# Check if GPU available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU count
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check GPU names
python3 -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

---

## ✅ Success Criteria

You're ready when:

- [x] `python3 --version` shows 3.8+
- [x] `python3 -c "import torch"` works
- [x] `python3 -c "import yaml"` works
- [x] `/Users/luungoc/Project/TTI/scripts/run_train.sh` is executable
- [x] Config files exist in `/Users/luungoc/Project/TTI/scripts/config/main/`
- [x] `./run_train.sh --epochs 2` runs without import errors

---

## 🚀 Next Steps

1. ✅ Install dependencies
2. ✅ Verify setup
3. ✅ Run: `./run_train.sh --epochs 2` (quick test)
4. ✅ Monitor: `tail -f logs/training/webarena/*/training.log`
5. ✅ Evaluate: `./run_eval.sh --checkpoint logs/.../best_model.pt`

---

## 📞 Support

If you encounter issues:

1. Check `verify_setup.sh` output
2. Check Python version (`python3 --version`)
3. Check PYTHONPATH (`echo $PYTHONPATH`)
4. Check file locations (`ls -R scripts/config/main/`)
5. Try CPU-only PyTorch first (easier to install)

---

**Status**: Setup guide complete
**Date**: 2026-05-18
