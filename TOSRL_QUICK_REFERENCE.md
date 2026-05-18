# TOS-RL Quick Reference Card

## 🚀 TL;DR

**TOS-RL** = LLM learns when to [THINK], [OBSERVE], or [ANSWER] through cost-aware RL. **No external networks.**

## 📦 Installation

```python
from tti.tos_rl import (
    TOSRLTrainer, TOSRLConfig,
    CostAwareUtility,
    BudgetState, TokenModeFormatter,
    PrefixBrancher, BranchCollector
)
```

## ⚙️ Configuration

```python
config = TOSRLConfig(
    group_size=4,              # K trajectories per task
    learning_rate=1e-5,
    mode_token_weight=2.0,     # Upweight mode tokens
    use_branching=True,
    max_prefixes_per_task=3,
    branches_per_prefix=3,

    # Cost ranges (randomized during training)
    lambda_env_range=(0.001, 0.1),      # Cost per step
    lambda_tok_range=(0.0001, 0.01),    # Cost per token
    lambda_loop_range=(0.01, 0.5),      # Repeated action cost
    lambda_bad_range=(0.1, 1.0),        # Invalid action cost
)
```

## 🔄 Training Loop

```python
trainer = TOSRLTrainer(config)

# 1. Collect K trajectories
trajectories = [traj1, traj2, traj3, traj4]  # K=4

# 2. Compute utilities and advantages
utilities, costs = trainer.process_trajectories(trajectories)
advantages = trainer.compute_group_advantages(utilities)

# 3. Prepare batch
batch = {
    "log_probs": log_probs,
    "log_probs_old": log_probs_old,
    "log_probs_ref": log_probs_ref,
    "advantages": advantages,
    "mode_log_probs": mode_log_probs,
    "mode_probs": mode_probs,
}

# 4. Training step
metrics = trainer.training_step(batch)
```

## 🎯 Trajectory Format

```python
trajectory = {
    "task_id": "task_123",
    "success": 1,              # 0 or 1
    "num_steps": 5,            # [OBSERVE] count
    "num_tokens": 256,         # Total generated
    "num_loops": 0,            # Repeated actions
    "num_bad_actions": 0,      # Invalid actions
    "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
}
```

## 🎤 Mode Token Formatting

```python
# Correct format (must start with mode token)
output = "[THINK] Let me reason..."
output = "[OBSERVE] Click the button"
output = "[ANSWER] The answer is..."

# Parsing
from tti.tos_rl import TokenModeFormatter
mode, content = TokenModeFormatter.parse_mode_and_content(output)
# Returns: ("OBSERVE", "Click the button")
```

## 💰 Budget State

```python
from tti.tos_rl import BudgetState

budget = BudgetState(
    env_interactions=30,    # Remaining [OBSERVE] steps
    tokens=4096,           # Remaining tokens
    time=300.0,            # Remaining time (seconds)
)

# Use in prompt
suffix = f"""
Remaining: {int(budget.env_interactions)} steps, {int(budget.tokens)} tokens
"""
```

## 🧮 Key Equations

### Cost-Aware Utility
```
U(τ) = R_task - λ_env·N_env - λ_tok·N_tok - λ_loop·N_loop - λ_bad·N_bad
```

### Group-Relative Advantage
```
Â_i = (U_i - mean(U)) / std(U)
(No critic network!)
```

### GRPO Loss
```
L = -Σ min(ρ_i * Â_i, clip(ρ_i) * Â_i)
    + λ_KL * KL(π || π_ref)
    - β_H * H(π)
```

### Mode Emphasis
```
L_mode = -α_m * Σ log π(m) * Â
(Upweight mode token learning)
```

## 📊 Inference

```python
from tti.tos_rl import TOSRLInference

inference = TOSRLInference(
    max_steps=30,
    max_tokens=4096,
)

result = inference.run_episode(
    task="Find the price",
    llm_fn=model.generate,
    initial_history="",
    initial_budget=BudgetState(env_interactions=30, tokens=4096),
)

# Returns
print(result['success'])      # 0 or 1
print(result['num_steps'])    # Steps taken
print(result['num_tokens'])   # Tokens used
print(result['modes'])        # List of modes
print(result['final_answer']) # Final answer
```

## 🔧 Common Tasks

### Parse Mode from LLM Output
```python
mode, content = TokenModeFormatter.parse_mode_and_content(llm_output)

if mode == "THINK":
    # Internal reasoning
    pass
elif mode == "OBSERVE":
    # Environment action
    execute_action(content)
elif mode == "ANSWER":
    # Final answer
    answer = extract_answer(content)
    check_success(answer)
```

### Track Metrics
```python
metrics = {
    "success": success,
    "num_steps": num_steps,
    "num_tokens": total_tokens,
    "num_loops": num_repeated_actions,
    "num_bad_actions": num_invalid_actions,
}
```

### Budget Conditioning
```python
budget_text = f"""
Remaining interactions: {int(budget.env_interactions)}
Remaining tokens: {int(budget.tokens)}

Use modes wisely!
"""

prompt = f"{task}\n{history}\n{budget_text}"
```

## ⚡ Performance Tips

1. **Start without branching** (`use_branching=False`)
2. **Upweight mode tokens** (`mode_token_weight=2.0`)
3. **Use K≥4** trajectories per task
4. **Randomize costs widely** (range not too narrow)
5. **Monitor mode distribution** (all three should be used)

## 🐛 Debugging

| Problem | Cause | Fix |
|---------|-------|-----|
| Always OBSERVE | Costs too low | ↑ `lambda_env` |
| Always ANSWER | Costs too high | ↓ `lambda_env` |
| No mode learning | Weight too low | ↑ `mode_token_weight` |
| No improvement | Trajectories not grouped | Check K=4 grouping |
| Memory error | Batch too large | ↓ `group_size` |

## 📚 Documentation Map

| Need | File |
|------|------|
| Quick start | This file |
| Concepts | `TOSRL_IMPLEMENTATION_GUIDE.md` |
| Architecture | `TOSRL_IMPLEMENTATION_COMPLETE.md` |
| API | `tti/tos_rl/README.md` |
| Integration | `TOSRL_INTEGRATION_WITH_TTI.md` |
| Checklist | `TOSRL_INTEGRATION_CHECKLIST.md` |
| Overview | `README_TOSRL.md` |

## 🎯 Integration Phases

1. **Prepare** (1 day)
   - [ ] Read docs
   - [ ] Setup environment

2. **Collect** (3-5 days)
   - [ ] Add mode token tracking
   - [ ] Track metrics

3. **Train** (1-2 days)
   - [ ] Implement batch prep
   - [ ] Integrate training

4. **Infer** (1 day)
   - [ ] Add inference
   - [ ] Test on validation

5. **Evaluate** (2-3 days)
   - [ ] Compare baselines
   - [ ] Analyze results

## 💡 Key Insights

- **No critic networks** - advantages from ranking
- **Single policy** - handles multiple cost regimes
- **LLM-only** - mode tokens are regular output
- **Cost-aware** - learns success-cost tradeoff
- **Learned stopping** - naturally learns when to stop

## ✅ Success Criteria

- ✓ Trajectories collected with mode tokens
- ✓ Training loop runs, loss decreases
- ✓ Success rate improves over baseline
- ✓ Mode distribution is diverse
- ✓ Lower cost for same success

## 🆘 Getting Help

1. Check **this file** first (you're here!)
2. Check `TOSRL_IMPLEMENTATION_GUIDE.md` for concepts
3. Check `TOSRL_INTEGRATION_WITH_TTI.md` for integration
4. Check code comments in `tti/tos_rl/`
5. Check `TOSRL_INTEGRATION_CHECKLIST.md` for step verification

## 📞 Quick Command Reference

```bash
# Verify TOS-RL is installed
python -c "from tti.tos_rl import *; print('✓')"

# Run example (standalone)
python scripts/train_tos_rl_example.py --epochs 3 --demo-only

# Run integration template (with mock TTI)
python scripts/train_tosrl_with_tti_integration.py --epochs 2

# Check module sizes
find tti/tos_rl -name "*.py" -exec wc -l {} +
```

## 🎓 Learning Path

```
1. README_TOSRL.md (10 min)
   ↓
2. TOSRL_IMPLEMENTATION_GUIDE.md (15 min)
   ↓
3. scripts/train_tos_rl_example.py (10 min)
   ↓
4. TOSRL_INTEGRATION_WITH_TTI.md (30 min)
   ↓
5. TOSRL_INTEGRATION_CHECKLIST.md (30 min)
   ↓
6. Start implementing! (1-2 weeks)
```

## 🎯 Core Concepts at a Glance

| Concept | Meaning | Example |
|---------|---------|---------|
| [THINK] | Reasoning step | "Let me think about this..." |
| [OBSERVE] | Environment action | "Click the button" |
| [ANSWER] | Stop & answer | "The answer is X" |
| **K** | Trajectories per task | 4 |
| **Cost** | Penalty for resource use | λ·N |
| **Utility** | Success - Cost | U = 1 - 0.01·5 |
| **Advantage** | Relative utility | (U - mean) / std |
| **GRPO** | Optimization algorithm | PPO variant |
| **Branching** | Credit assignment | Forced mode continuations |

---

**Print this card, bookmark it, use it! 📌**

*Last updated: 2026-05-18*
