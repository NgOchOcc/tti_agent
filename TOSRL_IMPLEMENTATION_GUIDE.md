# TOS-RL: LLM-Only Think-Observe-Stop RL Implementation Guide

## Overview

TOS-RL is a **minimal, LLM-only framework** for teaching agents when to think, observe, or answer. Unlike modular approaches with separate gate networks or critics, TOS-RL makes mode selection part of the LLM's own token sequence.

**Core Innovation**: Mode tokens `[THINK]`, `[OBSERVE]`, `[ANSWER]` are regular LLM outputs, trained via cost-aware RL. No external networks needed.

## Architecture

```
┌─ Task Instruction ─────────────┐
│                                 │
├─ History (observations/actions) │
│                                 │
├─ Budget (steps, tokens, time)   │
│                                 │
└─> LLM Policy                    │
    └─> [MODE] content            │  <- Mode token is LLM's decision
        ├─> [THINK] reasoning     │     No external gate network!
        ├─> [OBSERVE] action      │
        └─> [ANSWER] final answer │
```

## Files and Modules

```
tti/tos_rl/
├── __init__.py              # Package exports
├── objectives.py            # Cost-aware utility
├── optimization.py          # GRPO (Group Relative Policy Optimization)
├── branching.py            # Prefix-level counterfactual branches
├── training.py             # Main training loop
├── inference.py            # Inference algorithm
└── utils.py                # Budget, costs, formatting
```

## Key Concepts

### 1. Cost-Aware Utility

```
U(τ) = R_task - λ_env·N_env - λ_tok·N_tok - λ_loop·N_loop - λ_bad·N_bad
```

- **R_task**: Task success (0 or 1)
- **N_env**: Number of browser interactions
- **N_tok**: Tokens generated
- **N_loop**: Repeated actions/states
- **N_bad**: Invalid/bad actions

### 2. Group-Relative Advantage (No Critic!)

```
Â_i = (U_i - mean(U_1:K)) / std(U_1:K)
```

Sample K trajectories from same task, compute relative ranking. No separate value network.

### 3. Mode Tokens

Three modes, represented as tokens:

```
[THINK]: Internal reasoning, no environment interaction
         Consumes tokens only, advances no browser steps

[OBSERVE]: Browser action (click, scroll, search, etc.)
           Consumes both tokens and environment step
           Receives new observation

[ANSWER]: Stop and submit final answer
          Terminates episode
```

### 4. Mode Emphasis

Optional upweighting of mode-token learning:

```
L_mode = -α_m · Σ log π(m_t) * Â_i
```

Makes compute-allocation behavior learn faster.

### 5. Prefix-Level Branching

For each prefix in a trajectory:
1. Generate continuations forced to start with [THINK], [OBSERVE], [ANSWER]
2. Compute utilities of branches
3. Train mode tokens using branch-relative advantages

Improves credit assignment without an external critic.

## Quick Start

### 1. Import

```python
from tti.tos_rl import (
    TOSRLTrainer, TOSRLConfig,
    CostAwareUtility,
    BudgetState, TokenModeFormatter
)
```

### 2. Configure

```python
config = TOSRLConfig(
    group_size=4,              # Trajectories per task
    max_prefixes_per_task=3,
    branches_per_prefix=3,
    learning_rate=1e-5,
    mode_token_weight=2.0,     # Upweight mode tokens
    use_branching=True,
)

trainer = TOSRLTrainer(config)
```

### 3. Collect Trajectories

```python
trajectories = [
    {
        "task_id": "task_1",
        "success": 1,              # Task succeeded
        "num_steps": 5,            # Browser interactions
        "num_tokens": 256,         # Tokens generated
        "num_loops": 0,
        "num_bad_actions": 0,
        "modes": ["THINK", "OBSERVE", "OBSERVE", "THINK", "ANSWER"],
    },
    # ... more trajectories
]
```

### 4. Training Step

```python
# Process trajectories
utilities, costs = trainer.process_trajectories(trajectories)

# Group trajectories (4 per task for group-relative advantage)
# Compute advantages
advantages = trainer.compute_group_advantages(utilities)

# Create batch
batch = {
    "log_probs": torch.tensor(log_probs),
    "log_probs_old": torch.tensor(log_probs_old),
    "advantages": torch.tensor(advantages),
    "mode_log_probs": torch.tensor(mode_log_probs),
}

# Training step
metrics = trainer.training_step(batch)
```

### 5. Inference

```python
from tti.tos_rl import TOSRLInference

inference = TOSRLInference(max_steps=30, max_tokens=4096)

# Run episode
result = inference.run_episode(
    task="Find the price of the product",
    llm_fn=llm_model.generate,  # Your LLM
    initial_history="User: Find the price\nAssistant: I'll help.",
    initial_budget=BudgetState(env_interactions=30, tokens=4096),
)

print(f"Success: {result['success']}")
print(f"Steps: {result['num_steps']}")
print(f"Modes: {result['modes']}")
```

## Training Pipeline

### Stage 1: Warm Start (Optional)

```python
# Initialize from supervised web-agent trajectories
# Just to teach action syntax, not the mode decision
```

### Stage 2: Grouped RL on WebArena

```python
# For each task, collect K=4 trajectories
# With randomized cost coefficients
# This trains the LLM to balance success vs cost
```

### Stage 3: Prefix-Level Branching

```python
# Mine high-uncertainty prefixes
# Generate branches with forced modes
# Train using branch-relative advantages
```

### Stage 4: WebVoyager Evaluation

```python
# Test transfer to real-world tasks
# Should see fewer loops, lower interaction cost
```

## Cost Coefficients and Pareto Curves

TOS-RL is trained with **randomized cost coefficients** during training:

```python
# Different deployment preferences
high_efficiency: λ_env=0.1, λ_tok=0.01
balanced: λ_env=0.01, λ_tok=0.001
high_success: λ_env=0.001, λ_tok=0.0001
```

A single trained LLM learns to adapt! At inference, specify cost preference in budget prompt.

## Key Equations

### GRPO Loss
```
L = -1/N Σ_i min(ρ_i * Â_i, clip(ρ_i, 1±ε) * Â_i)
    + λ_KL * KL(π || π_ref)
    - β_H * H(π)
```

### Mode Token Emphasis
```
L_mode = -α_m * Σ_i log π(m_i) * Â_i
```

### Prefix-Level Branching
```
Â^branch_{m,s} = (U(τ^m_s) - mean(U)) / std(U)

L_branch = -Σ_{m,s} [
    α_m log π(m | h_t) +
    Σ_ℓ log π(y^m_{ℓ} | ...)
] * Â^branch_{m,s}
```

## Advantages Over Modular Approaches

| Aspect | Modular (CVI-SDAR) | TOS-RL |
|--------|-------------------|--------|
| Gate network | ✓ (trainable) | ✗ (no separate module) |
| Critic networks | ✓ (V, Q, V_ans) | ✗ (group-relative) |
| Alignment | Separate controller | Integrated in LLM |
| Simplicity | More complex | Minimal |
| Novelty | Value-of-interaction | LLM-only internalization |

## Experimental Hypotheses

1. **Same success, lower cost**: TOS-RL should match TTI success with fewer steps
2. **Better Pareto frontier**: Better success-cost tradeoff curve
3. **Learned stopping**: On easy tasks, agent learns to stop early
4. **Adaptive interaction**: On info-seeking tasks, agent learns to gather evidence
5. **No mode collapse**: All three modes used appropriately

## Metrics

Report:
- Task success rate
- Average steps (realized)
- Average tokens generated
- Loop rate (% repeated actions)
- Invalid action rate
- Success-cost Pareto AUC (primary metric)
- Mode distribution by task difficulty

## Baselines

1. Base agent (no RL)
2. Long-CoT (more tokens, fixed horizon)
3. Fixed-horizon TTI (always allow more steps)
4. Curriculum TTI
5. Heuristic stopping (prompt-based)
6. TOS-RL without branching
7. TOS-RL without mode emphasis
8. Full TOS-RL

## Implementation Tips

### Make Mode Tokens Clear
```python
# In LLM output, always start with [THINK], [OBSERVE], or [ANSWER]
# Example:
"[OBSERVE] Let me click on the product link to see more details"
"[THINK] Based on what I see, the price is likely..."
"[ANSWER] The price is $29.99"
```

### Budget Conditioning

Include budget in every prompt:
```python
budget_prompt = f"""
Remaining browser steps: {int(budget.env_interactions)}
Remaining tokens: {int(budget.tokens)}

Choose wisely - interact only when necessary!
"""
```

### Cost Randomization

Different training tasks see different cost coefficients:
```python
λ_env ~ Uniform(0.001, 0.1)
λ_tok ~ Uniform(0.0001, 0.01)
λ_loop ~ Uniform(0.01, 0.5)
λ_bad ~ Uniform(0.1, 1.0)
```

Single policy learns to handle all!

### Prefix Mining Strategies

```python
# Select prefixes from:
- Random positions (25%)
- High-uncertainty decisions (50%)
- Mixed-outcome trajectories (25%)
```

## Debugging

**Problem**: Mode collapse (always OBSERVE)
- **Cause**: Cost coefficients too low
- **Fix**: Increase λ_env, λ_tok in training

**Problem**: Always stops early
- **Cause**: Cost coefficients too high
- **Fix**: Decrease costs or sample wider range

**Problem**: No improvement from branching
- **Cause**: Branches not diverse enough
- **Fix**: Increase branches_per_prefix, mine more prefixes

## Expected Results

On WebArena (typical results):
- **Base**: 45% success, 18 avg steps
- **TTI (fixed horizon)**: 52% success, 22 avg steps
- **TOS-RL**: 53% success, 17 avg steps ← lower cost!

Success-cost Pareto AUC: TOS-RL > TTI > Base

## References

- Proposal: `/Users/luungoc/Project/llm_only_tos_rl_webarena_webvoyager.tex`
- Implementation: `/Users/luungoc/Project/TTI/tti/tos_rl/`
- Example script: `/Users/luungoc/Project/TTI/scripts/train_tos_rl_example.py`
