# TOS-RL: LLM-Only Think-Observe-Stop Reinforcement Learning

A **minimal, LLM-only framework** for training agents to decide when to think, observe, or answer.

## Quick Summary

**The Idea**: Make `[THINK]`, `[OBSERVE]`, and `[ANSWER]` regular LLM output tokens, not separate modules. Train them end-to-end with cost-aware RL.

**The Advantage**: No external gate networks, no separate critic. Just the LLM learning to emit the right mode token at the right time.

## Core Components

### 1. objectives.py
**Cost-aware utility computation**
```
U(τ) = R_task - λ_env·N_env - λ_tok·N_tok - λ_loop·N_loop - λ_bad·N_bad
```
- CostAwareUtility: Container for trajectory costs
- compute_cost_aware_utility(): Single trajectory utility
- batch_compute_utilities(): Batch processing

### 2. optimization.py
**GRPO (Group Relative Policy Optimization)**
- No separate critic - advantages from relative ranking within groups
- GroupRelativePolicyOptimization: Advantage computation
- GRPOLoss: GRPO loss function
- Mode-token emphasis for faster learning
- KL penalty and entropy regularization

### 3. branching.py
**Prefix-level counterfactual branches**
- PrefixBrancher: Select informative prefixes from trajectories
- BranchCollector: Manage and evaluate branches
- Improved credit assignment without external critic

### 4. training.py
**Main training loop**
- TOSRLConfig: Configuration dataclass
- TOSRLTrainer: Training orchestrator
- Integrates GRPO, branching, and cost-aware objectives
- Budget-conditioned training with randomized costs

### 5. inference.py
**Inference algorithm**
- TOSRLInference: Runtime agent controller
- TokenMode: Enum for mode tokens
- Budget tracking and mode parsing
- InferenceStatistics: Episode tracking

### 6. utils.py
**Utilities**
- BudgetState: Remaining budget management
- CostMetrics: Cost tracking
- TokenModeFormatter: Mode token parsing/formatting
- CostCoefficientSchedule: Randomized costs for training

## Usage Example

### Training

```python
from tti.tos_rl import TOSRLTrainer, TOSRLConfig
from tti.tos_rl import batch_compute_utilities, normalize_utilities

# Configure
config = TOSRLConfig(
    group_size=4,              # K trajectories per task
    max_prefixes_per_task=3,
    branches_per_prefix=3,
    mode_token_weight=2.0,     # Upweight mode tokens
)

trainer = TOSRLTrainer(config)

# Collect trajectories with modes
trajectories = [
    {
        "success": 1,
        "num_steps": 5,
        "num_tokens": 256,
        "num_loops": 0,
        "num_bad_actions": 0,
        "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
    },
    # ... more trajectories
]

# Group into K-tuples (same task)
# Compute utilities with randomized costs
utilities, costs = trainer.process_trajectories(group_of_4)

# Compute group advantages (no critic!)
advantages = trainer.compute_group_advantages(utilities)

# Prepare batch
batch = {
    "log_probs": log_probs,
    "log_probs_old": log_probs_old,
    "advantages": advantages,
    "mode_log_probs": mode_log_probs,
}

# Training step
metrics = trainer.training_step(batch)
```

### Inference

```python
from tti.tos_rl import TOSRLInference, BudgetState

inference = TOSRLInference(max_steps=30, max_tokens=4096)

# Mock LLM
def llm_model(prompt):
    # Returns string starting with [THINK], [OBSERVE], or [ANSWER]
    return "[OBSERVE] Click on the product link"

# Run episode
result = inference.run_episode(
    task="Find the price of the laptop",
    llm_fn=llm_model,
    initial_budget=BudgetState(env_interactions=30, tokens=4096),
)

print(f"Success: {result['success']}")
print(f"Modes used: {result['modes']}")
print(f"Steps: {result['num_steps']}")
```

## Key Features

### ✅ LLM-Only
- No external gate network
- No separate critic network
- Only the LLM parameters are trained
- Mode tokens are regular LLM output

### ✅ Group-Relative Advantages
```
Â_i = (U_i - mean(U_1:K)) / std(U_1:K)
```
- Sample K trajectories from same task
- Rank them by utility
- No value network needed!

### ✅ Mode Emphasis
Upweight mode-token learning:
```
L_mode = -α_m · Σ log π(m) * Â
```
Compute decisions learn faster

### ✅ Prefix Branching
Generate counterfactual continuations:
- Force [THINK] start → compute utility
- Force [OBSERVE] start → compute utility
- Force [ANSWER] start → compute utility
- Use branch-relative advantages for training
- Improves credit assignment without critic

### ✅ Budget Conditioning
Train with randomized cost coefficients:
```
λ_env ~ Uniform(0.001, 0.1)
λ_tok ~ Uniform(0.0001, 0.01)
```
Single policy learns multiple cost regimes!

### ✅ Cost-Aware Utility
```
U(τ) = R_task - λ_env·N_env - λ_tok·N_tok - λ_loop·N_loop - λ_bad·N_bad
```
Balances success vs efficiency

## API Reference

### CostAwareUtility
```python
utility_obj = CostAwareUtility(
    task_reward=1.0,
    num_env_steps=5,
    num_tokens=256,
    num_loops=0,
    num_bad_actions=0,
)

utility = utility_obj.compute(
    lambda_env=0.01,
    lambda_tok=0.001,
)
```

### GroupRelativePolicyOptimization
```python
advantages = GroupRelativePolicyOptimization.compute_group_advantages(
    utilities=np.array([0.5, 0.3, 0.4, 0.2])  # [K]
)
# Returns: [0.42, -0.45, 0.10, -0.07]  # Normalized
```

### GRPOLoss
```python
loss_fn = GRPOLoss(config)

total_loss, metrics = loss_fn.compute_total_loss(
    log_probs=log_probs_new,
    log_probs_old=log_probs_old,
    log_probs_ref=log_probs_ref,
    advantages=advantages,
    mode_log_probs=mode_log_probs,
    mode_probs=mode_probs,
)
```

### PrefixBrancher
```python
brancher = PrefixBrancher(max_prefixes_per_trajectory=3)

prefixes = brancher.select_prefixes(
    trajectory={
        "num_steps": 10,
        "success": 1,
        "modes": ["THINK", "OBSERVE", ...],
    }
)
```

### BranchCollector
```python
collector = BranchCollector()

collector.add_branch(
    prefix=prefix,
    mode="OBSERVE",
    tokens=[...],
    utility=0.45,
)

advantages = collector.compute_branch_advantages(prefix)
best_mode = collector.get_mode_target(prefix)
```

### TOSRLInference
```python
inference = TOSRLInference(max_steps=30)

result = inference.run_episode(
    task="Find the price",
    llm_fn=llm_model,
    initial_budget=BudgetState(),
)
```

### TokenModeFormatter
```python
# Parsing
mode, content = TokenModeFormatter.parse_mode_and_content(
    "[OBSERVE] Click on the search button"
)
# Returns: ("OBSERVE", "Click on the search button")

# Formatting
output = TokenModeFormatter.format_with_mode("THINK", "reasoning text")
# Returns: "[THINK] reasoning text"
```

## Training Hyperparameters

Key config options:
```python
config = TOSRLConfig(
    group_size=4,              # K trajectories per task
    learning_rate=1e-5,
    max_grad_norm=1.0,

    # GRPO
    clip_ratio=0.2,            # PPO clipping
    kl_weight=0.1,             # KL penalty
    entropy_weight=0.01,       # Entropy bonus
    mode_token_weight=2.0,     # Mode emphasis factor

    # Branching
    max_prefixes_per_task=3,   # Prefixes to branch from
    branches_per_prefix=3,     # [THINK], [OBSERVE], [ANSWER]
    use_branching=True,

    # Cost ranges (randomized during training)
    lambda_env_range=(0.001, 0.1),
    lambda_tok_range=(0.0001, 0.01),
    lambda_loop_range=(0.01, 0.5),
    lambda_bad_range=(0.1, 1.0),
)
```

## Key Equations

### Group Relative Advantage
```
Â_i = (U_i - mean(U_1:K)) / std(U_1:K)
```

### GRPO Objective
```
L = -1/N Σ_i min(ρ_i * Â_i, clip(ρ_i, 1±ε) * Â_i)
    + λ_KL * KL(π || π_ref)
    - β_H * H(π)
```

### Mode Emphasis
```
L_mode = -α_m * Σ_i log π(m_i) * Â_i
```

### Cost-Aware Utility
```
U(τ) = R_task - Σ λ_c * N_c
     = R_task - λ_env·N_env - λ_tok·N_tok
       - λ_loop·N_loop - λ_bad·N_bad
```

## Performance Tips

1. **Start with no branching** - simpler training first
2. **Upweight mode tokens** - they're the critical decision
3. **Randomize costs widely** - teaches adaptability
4. **Large group size** - better advantage estimates (K≥4)
5. **Monitor mode distribution** - all three should be used

## Debugging

| Issue | Cause | Fix |
|-------|-------|-----|
| Always OBSERVE | Costs too low | Increase λ_env, λ_tok |
| Always ANSWER | Costs too high | Decrease λ_env, λ_tok |
| Mode not learning | Mode weight too low | Increase mode_token_weight |
| No improvement | Branching issues | Increase branches_per_prefix |

## Comparison with Alternatives

| Feature | CVI-SDAR | TOS-RL |
|---------|----------|--------|
| Gate network | Yes | No |
| Critic networks | 3 (V,Q,V_ans) | 0 (group-relative) |
| Mode tokens | Separate head | LLM output |
| Complexity | Higher | Minimal |
| Interpretability | Critic values | Token probabilities |

## Files and Structure

```
tti/tos_rl/
├── __init__.py              # Exports
├── objectives.py            # Cost-aware utility (200 lines)
├── optimization.py          # GRPO loss (300 lines)
├── branching.py            # Prefix branching (300 lines)
├── training.py             # Main trainer (250 lines)
├── inference.py            # Inference (300 lines)
├── utils.py                # Utilities (250 lines)
└── README.md               # This file
```

## Example Scripts

- `scripts/train_tos_rl_example.py`: Complete training pipeline

## References

- Proposal: `llm_only_tos_rl_webarena_webvoyager.tex`
- Guide: `TOSRL_IMPLEMENTATION_GUIDE.md`
