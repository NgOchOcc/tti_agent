# CVI-SDAR: Counterfactual Value-of-Interaction Self-Distilled Agentic RL

A modular implementation of CVI-SDAR, an advanced reinforcement learning framework for training web agents to make optimal decisions about when to think, observe, verify, recover, or answer.

## Quick Reference

### Core Classes

| Module | Class | Purpose |
|--------|-------|---------|
| critics.py | CriticNetwork | MLP for value estimation |
| critics.py | create_critics() | Factory for V, Q, V_ans critics |
| mode_policy.py | ModePolicy | 6-way mode classification |
| mode_policy.py | Mode | Enum of 6 mode types |
| distillation.py | CVIDistillationLoss | Gated self-distillation loss |
| distillation.py | CVIGate | Detached CVI gate |
| prefix_mining.py | PrefixMiner | Intelligent prefix selection |
| prefix_mining.py | BranchGenerator | Forced branch creation |
| training.py | CVISARTrainer | Main training orchestrator |
| training.py | CVISARConfig | Configuration dataclass |
| inference.py | CVISARInferenceController | Runtime decision making |
| inference.py | DeltaSchedule | Stopping threshold strategies |
| utils.py | Budget | Budget state tracking |
| utils.py | compute_cvi_gap() | CVI gap computation |

## Module Overview

### critics.py
Implements value function networks used to estimate:
- `V(h,b)`: Overall expected utility
- `Q(h,b,m)`: Mode-specific expected utility
- `V_ans(h,b)`: Utility of answering immediately

```python
# Create all three critics
critics = create_critics(
    input_dim=768,
    num_modes=6,
    critic_hidden_dims=(256, 128),
)
```

### mode_policy.py
Implements the mode selection policy π_φ(m|h,b).

**6 Modes:**
- 0: THINK - Internal reasoning
- 1: OBSERVE - Gather information
- 2: ACT - Task-progressing action
- 3: VERIFY - Verification/checking
- 4: RECOVER - Backtrack/recovery
- 5: ANSWER - Stop and submit

```python
policy = ModePolicy(
    input_dim=768,
    hidden_dims=(128,),
    num_modes=6,
)

# Get probability distribution
probs = policy(state_embedding)  # [batch, 6]

# Sample modes
modes, log_probs = policy.sample_mode(state_embedding)
```

### distillation.py
Implements CVI-gated self-distillation for dense supervision.

Key insight: Only distill when teacher branch value exceeds answering (positive CVI advantage) and is reliable (low uncertainty).

```python
distill_loss = CVIDistillationLoss(num_modes=6)

output = distill_loss(
    mode_logits=logits,           # [batch, 6]
    teacher_modes=teacher_modes,  # [batch]
    cvi_advantages=advantages,    # [batch]
)

loss = output["total_loss"]
```

### prefix_mining.py
Selects informative prefixes and generates counterfactual branches.

**Sampling strategies:**
- `"random"`: Random prefixes
- `"entropy"`: High-entropy decision points
- `"position"`: Regular intervals through trajectory
- `"mixed"`: Combination of above + special heuristics

```python
miner = PrefixMiner(
    sampling_strategy="mixed",
    max_prefixes_per_trajectory=5,
)

prefixes = miner.mine_prefixes(trajectory)

generator = BranchGenerator(num_modes=6)
branches = generator.generate_branches(prefix, trajectory)
teacher_mode, _ = generator.compute_teacher_mode(branches)
```

### training.py
Main training loop combining all components.

```python
config = CVISARConfig(
    state_embedding_dim=768,
    num_modes=6,
    batch_size=32,
)

trainer = CVISARTrainer(config)

for epoch in range(num_epochs):
    batch = prepare_batch(trajectories)
    prefix_data = prepare_prefix_data(trajectories)
    losses = trainer.train_step(batch, prefix_data)

    trainer.save_checkpoint(f"epoch_{epoch}.pt")
```

### inference.py
Runtime decision making based on learned critics.

**Decision process:**
1. Estimate V_ans and Q for all modes
2. Compute CVI gaps: Δ_m = Q_m - V_ans
3. Select best mode: m* = argmax Δ_m
4. Continue if max gap > threshold, else answer

```python
controller = CVISARInferenceController(
    mode_policy=trainer.mode_policy,
    critics=trainer.critics,
    delta_schedule=DeltaSchedule.budget_linear(0.1, 1.0),
)

decision = controller.get_cvi_decision(state_embedding, budget)

if decision.should_continue:
    mode = decision.mode
else:
    mode = 5  # ANSWER
```

### utils.py
Utilities for budget tracking, CVI computation, and RL helpers.

```python
# Budget management
budget = Budget(
    env_interactions=30,
    tokens=4096,
    verifier_calls=10,
)

# CVI gap computation
cvi_gaps = compute_cvi_gap(q_values, v_ans)  # [batch, num_modes]

# Mode selection by CVI
modes, gaps = select_mode_by_cvi(
    q_values, v_ans,
    delta_threshold=0.0,
)
```

## Key Equations

### Cost-Aware Utility
```
U(τ) = R(τ) - λ_env·N_env - λ_tok·N_tok - λ_ver·N_ver - λ_loop·N_loop
```

### Counterfactual Value-of-Interaction Gap
```
Δ_m(h,b) = Q(h,b,m) - V_ans(h,b)
```

### CVI-Gated Distillation
```
g_t = sg[σ(β(Â_CVI(m*) - η·σ̂_t - ρ))]
L_SD = -E[g_t · log π_φ(m_T* | h,b)]
```

### Optimal Stopping Rule
```
m_t* = argmax_{m ≠ ANSWER} Δ_m(h,b)
continue iff max_m Δ_m(h,b) > δ(b)
```

## Training Pipeline

1. **Trajectory Collection**
   - Run agent with multiple budgets
   - Log observations, actions, modes, rewards
   - Compute cost metrics (steps, tokens, loops, etc.)

2. **Prefix Mining**
   - Select informative decision points
   - Strategies: random, entropy-based, position-based, mixed

3. **Branch Generation**
   - Create forced continuations for each mode
   - Compute branch utilities with cost penalties

4. **Critic Pretraining**
   - Train V, Q, V_ans on collected branches
   - Initialize with meaningful signal

5. **Online CVI-SDAR RL**
   - Collect new trajectories using current policy
   - Generate new branches for distillation
   - Update mode policy with PPO
   - Update critics with MSE loss
   - Apply gated distillation loss

## Loss Functions

### RL Loss (PPO)
```python
L_RL = -E[min(r_t·A_t, clip(r_t,1±ε)·A_t)] - β_H·H(π)
```

### Critic Loss (MSE)
```python
L_critic = E[(V-G)² + (Q_m-G)²]
```

### Distillation Loss (Gated)
```python
L_SD = -E[g_t · log π_φ(m_T*|h,b)]
```

### Total Loss
```python
L_total = L_RL + α_Q·L_critic + α_ans·L_ans + λ_SD·L_SD + λ_KL·KL(π||π_ref)
```

## Configuration

All hyperparameters are in `CVISARConfig`:

```python
config = CVISARConfig(
    # Architecture
    state_embedding_dim=768,
    mode_hidden_dims=(128,),
    critic_hidden_dims=(256, 128),
    num_modes=6,
    dropout=0.1,

    # Learning rates
    lr_mode=1e-4,
    lr_critics=1e-4,

    # Loss weights
    weight_rl=1.0,
    weight_critic=0.5,
    weight_ans=0.5,
    weight_distillation=0.2,
    weight_kl=0.1,
    weight_entropy=0.01,

    # PPO parameters
    ppo_clip_ratio=0.2,
    ppo_epochs=3,
    batch_size=32,
    gamma=0.99,
    gae_lambda=0.95,

    # Distillation
    distillation_beta=2.0,
    distillation_eta=0.5,
    distillation_rho=0.0,

    # Curriculum
    curriculum_steps=1000,
    initial_cost_scale=0.0,
    final_cost_scale=1.0,

    # Prefix mining
    prefix_sampling_strategy="mixed",
    prefixes_per_trajectory=5,
    branches_per_prefix=3,

    device=torch.device("cuda:0"),
)
```

## Typical Metrics

- **Success Rate**: % tasks completed correctly
- **Average Steps**: Mean environment interactions per task
- **Loop Rate**: % steps that repeat previous actions
- **Early Stops**: % tasks where agent stops before max horizon
- **Mode Distribution**: % of each mode used
- **CVI Gaps**: Statistical properties of Δ_m values
- **Gate Activation**: % of distillation that gets gated through

## Common Patterns

### Extract State Embedding
```python
def get_state_embedding(trajectory, step_idx):
    # Your implementation here
    # Should return [embedding_dim] numpy array or tensor
    return embedding
```

### Prepare Training Batch
```python
batch = {
    "state_embeddings": torch.stack([...]),  # [batch, 768]
    "modes": torch.tensor([...]),            # [batch]
    "returns": torch.tensor([...]),          # [batch]
    "old_log_probs": torch.tensor([...]),    # [batch]
    "advantages": torch.tensor([...]),       # [batch]
}
```

### Run Training Step
```python
losses = trainer.train_step(batch, prefix_data)
print(f"PPO: {losses['ppo_loss']:.4f}, "
      f"Critic: {losses['critic_loss']:.4f}, "
      f"Distill: {losses['distill_loss']:.4f}")
```

### Save Checkpoint
```python
trainer.save_checkpoint("checkpoints/epoch_10.pt")
```

### Load Checkpoint
```python
trainer.load_checkpoint("checkpoints/epoch_10.pt")
```

## Performance Tips

1. **Memory Efficient**:
   - Use lightweight critics (hidden_dims=(128,) or (256,))
   - Reduce batch_size if OOM
   - Reduce prefixes_per_trajectory

2. **Faster Training**:
   - Use entropy sampling strategy
   - Reduce ppo_epochs (default 3)
   - Use lower resolution state embeddings

3. **Better Results**:
   - Use "mixed" prefix sampling strategy
   - Increase prefixes_per_trajectory (5-10)
   - Use privileged teacher budget for branches
   - Start with low cost pressure, increase gradually

## Debugging

**Distillation not helping?**
- Check gate activation rate (target: 30-70%)
- Verify CVI advantages are computed correctly
- Reduce distillation weight initially

**Mode collapse?**
- Increase entropy coefficient
- Check mode policy is getting updated
- Verify rewards and advantages are correct

**Critic divergence?**
- Reduce learning rate
- Add gradient clipping
- Check returns computation

## References

- Full proposal: `CVISDAR_INTEGRATION_GUIDE.md`
- Integration guide: `CVISDAR_INTEGRATION_GUIDE.md`
- Example script: `scripts/train_cvi_sdar_example.py`
- Tests: `tests/test_cvi_sdar.py`
