# Integrating TOS-RL with TTI Training Pipeline

This guide explains how to integrate the TOS-RL framework with your existing TTI codebase.

## Overview

TOS-RL (Think-Observe-Stop RL) is a minimal, LLM-only approach to learning when to think, observe, or answer. It integrates with TTI by:

1. **Collecting trajectories** from TTI's existing agent execution loop
2. **Tracking mode tokens** ([THINK], [OBSERVE], [ANSWER]) in LLM output
3. **Computing cost-aware utilities** based on task success and resource usage
4. **Training with group-relative advantages** (no external critic networks)
5. **Optionally using prefix branching** for improved credit assignment

## Step 1: Modify Trajectory Collection

Your TTI trajectory collection needs to track mode tokens. Update your agent execution loop:

```python
# In your TTI agent's rollout/trajectory collection code

from tti.tos_rl import TokenModeFormatter, BudgetState

def collect_trajectory(task, agent, initial_budget):
    """
    Collect trajectory with mode token tracking for TOS-RL.

    Args:
        task: Task specification
        agent: Your LLM-based agent
        initial_budget: BudgetState with remaining interactions/tokens

    Returns:
        trajectory dict with TOS-RL compatible format
    """

    history = ""
    modes = []
    num_steps = 0
    total_tokens = 0
    num_loops = 0
    num_bad_actions = 0
    success = 0

    budget = BudgetState(
        env_interactions=initial_budget['steps'],
        tokens=initial_budget['tokens'],
        time=initial_budget['time']
    )

    while not done and budget.env_interactions > 0:
        # Add budget info to prompt
        prompt = history + "\n" + create_budget_prompt_suffix(budget)

        # Get LLM output (should start with [THINK], [OBSERVE], or [ANSWER])
        llm_output = agent.generate(prompt)
        total_tokens += count_tokens(llm_output)

        # Parse mode token from LLM output
        mode, content = TokenModeFormatter.parse_mode_and_content(llm_output)

        if mode not in ["THINK", "OBSERVE", "ANSWER"]:
            # Default to OBSERVE if mode not recognized
            mode = "OBSERVE"

        modes.append(mode)

        # Execute mode
        if mode == "THINK":
            # Internal reasoning - no env interaction
            history += f"\n[THINK] {content}"

        elif mode == "OBSERVE":
            # Browser interaction
            action = parse_action(content)
            observation = execute_action(action, env)
            history += f"\n[OBSERVE] {content}\nObservation: {observation}"

            num_steps += 1
            budget.env_interactions -= 1

            # Track loops and bad actions
            if is_repeated_action(action, history):
                num_loops += 1
            if is_invalid_action(action, env):
                num_bad_actions += 1

        elif mode == "ANSWER":
            # Submit final answer
            answer = extract_answer(content)
            history += f"\n[ANSWER] {answer}"
            success = evaluate_answer(answer, task)
            break

    # Build trajectory dict compatible with TOS-RL
    trajectory = {
        "task_id": task['id'],
        "success": success,
        "num_steps": num_steps,
        "num_tokens": total_tokens,
        "num_loops": num_loops,
        "num_bad_actions": num_bad_actions,
        "modes": modes,
        "total_tokens": total_tokens,
    }

    return trajectory
```

## Step 2: Group Trajectories and Compute Utilities

After collecting K trajectories from the same task, compute their utilities:

```python
from tti.tos_rl import TOSRLTrainer, TOSRLConfig

# Configure TOS-RL
config = TOSRLConfig(
    group_size=4,              # K trajectories per task
    max_prefixes_per_task=3,
    branches_per_prefix=3,
    learning_rate=1e-5,
    mode_token_weight=2.0,     # Upweight mode tokens
    use_branching=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

trainer = TOSRLTrainer(config)

# After collecting K=4 trajectories from same task
group_of_4_trajectories = [traj1, traj2, traj3, traj4]

# Compute utilities with randomized cost coefficients
# (trainer samples different λ values to train budget-aware policy)
utilities, costs = trainer.process_trajectories(group_of_4_trajectories)

# Compute group-relative advantages (no external critic!)
advantages = trainer.compute_group_advantages(utilities)

print(f"Utilities: {utilities}")
print(f"Advantages (relative): {advantages}")
# Example: [0.42, -0.45, 0.10, -0.07]
```

## Step 3: Prepare Training Batch

Convert trajectory data to training batch format:

```python
import torch

def prepare_tosrl_batch(trajectories, model, reference_model):
    """
    Prepare batch for TOS-RL training step.

    Args:
        trajectories: List of [K] trajectory dicts
        model: Your LLM policy model
        reference_model: Reference model for KL penalty

    Returns:
        Batch dict ready for trainer.training_step()
    """

    # Get log probabilities from model
    # This would come from your LLM inference
    log_probs = get_log_probs_from_trajectories(trajectories, model)  # [K]
    log_probs_old = log_probs.clone().detach()
    log_probs_ref = get_log_probs_from_trajectories(trajectories, reference_model)  # [K]

    # Get mode-specific log probabilities
    # Log prob of the actual mode token chosen in each trajectory
    mode_log_probs = get_mode_log_probs(trajectories, model)  # [K]

    # Mode probability distribution (for entropy regularization)
    # Probability of each mode (THINK, OBSERVE, ANSWER)
    mode_probs = get_mode_probabilities(trajectories, model)  # [K, 3]

    # Compute utilities and advantages (as shown in Step 2)
    utilities, _ = trainer.process_trajectories(trajectories)
    advantages = trainer.compute_group_advantages(utilities)

    # Create batch
    batch = {
        "log_probs": torch.tensor(log_probs, dtype=torch.float32),
        "log_probs_old": torch.tensor(log_probs_old, dtype=torch.float32),
        "log_probs_ref": torch.tensor(log_probs_ref, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "mode_log_probs": torch.tensor(mode_log_probs, dtype=torch.float32),
        "mode_probs": torch.tensor(mode_probs, dtype=torch.float32),
    }

    return batch
```

## Step 4: Training Loop Integration

Add TOS-RL training to your existing TTI training loop:

```python
def training_loop(num_epochs, tasks_per_epoch, trajectories_per_task=4):
    """
    Main training loop integrating TOS-RL with TTI.
    """

    trainer = TOSRLTrainer(config)

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        epoch_metrics = {
            "total_loss": [],
            "policy_loss": [],
            "mode_loss": [],
            "success_rate": [],
        }

        # Sample tasks for this epoch
        tasks = sample_tasks(tasks_per_epoch)

        for task in tasks:
            # Collect K trajectories from same task
            trajectories = []
            for i in range(trajectories_per_task):
                # Use your existing agent with budget constraints
                traj = collect_trajectory(
                    task=task,
                    agent=policy_model,
                    initial_budget={
                        'steps': 30,
                        'tokens': 4096,
                        'time': 300.0
                    }
                )
                trajectories.append(traj)

            # Prepare batch
            batch = prepare_tosrl_batch(trajectories, policy_model, reference_model)

            # TOS-RL training step
            metrics = trainer.training_step(batch)

            # Track metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])

            # Optional: Prefix-level branching (improved credit assignment)
            if config.use_branching:
                branch_metrics = trainer.branch_training_step(trajectories)
                epoch_metrics['branch_loss'] = branch_metrics.get('branch_loss', 0)

        # Log epoch results
        print(f"Epoch {epoch + 1} results:")
        for key, values in epoch_metrics.items():
            if values:
                avg = sum(values) / len(values)
                print(f"  {key}: {avg:.4f}")

        # Validation on hold-out tasks
        val_tasks = sample_tasks(32, split='val')
        val_metrics = trainer.evaluate(val_tasks)
        print(f"Validation - Success: {val_metrics.get('success_rate', 0):.2%}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(policy_model, f"tos_rl_epoch_{epoch + 1}")
```

## Step 5: Inference Integration

Use TOS-RL inference for deployment:

```python
from tti.tos_rl import TOSRLInference, BudgetState

def deploy_tosrl_agent(task, policy_model, budget_preference='balanced'):
    """
    Deploy trained TOS-RL agent with cost-aware inference.

    Args:
        task: Task to solve
        policy_model: Trained LLM policy
        budget_preference: 'high_efficiency', 'balanced', or 'high_success'
    """

    inference = TOSRLInference(max_steps=30, max_tokens=4096)

    # Set initial budget based on preference
    if budget_preference == 'high_efficiency':
        initial_budget = BudgetState(
            env_interactions=10,  # Strict limit
            tokens=2000,
        )
    elif budget_preference == 'high_success':
        initial_budget = BudgetState(
            env_interactions=50,  # Generous limit
            tokens=8192,
        )
    else:  # balanced
        initial_budget = BudgetState(
            env_interactions=30,
            tokens=4096,
        )

    # Run episode
    result = inference.run_episode(
        task=task['description'],
        llm_fn=policy_model.generate,  # Your LLM generation function
        initial_history=task.get('context', ''),
        initial_budget=initial_budget,
    )

    return {
        'answer': result['final_answer'],
        'success': result['success'],
        'steps_used': result['num_steps'],
        'tokens_used': result['num_tokens'],
        'modes': result['modes'],
    }
```

## Key Integration Points

### 1. Mode Token Formatting

Ensure your LLM outputs start with mode tokens:

```python
# ✓ Correct format
output = "[OBSERVE] I need to click on the search button to search for the product"
output = "[THINK] Based on the price I see, the answer is likely $199.99"
output = "[ANSWER] The product costs $199.99"

# ✗ Wrong format (will default to OBSERVE)
output = "I need to click on the search button"
output = "The answer is $199.99"
```

### 2. Budget Conditioning

Include budget info in prompts:

```python
from tti.tos_rl.utils import create_budget_prompt_suffix

budget = BudgetState(env_interactions=15, tokens=2048)
budget_suffix = create_budget_prompt_suffix(budget)

prompt = f"""
Current state: {history}

{budget_suffix}

What should you do next?
"""
```

### 3. Cost Randomization

The trainer automatically randomizes costs during training:

```python
config = TOSRLConfig(
    # Cost ranges (sampled uniformly during training)
    lambda_env_range=(0.001, 0.1),      # Cost per environment step
    lambda_tok_range=(0.0001, 0.01),    # Cost per token
    lambda_loop_range=(0.01, 0.5),      # Cost for repeated actions
    lambda_bad_range=(0.1, 1.0),        # Cost for invalid actions
)

# Single trained policy learns to handle all cost regimes!
```

## File Structure

After integration, your project structure will be:

```
/Users/luungoc/Project/TTI/
├── tti/
│   ├── tos_rl/              # TOS-RL framework
│   │   ├── __init__.py
│   │   ├── objectives.py
│   │   ├── optimization.py
│   │   ├── branching.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   ├── utils.py
│   │   └── README.md
│   ├── agents/              # Your existing TTI agents
│   ├── environments/        # Environment code
│   └── ...
├── scripts/
│   ├── train_tos_rl_example.py        # Example (for reference)
│   └── train_tosrl_with_tti.py        # Integration script
├── TOSRL_IMPLEMENTATION_GUIDE.md
├── TOSRL_INTEGRATION_WITH_TTI.md      # This file
└── ...
```

## Expected Integration Workflow

```
1. TTI Trajectory Collection
   ↓
2. Mode Token Tracking ([THINK], [OBSERVE], [ANSWER])
   ↓
3. Group K trajectories per task
   ↓
4. Compute cost-aware utilities (randomized costs)
   ↓
5. Compute group-relative advantages (no critic!)
   ↓
6. Optional: Prefix-level branching for credit assignment
   ↓
7. GRPO training step with mode emphasis
   ↓
8. Update LLM policy parameters
   ↓
9. Repeat from step 1 with improved policy
```

## Debugging Integration

**Problem**: Mode tokens not being parsed correctly
- **Check**: Ensure LLM output starts with `[THINK]`, `[OBSERVE]`, or `[ANSWER]`
- **Fix**: Adjust LLM prompting to enforce mode token format

**Problem**: Always choosing one mode (mode collapse)
- **Check**: Are cost coefficients too low/high?
- **Fix**: Adjust `lambda_env`, `lambda_tok` ranges in config

**Problem**: No improvement from training
- **Check**: Are trajectories being grouped correctly (K per task)?
- **Fix**: Ensure K=4+ trajectories from same task are grouped together

**Problem**: Memory issues with prefix branching
- **Check**: Is `max_prefixes_per_task` too high?
- **Fix**: Reduce to 2-3, or disable branching for first training phase

## Next Steps

1. **Update trajectory collection** to track mode tokens
2. **Run training pipeline** on WebArena tasks
3. **Evaluate success-cost frontier** vs baselines (TTI, CVI-SDAR)
4. **Transfer to WebVoyager** for real-world evaluation
5. **Ablation studies**:
   - Without mode emphasis
   - Without prefix branching
   - Different K values
   - Different cost ranges

## References

- **TOS-RL Framework**: `tti/tos_rl/`
- **Implementation Guide**: `TOSRL_IMPLEMENTATION_GUIDE.md`
- **API Reference**: `tti/tos_rl/README.md`
- **Example Script**: `scripts/train_tos_rl_example.py`
- **Original Proposal**: `/Users/luungoc/Project/llm_only_tos_rl_webarena_webvoyager.tex`
