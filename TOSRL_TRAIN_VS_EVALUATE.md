# TOS-RL: Training vs Evaluation Guide

## Quick Overview

**Training**: Collect K trajectories per task → Compute utilities → Update LLM parameters via GRPO loss

**Evaluation**: Run single episodes → Measure success rate, steps, costs → Compare with baselines

---

## 🎓 Training Mode

### Training Loop Structure

```python
from tti.tos_rl import TOSRLTrainer, TOSRLConfig
import torch

# 1. Configure
config = TOSRLConfig(
    group_size=4,              # K trajectories per task
    learning_rate=1e-5,
    mode_token_weight=2.0,
    use_branching=True,
)

# 2. Create trainer
trainer = TOSRLTrainer(config)

# 3. Training loop
for epoch in range(num_epochs):
    for task in tasks_for_epoch:
        # Collect K trajectories from same task
        trajectories = []
        for i in range(4):  # K=4
            traj = collect_trajectory(task, policy_model)
            trajectories.append(traj)

        # Compute utilities with randomized costs
        utilities, costs = trainer.process_trajectories(trajectories)

        # Compute advantages (no critic!)
        advantages = trainer.compute_group_advantages(utilities)

        # Prepare batch
        batch = prepare_batch(trajectories, advantages)

        # Training step
        metrics = trainer.training_step(batch)

        # Optional: prefix branching (advanced)
        if config.use_branching:
            branch_metrics = trainer.branch_training_step(trajectories)

        print(f"Loss: {metrics['total_loss']:.4f}")
```

### Complete Training Example

```python
#!/usr/bin/env python3
"""Training with TOS-RL"""

import torch
import numpy as np
from tti.tos_rl import TOSRLTrainer, TOSRLConfig

def train_tosrl():
    # Configuration
    config = TOSRLConfig(
        group_size=4,
        max_prefixes_per_task=3,
        branches_per_prefix=3,
        learning_rate=1e-5,
        mode_token_weight=2.0,
        use_branching=False,  # Start without branching
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Create trainer
    trainer = TOSRLTrainer(config)

    # Training loop
    num_epochs = 10
    tasks_per_epoch = 32

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        epoch_loss = []
        epoch_success = []

        for task_idx in range(tasks_per_epoch):
            # 1. Collect K trajectories
            trajectories = []
            for k in range(config.group_size):
                traj = collect_trajectory(task_idx)
                trajectories.append(traj)

            # 2. Process trajectories
            utilities, costs = trainer.process_trajectories(trajectories)

            # 3. Compute advantages
            advantages = trainer.compute_group_advantages(utilities)

            # 4. Prepare batch
            batch = {
                "log_probs": torch.randn(len(trajectories)),
                "log_probs_old": torch.randn(len(trajectories)),
                "log_probs_ref": torch.randn(len(trajectories)),
                "advantages": torch.tensor(advantages, dtype=torch.float32),
                "mode_log_probs": torch.randn(len(trajectories)),
                "mode_probs": torch.ones(len(trajectories), 3) / 3,
            }

            # 5. Training step
            metrics = trainer.training_step(batch)

            epoch_loss.append(metrics['total_loss'])
            success_rate = sum(t['success'] for t in trajectories) / len(trajectories)
            epoch_success.append(success_rate)

        # Log epoch statistics
        print(f"Loss: {np.mean(epoch_loss):.4f}")
        print(f"Success: {np.mean(epoch_success):.2%}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                policy_model.state_dict(),
                f"checkpoints/tosrl_epoch_{epoch + 1}.pt"
            )

def collect_trajectory(task_idx):
    """Mock trajectory collection. Replace with actual TTI agent."""
    return {
        "task_id": f"task_{task_idx}",
        "success": np.random.randint(0, 2),
        "num_steps": np.random.randint(3, 20),
        "num_tokens": np.random.randint(100, 500),
        "num_loops": np.random.randint(0, 3),
        "num_bad_actions": np.random.randint(0, 2),
        "modes": ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
    }

if __name__ == "__main__":
    train_tosrl()
```

### Training Hyperparameters to Tune

```python
config = TOSRLConfig(
    # Main parameters
    group_size=4,                   # K trajectories per task
    learning_rate=1e-5,             # LLM learning rate

    # GRPO loss weights
    clip_ratio=0.2,                 # PPO clipping epsilon
    kl_weight=0.1,                  # KL penalty λ
    entropy_weight=0.01,            # Entropy bonus β
    mode_token_weight=2.0,          # Mode emphasis α_m

    # Branching (advanced)
    use_branching=False,            # Start without
    max_prefixes_per_task=3,
    branches_per_prefix=3,

    # Cost randomization (key for multi-regime learning!)
    lambda_env_range=(0.001, 0.1),      # Environment step cost
    lambda_tok_range=(0.0001, 0.01),    # Token generation cost
    lambda_loop_range=(0.01, 0.5),      # Loop/repeat cost
    lambda_bad_range=(0.1, 1.0),        # Invalid action cost
)
```

---

## 📊 Evaluation Mode

### Evaluation Loop Structure

```python
from tti.tos_rl import TOSRLInference, BudgetState

def evaluate_tosrl(policy_model, eval_tasks, cost_preference='balanced'):
    """Run evaluation on held-out test set"""

    # Create inference controller
    inference = TOSRLInference(
        max_steps=30,
        max_tokens=4096,
    )

    results = {
        "success": [],
        "num_steps": [],
        "num_tokens": [],
        "modes": [],
    }

    # Set budget based on cost preference
    if cost_preference == 'high_efficiency':
        budget = BudgetState(env_interactions=10, tokens=2000)
    elif cost_preference == 'high_success':
        budget = BudgetState(env_interactions=50, tokens=8192)
    else:  # balanced
        budget = BudgetState(env_interactions=30, tokens=4096)

    # Run episodes
    for task in eval_tasks:
        result = inference.run_episode(
            task=task['description'],
            llm_fn=policy_model.generate,
            initial_budget=budget,
        )

        results["success"].append(result['success'])
        results["num_steps"].append(result['num_steps'])
        results["num_tokens"].append(result['num_tokens'])
        results["modes"].append(result['modes'])

    # Compute metrics
    metrics = {
        "success_rate": np.mean(results["success"]),
        "mean_steps": np.mean(results["num_steps"]),
        "mean_tokens": np.mean(results["num_tokens"]),
    }

    return metrics, results
```

### Complete Evaluation Example

```python
#!/usr/bin/env python3
"""Evaluation with TOS-RL"""

import numpy as np
from tti.tos_rl import TOSRLInference, BudgetState

def evaluate_tosrl():
    # Load evaluation tasks
    eval_tasks = load_eval_tasks()  # Your WebArena validation set

    # Load trained model
    policy_model = load_trained_model("checkpoints/tosrl_epoch_10.pt")

    # Test with different cost preferences
    cost_preferences = ['high_efficiency', 'balanced', 'high_success']

    all_results = {}

    for preference in cost_preferences:
        print(f"\nEvaluation: Cost preference = {preference}")
        print("=" * 60)

        # Run evaluation
        metrics, detailed_results = evaluate_tosrl(
            policy_model,
            eval_tasks,
            cost_preference=preference
        )

        all_results[preference] = {
            "metrics": metrics,
            "details": detailed_results,
        }

        # Print results
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Avg Steps: {metrics['mean_steps']:.1f}")
        print(f"Avg Tokens: {metrics['mean_tokens']:.0f}")

        # Analyze mode distribution
        all_modes = []
        for mode_list in detailed_results["modes"]:
            all_modes.extend(mode_list)

        mode_counts = {m: all_modes.count(m) for m in ["THINK", "OBSERVE", "ANSWER"]}
        total = sum(mode_counts.values())

        print("\nMode Distribution:")
        for mode, count in mode_counts.items():
            pct = 100 * count / total
            print(f"  {mode}: {pct:.1f}% ({count}/{total})")

    # Compare with baselines
    print("\n" + "=" * 60)
    print("Comparison with Baselines:")
    print("=" * 60)

    baselines = {
        "Base Agent": {"success": 0.45, "steps": 18},
        "TTI (fixed)": {"success": 0.52, "steps": 22},
        "CVI-SDAR": {"success": 0.53, "steps": 18},
    }

    for name, baseline in baselines.items():
        print(f"{name}:")
        print(f"  Success: {baseline['success']:.2%}")
        print(f"  Avg Steps: {baseline['steps']:.1f}")

    print(f"\nTOS-RL (balanced):")
    tosrl_metrics = all_results['balanced']['metrics']
    print(f"  Success: {tosrl_metrics['success_rate']:.2%}")
    print(f"  Avg Steps: {tosrl_metrics['mean_steps']:.1f}")

    return all_results

def load_eval_tasks():
    """Load WebArena validation tasks. Replace with your code."""
    return [
        {"id": "task_1", "description": "Find the price of product X"},
        {"id": "task_2", "description": "Book a flight"},
        # ... more tasks
    ]

def load_trained_model(checkpoint_path):
    """Load trained policy model. Replace with your code."""
    # return your_llm_model.load_from_checkpoint(checkpoint_path)
    pass

if __name__ == "__main__":
    results = evaluate_tosrl()
```

---

## 🎯 Key Differences

| Aspect | Training | Evaluation |
|--------|----------|-----------|
| **Goal** | Update LLM parameters | Measure performance |
| **Trajectories** | K=4 per task (group) | 1 per task |
| **Cost Coefficients** | Randomized | Fixed (by preference) |
| **Optimization** | GRPO loss step | No optimization |
| **Advantages** | Computed from group | Not needed |
| **Batch Preparation** | Full batch (log probs, etc) | Just inference |
| **Loss Computation** | Yes | No |
| **Gradient Updates** | Yes | No |
| **Budget** | Varies by cost coeff | Fixed per preference |

---

## 🔀 Mixed Training & Evaluation Workflow

Most commonly, you'll do **training + periodic evaluation**:

```python
def train_with_evaluation():
    trainer = TOSRLTrainer(config)
    inference = TOSRLInference(max_steps=30, max_tokens=4096)

    for epoch in range(num_epochs):
        # === TRAINING ===
        print(f"Training Epoch {epoch + 1}")

        for task in training_tasks:
            # Collect K trajectories
            trajectories = [collect_trajectory(task) for _ in range(4)]

            # Training step
            utilities, _ = trainer.process_trajectories(trajectories)
            advantages = trainer.compute_group_advantages(utilities)
            batch = prepare_batch(trajectories, advantages)
            metrics = trainer.training_step(batch)

        # === PERIODIC EVALUATION ===
        if (epoch + 1) % eval_interval == 0:
            print(f"\nEvaluation after Epoch {epoch + 1}")

            success_rates = []
            for task in eval_tasks:
                result = inference.run_episode(
                    task=task['description'],
                    llm_fn=policy_model.generate,
                    initial_budget=BudgetState(env_interactions=30, tokens=4096)
                )
                success_rates.append(result['success'])

            eval_success = np.mean(success_rates)
            print(f"Validation Success: {eval_success:.2%}")

            # Save best checkpoint
            if eval_success > best_success:
                save_checkpoint(policy_model, f"best_model.pt")
                best_success = eval_success
```

---

## 📈 Metrics to Track

### Training Metrics
```python
metrics = trainer.training_step(batch)

print(f"Total Loss: {metrics['total_loss']:.4f}")
print(f"Policy Loss: {metrics['policy_loss']:.4f}")
print(f"Mode Loss: {metrics['mode_loss']:.4f}")
print(f"KL Divergence: {metrics['kl_divergence']:.4f}")
print(f"Entropy: {metrics['entropy']:.4f}")
```

### Evaluation Metrics
```python
metrics, results = evaluate_tosrl(model, tasks, preference='balanced')

print(f"Success Rate: {metrics['success_rate']:.2%}")
print(f"Mean Steps: {metrics['mean_steps']:.1f}")
print(f"Mean Tokens: {metrics['mean_tokens']:.0f}")

# Mode distribution
mode_dist = analyze_modes(results['modes'])
print(f"Mode Distribution: {mode_dist}")
```

---

## 🚀 Running in Practice

### Run Training Only
```bash
python train_tosrl.py --mode train --epochs 10 --tasks-per-epoch 32
```

### Run Evaluation Only
```bash
python eval_tosrl.py --mode eval --checkpoint best_model.pt --tasks 100
```

### Run Train + Eval
```bash
python train_tosrl.py --mode train_eval --epochs 10 --eval-interval 2
```

### Run Pareto Analysis (Multiple Cost Preferences)
```bash
python eval_tosrl.py --mode pareto --checkpoint best_model.pt --preferences high_efficiency,balanced,high_success
```

---

## 💾 Checkpointing

### Saving During Training
```python
# Save every N epochs
if (epoch + 1) % save_interval == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state': policy_model.state_dict(),
        'config': config,
        'metrics': training_metrics,
    }
    torch.save(checkpoint, f"checkpoints/tosrl_epoch_{epoch + 1}.pt")

# Save best model
if eval_success > best_success:
    torch.save(policy_model.state_dict(), "checkpoints/best_model.pt")
    best_success = eval_success
```

### Loading for Evaluation
```python
checkpoint = torch.load("checkpoints/best_model.pt")
policy_model.load_state_dict(checkpoint)
policy_model.eval()  # Set to evaluation mode

# Run evaluation
metrics, results = evaluate_tosrl(policy_model, eval_tasks)
```

---

## 🎯 Quick Comparison Table

```
Operation               Training          Evaluation
─────────────────────────────────────────────────────
K trajectories          4 per task        1 per task
Cost coefficients       Random            Fixed
Batch preparation       Full              Minimal
Loss computation        Yes               No
Parameter updates       Yes               No
Advantage computation   Yes               No
Epochs/iterations       Multiple          Single pass
Goal                    Optimize LLM      Measure perf
Time per task           Longer            Shorter
```

---

## 📋 Common Patterns

### Training Loop Pattern
```python
for epoch in range(epochs):
    for batch_idx, tasks in enumerate(training_batches):
        # Collect data
        trajectories = collect_trajectories(tasks)

        # Process
        utilities = trainer.process_trajectories(trajectories)
        advantages = trainer.compute_group_advantages(utilities)

        # Train
        metrics = trainer.training_step(prepare_batch(advantages))

        # Log
        log_metrics(metrics)
```

### Evaluation Pattern
```python
policy_model.eval()
with torch.no_grad():
    all_results = []
    for task in eval_tasks:
        result = inference.run_episode(task, policy_model.generate)
        all_results.append(result)

    metrics = aggregate_results(all_results)
    report_metrics(metrics)
```

---

## 🎓 When to Train vs Evaluate

### Train When:
- Starting from scratch or pre-trained model
- Want to improve performance
- Hyperparameter tuning
- Curriculum learning (progressive difficulty)

### Evaluate When:
- Want to measure final performance
- Comparing with baselines
- Testing on unseen tasks
- Analyzing learned behavior
- Generating results for paper

### Do Both When:
- Standard research practice
- Want to track learning curves
- Early stopping based on validation
- Hyperparameter selection
- Final benchmarking

---

## ⏱️ Typical Workflow Timeline

```
Week 1-2: Training
  ├─ Epoch 1-3: Initial training
  ├─ Eval every epoch
  └─ Save checkpoints

Week 2-3: Fine-tuning
  ├─ Adjust hyperparameters
  ├─ Enable prefix branching
  └─ Continue training

Week 3: Final Evaluation
  ├─ Load best checkpoint
  ├─ Run full evaluation
  ├─ Pareto analysis
  └─ Generate results

Week 4: Analysis
  ├─ Compare with baselines
  ├─ Mode analysis
  ├─ Transfer to WebVoyager
  └─ Write paper
```

---

## 🔍 Debugging

### Training Issues
- Loss not decreasing → Check learning rate, batch size
- NaN losses → Check advantage normalization
- Mode collapse → Check cost coefficient ranges

### Evaluation Issues
- Low success rate → Model may not have trained
- High step usage → Costs too low during training
- Uneven mode distribution → Check mode emphasis weight

---

**Next**: Follow TOSRL_INTEGRATION_CHECKLIST.md for detailed steps on integrating with your TTI codebase!
