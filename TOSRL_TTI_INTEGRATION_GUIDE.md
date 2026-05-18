# TOS-RL + TTI Agent Integration Guide

**Purpose**: How to integrate real trajectory collection into your TTI agent
**Status**: Complete integration guide for data collection
**Date**: 2026-05-18

---

## Overview

This guide shows how to modify your existing TTI agent to collect trajectories in the format required by TOS-RL real data training.

### What We're Collecting
```
trajectory = {
    "task_id": "webarena_task_001",
    "success": 1,                              # Did agent succeed?
    "num_steps": 5,                            # OBSERVE action count
    "num_tokens": 250,                         # Total tokens used
    "num_loops": 0,                            # Repeated actions
    "num_bad_actions": 0,                      # Invalid actions
    "modes": ["THINK", "OBSERVE", "ANSWER"]   # Mode sequence
}
```

---

## Step 1: Add Trajectory Collection to TTI Agent

### Minimal Integration (15 lines)

Add these imports and variables to your TTI agent class:

```python
import json
from pathlib import Path

class TTIAgent:
    def __init__(self, trajectory_file: str = None):
        # ... existing init code ...

        # Add trajectory collection
        self.trajectory_file = trajectory_file
        if self.trajectory_file:
            Path(self.trajectory_file).parent.mkdir(parents=True, exist_ok=True)

        # Tracking variables (reset each episode)
        self.modes_used = []
        self.num_steps = 0
        self.num_tokens = 0
        self.num_repeated_actions = 0
        self.num_invalid_actions = 0
```

### Full Integration Example

```python
import json
from pathlib import Path
from typing import Dict, Any, Optional

class TTIAgentWithTrajectoryTracking:
    """TTI agent with real-time trajectory collection."""

    def __init__(
        self,
        model_name: str,
        trajectory_file: Optional[str] = None,
        **kwargs
    ):
        # ... existing initialization ...
        self.model = load_model(model_name)
        self.tokenizer = load_tokenizer(model_name)

        # Trajectory collection setup
        self.trajectory_file = trajectory_file
        if self.trajectory_file:
            Path(self.trajectory_file).parent.mkdir(parents=True, exist_ok=True)
            print(f"Trajectories will be saved to: {self.trajectory_file}")

    def _reset_trajectory_tracking(self) -> None:
        """Reset tracking variables for new episode."""
        self.modes_used = []
        self.num_steps = 0
        self.num_tokens = 0
        self.num_repeated_actions = 0
        self.num_invalid_actions = 0

    def _record_mode(self, mode: str) -> None:
        """Record a mode token."""
        self.modes_used.append(mode)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in generated text."""
        return len(self.tokenizer.encode(text))

    def run_episode(self, task: Dict[str, Any]) -> bool:
        """
        Run single episode and optionally save trajectory.

        Args:
            task: Task dict with 'id' and 'description'

        Returns:
            success: Whether task succeeded
        """
        # Reset tracking
        self._reset_trajectory_tracking()
        task_id = task["id"]

        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"Description: {task.get('description', 'N/A')}")
        print(f"{'='*60}")

        try:
            # Your agent's main loop
            success = False
            for step_num in range(self.max_steps):
                # 1. THINK: Generate reasoning
                self._record_mode("THINK")
                reasoning = self.generate_reasoning(task)
                self.num_tokens += self._count_tokens(reasoning)

                # 2. OBSERVE: Interact with environment
                self._record_mode("OBSERVE")
                observation, is_repeated = self.take_action(reasoning, task)
                self.num_steps += 1
                self.num_tokens += self._count_tokens(observation)

                if is_repeated:
                    self.num_repeated_actions += 1

                # Check for success condition
                if self.check_success(observation, task):
                    success = True
                    break

                # Check for max steps
                if step_num >= self.max_steps - 1:
                    break

            # 3. ANSWER: Generate final answer
            self._record_mode("ANSWER")
            final_answer = self.generate_answer(task)
            self.num_tokens += self._count_tokens(final_answer)

            # Verify success
            if not success:
                success = self.verify_answer(final_answer, task)

            # Save trajectory if collecting
            if self.trajectory_file:
                self._save_trajectory(task_id, success)

            return success

        except Exception as e:
            print(f"Error in episode: {e}")
            if self.trajectory_file:
                self._save_trajectory(task_id, False)
            return False

    def _save_trajectory(self, task_id: str, success: bool) -> None:
        """Save trajectory to JSONL file."""
        trajectory = {
            "task_id": task_id,
            "success": int(success),
            "num_steps": self.num_steps,
            "num_tokens": self.num_tokens,
            "num_loops": self.num_repeated_actions,
            "num_bad_actions": self.num_invalid_actions,
            "modes": self.modes_used,
        }

        with open(self.trajectory_file, "a") as f:
            f.write(json.dumps(trajectory) + "\n")

        print(f"  Trajectory saved: success={success}, steps={self.num_steps}, "
              f"tokens={self.num_tokens}, modes={len(self.modes_used)}")

    def run_batch(self, tasks: list, save_trajectories: bool = True) -> Dict[str, Any]:
        """
        Run multiple tasks and collect trajectories.

        Args:
            tasks: List of task dicts
            save_trajectories: Whether to save trajectories

        Returns:
            results: Summary statistics
        """
        if save_trajectories and not self.trajectory_file:
            raise ValueError("trajectory_file must be set to save trajectories")

        results = {
            "total_tasks": len(tasks),
            "successful": 0,
            "failed": 0,
            "avg_steps": 0,
            "avg_tokens": 0,
            "trajectories_file": self.trajectory_file if save_trajectories else None,
        }

        all_steps = []
        all_tokens = []

        for task in tasks:
            success = self.run_episode(task)
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1

            all_steps.append(self.num_steps)
            all_tokens.append(self.num_tokens)

        if all_steps:
            results["avg_steps"] = sum(all_steps) / len(all_steps)
            results["avg_tokens"] = sum(all_tokens) / len(all_tokens)

        return results
```

---

## Step 2: Track Specific Metrics

### Option A: Simple Counting

```python
class TTIAgent:
    def generate_reasoning(self, task):
        """Generate reasoning with token counting."""
        prompt = f"Task: {task['description']}\n\nThinking: "

        # Generate with your model
        reasoning = self.model.generate(prompt, max_tokens=500)

        # Track tokens
        self.num_tokens += len(self.tokenizer.encode(reasoning))
        self._record_mode("THINK")

        return reasoning

    def take_action(self, reasoning, task):
        """Take environment action and track."""
        action = self.parse_action(reasoning)
        observation = self.env.step(action)

        self.num_steps += 1
        self.num_tokens += len(self.tokenizer.encode(observation))
        self._record_mode("OBSERVE")

        # Detect repeated action
        if action == self.last_action:
            self.num_repeated_actions += 1

        self.last_action = action
        return observation, action == self.last_action
```

### Option B: Batch Token Counting (More Efficient)

```python
def run_episode(self, task):
    """Run episode with batch token counting."""
    self._reset_trajectory_tracking()

    # Collect all generated text
    all_text = []

    for step in range(self.max_steps):
        reasoning = self.generate_reasoning(task)
        all_text.append(reasoning)
        self._record_mode("THINK")

        observation = self.take_action(reasoning, task)
        all_text.append(observation)
        self._record_mode("OBSERVE")

        if self.check_success(observation, task):
            break

    final_answer = self.generate_answer(task)
    all_text.append(final_answer)
    self._record_mode("ANSWER")

    # Count all tokens at once (more efficient)
    all_text_concat = "".join(all_text)
    self.num_tokens = len(self.tokenizer.encode(all_text_concat))

    success = self.verify_answer(final_answer, task)
    if self.trajectory_file:
        self._save_trajectory(task["id"], success)

    return success
```

---

## Step 3: Collect Trajectories from WebArena/WebVoyager

### Integration with WebArena Agent

```python
from tti_webarena import WebArenaAgent
import json

class WebArenaWithTOSRL(WebArenaAgent):
    """WebArena agent with TOS-RL trajectory collection."""

    def __init__(self, trajectory_file: str = None, **kwargs):
        super().__init__(**kwargs)
        self.trajectory_file = trajectory_file
        if trajectory_file:
            import pathlib
            pathlib.Path(trajectory_file).parent.mkdir(parents=True, exist_ok=True)

    def run_task(self, task, save_trajectory=True):
        """Run task and save trajectory."""
        self._reset_tracking()
        task_id = task["task_id"]

        # Run agent
        success, steps, tokens = self.execute_task(task)

        # Save trajectory if requested
        if save_trajectory and self.trajectory_file:
            trajectory = {
                "task_id": task_id,
                "success": int(success),
                "num_steps": steps,
                "num_tokens": tokens,
                "num_loops": self.repeated_actions,
                "num_bad_actions": self.invalid_actions,
                "modes": self.modes,
            }
            with open(self.trajectory_file, "a") as f:
                f.write(json.dumps(trajectory) + "\n")

        return success

    def collect_trajectories(self, task_list, output_file):
        """Collect trajectories from multiple tasks."""
        results = {
            "total": len(task_list),
            "successful": 0,
            "failed": 0,
        }

        for task in task_list:
            success = self.run_task(task, save_trajectory=True)
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1

            print(f"Task {task['task_id']}: {'✓' if success else '✗'}")

        print(f"\nCollection summary:")
        print(f"  Trajectories: {results['total']}")
        print(f"  Success: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Saved to: {output_file}")

        return results
```

### Example Usage

```python
from tti_webarena import load_tasks
from tti_agent import WebArenaWithTOSRL

# Initialize agent with trajectory collection
agent = WebArenaWithTOSRL(
    model_id="claude-3-opus",
    trajectory_file="/Users/luungoc/Project/TTI/data/webarena_trajectories.jsonl"
)

# Load WebArena tasks
tasks = load_tasks("webarena", num_tasks=100)

# Collect trajectories
results = agent.collect_trajectories(tasks)

# Now ready for training!
print(f"\n✓ Trajectories ready for training")
print(f"Run: ./run_train_real_data.sh --trajectory-file {agent.trajectory_file}")
```

---

## Step 4: Verify Collected Data

### Check Format

```bash
# Check file exists
ls -lh data/trajectories.jsonl

# Count trajectories
wc -l data/trajectories.jsonl

# View first 3 trajectories
head -3 data/trajectories.jsonl | python3 -m json.tool
```

### Validate Format

```python
# validate_trajectories.py
import json
import sys

def validate_trajectories(filepath):
    """Validate trajectory file format."""
    required_fields = {'task_id', 'success', 'num_steps', 'num_tokens', 'modes'}

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                traj = json.loads(line)

                # Check required fields
                if not all(field in traj for field in required_fields):
                    missing = required_fields - set(traj.keys())
                    print(f"❌ Line {line_num}: Missing {missing}")
                    return False

                # Check types
                if not isinstance(traj['success'], int) or traj['success'] not in [0, 1]:
                    print(f"❌ Line {line_num}: Invalid success value")
                    return False

                if not isinstance(traj['modes'], list):
                    print(f"❌ Line {line_num}: modes must be list")
                    return False

            except json.JSONDecodeError as e:
                print(f"❌ Line {line_num}: Invalid JSON - {e}")
                return False

    print(f"✓ All trajectories valid")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_trajectories.py <file>")
        sys.exit(1)

    if validate_trajectories(sys.argv[1]):
        sys.exit(0)
    else:
        sys.exit(1)
```

Run validation:
```bash
python3 validate_trajectories.py data/trajectories.jsonl
```

---

## Step 5: Analyze Collection Results

```python
# analyze_collection.py
import json
from collections import defaultdict

def analyze_trajectories(filepath):
    """Analyze collected trajectories."""

    trajectories = []
    task_groups = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            traj = json.loads(line)
            trajectories.append(traj)
            task_groups[traj['task_id']].append(traj)

    # Statistics
    success_count = sum(1 for t in trajectories if t['success'])
    total_tokens = sum(t['num_tokens'] for t in trajectories)
    total_steps = sum(t['num_steps'] for t in trajectories)

    print("COLLECTION ANALYSIS")
    print("=" * 60)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Unique tasks: {len(task_groups)}")
    print(f"Trajectories per task: {len(trajectories) / len(task_groups):.1f}")
    print()
    print(f"Success rate: {100 * success_count / len(trajectories):.1f}%")
    print(f"Avg steps: {total_steps / len(trajectories):.1f}")
    print(f"Avg tokens: {total_tokens / len(trajectories):.0f}")
    print()

    # Mode analysis
    all_modes = []
    for t in trajectories:
        all_modes.extend(t['modes'])

    print("Mode distribution:")
    print(f"  THINK: {100 * all_modes.count('THINK') / len(all_modes):.1f}%")
    print(f"  OBSERVE: {100 * all_modes.count('OBSERVE') / len(all_modes):.1f}%")
    print(f"  ANSWER: {100 * all_modes.count('ANSWER') / len(all_modes):.1f}%")
    print()

    # Task grouping for training
    valid_groups = sum(1 for tasks in task_groups.values() if len(tasks) >= 4)
    print(f"Valid training groups (4+ trajectories): {valid_groups}")
    print(f"Training trajectories available: {valid_groups * 4}")

    return {
        "total": len(trajectories),
        "success_rate": success_count / len(trajectories),
        "avg_steps": total_steps / len(trajectories),
        "avg_tokens": total_tokens / len(trajectories),
        "valid_groups": valid_groups,
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_collection.py <file>")
        sys.exit(1)

    stats = analyze_trajectories(sys.argv[1])
```

---

## Complete Example: End-to-End Integration

```python
#!/usr/bin/env python3
"""
Complete example: TTI agent → collect trajectories → train TOS-RL → evaluate.
"""

import subprocess
from pathlib import Path

# 1. Initialize agent with trajectory collection
print("Step 1: Initialize agent with trajectory collection")
agent = WebArenaWithTOSRL(
    model_id="claude-3-opus",
    trajectory_file="./data/webarena_real_trajectories.jsonl"
)

# 2. Collect trajectories
print("\nStep 2: Collect trajectories from 100 WebArena tasks")
tasks = load_webarena_tasks(num_tasks=100)
results = agent.collect_trajectories(tasks)

# 3. Validate trajectories
print("\nStep 3: Validate trajectory format")
subprocess.run([
    "python3", "validate_trajectories.py",
    "./data/webarena_real_trajectories.jsonl"
])

# 4. Analyze collection
print("\nStep 4: Analyze collection")
subprocess.run([
    "python3", "analyze_collection.py",
    "./data/webarena_real_trajectories.jsonl"
])

# 5. Train TOS-RL on real data
print("\nStep 5: Train TOS-RL on real data")
subprocess.run([
    "./run_train_real_data.sh",
    "--trajectory-file", "./data/webarena_real_trajectories.jsonl",
    "--epochs", "20",
    "--experiment", "tti_agent_real_data"
])

# 6. Evaluate trained model
print("\nStep 6: Evaluate trained model")
checkpoint = Path("./logs/training/webarena/tti_agent_real_data/checkpoints/best_model.pt")
if checkpoint.exists():
    subprocess.run([
        "./run_eval.sh",
        "--checkpoint", str(checkpoint),
        "--cost-preference", "balanced"
    ])

print("\n✓ Complete end-to-end pipeline finished!")
```

---

## Troubleshooting

### Issue: Trajectories not being saved
**Check**:
1. `trajectory_file` parameter is set in agent initialization
2. Output directory exists and is writable
3. No exceptions in `run_episode()` before `_save_trajectory()` call

### Issue: Wrong trajectory format
**Check**:
1. All required fields present: task_id, success, num_steps, num_tokens, modes
2. modes is a list of strings: ["THINK", "OBSERVE", "ANSWER"]
3. Each line is valid JSON (test with: `python3 -m json.tool < file.jsonl`)

### Issue: Low success rate in collected data
**This is normal!** Your agent's real success rate (45-60%) is more realistic than random mock data (50%). TOS-RL will learn from this real distribution.

### Issue: Too few trajectories per task
**Solution**: Increase collection time or lower `group_size` parameter.
```bash
./run_train_real_data.sh --trajectory-file data/traj.jsonl --group-size 2
```

---

## Best Practices

1. **Collect diverse trajectories**: Different task difficulties, domains, etc.
2. **Ensure realistic distribution**: Don't filter trajectories - include failures
3. **Use sufficient samples**: Aim for 100+ unique tasks × 4+ trajectories each
4. **Validate before training**: Run analysis and validation scripts
5. **Commit trajectory collection code**: Make it reproducible

---

## Files Reference

| File | Purpose |
|------|---------|
| TTI agent code | Modify to add trajectory collection |
| `data/trajectories.jsonl` | Output trajectory file (JSONL format) |
| `validate_trajectories.py` | Validate format |
| `analyze_collection.py` | Analyze statistics |
| `run_train_real_data.sh` | Train on collected data |

---

**Next**:
1. Modify your TTI agent following Step 1
2. Collect trajectories: `agent.run_batch(tasks, save_trajectories=True)`
3. Train: `./run_train_real_data.sh --trajectory-file data/trajectories.jsonl`

---

**Last Updated**: 2026-05-18
**Ready to integrate**: Yes
