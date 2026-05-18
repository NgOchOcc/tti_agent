#!/usr/bin/env python3
"""
Create sample trajectory file for testing real data training.

This script generates synthetic trajectories in the correct JSONL format.
In production, trajectories come from your actual TTI agent.

Usage:
  python3 create_sample_trajectories.py                    # 100 trajectories
  python3 create_sample_trajectories.py --num-tasks 500    # 500 trajectories
  python3 create_sample_trajectories.py --num-trajectories 4  # 4 per task
"""

import json
import argparse
import random
from pathlib import Path


def create_sample_trajectories(
    num_tasks: int = 100,
    num_trajectories_per_task: int = 4,
    output_file: str = "../data/sample_trajectories.jsonl"
) -> None:
    """
    Create sample trajectory file for testing.

    Args:
        num_tasks: Number of unique tasks
        num_trajectories_per_task: Trajectories per task
        output_file: Output JSONL file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_tasks} tasks × {num_trajectories_per_task} trajectories...")
    print(f"Output: {output_path}")

    total_trajectories = 0

    # Mode patterns (realistic patterns from LLM agents)
    mode_patterns = [
        ["THINK", "OBSERVE", "ANSWER"],
        ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
        ["OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"],
        ["THINK", "OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"],
        ["THINK", "THINK", "OBSERVE", "OBSERVE", "ANSWER"],
        ["OBSERVE", "ANSWER"],
        ["THINK", "ANSWER"],
    ]

    with open(output_path, "w") as f:
        for task_id in range(1, num_tasks + 1):
            for traj_idx in range(num_trajectories_per_task):
                # Generate trajectory
                trajectory = {
                    "task_id": f"task_{task_id:04d}",
                    "success": random.randint(0, 1),
                    "num_steps": random.randint(3, 25),
                    "num_tokens": random.randint(100, 800),
                    "num_loops": random.randint(0, 3),
                    "num_bad_actions": random.randint(0, 2),
                    "modes": random.choice(mode_patterns),
                }

                # Write to JSONL
                f.write(json.dumps(trajectory) + "\n")
                total_trajectories += 1

    # Verify file
    with open(output_path, "r") as f:
        lines = f.readlines()

    print(f"\n✓ Generated {len(lines)} trajectories")

    # Analyze
    trajectories = [json.loads(line) for line in lines]
    success_count = sum(1 for t in trajectories if t["success"])

    print(f"\nStatistics:")
    print(f"  Unique tasks: {num_tasks}")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Success rate: {100 * success_count / len(trajectories):.1f}%")
    print(f"  Avg steps: {sum(t['num_steps'] for t in trajectories) / len(trajectories):.1f}")
    print(f"  Avg tokens: {sum(t['num_tokens'] for t in trajectories) / len(trajectories):.0f}")

    # Sample
    print(f"\nFirst 3 trajectories:")
    for i, traj in enumerate(trajectories[:3]):
        print(f"  {i+1}. task_id={traj['task_id']}, success={traj['success']}, "
              f"steps={traj['num_steps']}, tokens={traj['num_tokens']}")

    print(f"\n✓ Ready for training!")
    print(f"\nNext: Run training with")
    print(f"  ./run_train_real_data.sh --trajectory-file {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create sample trajectory file for testing"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Number of unique tasks (default: 100)"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=4,
        help="Trajectories per task (default: 4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/sample_trajectories.jsonl",
        help="Output file path"
    )

    args = parser.parse_args()

    create_sample_trajectories(
        num_tasks=args.num_tasks,
        num_trajectories_per_task=args.num_trajectories,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
