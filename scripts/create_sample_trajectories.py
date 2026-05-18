#!/usr/bin/env python3
"""Generate sample trajectory data for testing."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """Generate synthetic trajectory data."""

    MODE_PATTERNS = [
        ["THINK", "OBSERVE", "ANSWER"],
        ["THINK", "OBSERVE", "OBSERVE", "ANSWER"],
        ["OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"],
        ["THINK", "OBSERVE", "OBSERVE", "OBSERVE", "ANSWER"],
        ["THINK", "THINK", "OBSERVE", "OBSERVE", "ANSWER"],
    ]

    @staticmethod
    def generate(num_tasks: int, num_per_task: int = 4) -> List[Dict[str, Any]]:
        """Generate synthetic trajectories."""
        trajectories = []

        for task_id in range(1, num_tasks + 1):
            for _ in range(num_per_task):
                trajectory = {
                    "task_id": f"task_{task_id:04d}",
                    "success": random.randint(0, 1),
                    "num_steps": random.randint(3, 25),
                    "num_tokens": random.randint(100, 800),
                    "num_loops": random.randint(0, 3),
                    "num_bad_actions": random.randint(0, 2),
                    "modes": random.choice(TrajectoryGenerator.MODE_PATTERNS),
                }
                trajectories.append(trajectory)

        return trajectories

    @staticmethod
    def save(trajectories: List[Dict[str, Any]], output_file: str) -> None:
        """Save trajectories to JSONL file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for traj in trajectories:
                f.write(json.dumps(traj) + "\n")

        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")

    @staticmethod
    def analyze(trajectories: List[Dict[str, Any]]) -> None:
        """Analyze trajectory statistics."""
        success_count = sum(1 for t in trajectories if t["success"])
        avg_steps = sum(t["num_steps"] for t in trajectories) / len(trajectories)
        avg_tokens = sum(t["num_tokens"] for t in trajectories) / len(trajectories)

        all_modes = []
        for t in trajectories:
            all_modes.extend(t["modes"])

        print(f"\nStatistics:")
        print(f"  Total: {len(trajectories)}")
        print(f"  Success rate: {100 * success_count / len(trajectories):.1f}%")
        print(f"  Avg steps: {avg_steps:.1f}")
        print(f"  Avg tokens: {avg_tokens:.0f}")
        print(f"  Mode distribution:")
        print(f"    THINK: {100 * all_modes.count('THINK') / len(all_modes):.1f}%")
        print(f"    OBSERVE: {100 * all_modes.count('OBSERVE') / len(all_modes):.1f}%")
        print(f"    ANSWER: {100 * all_modes.count('ANSWER') / len(all_modes):.1f}%")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate sample trajectories")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=20,
        help="Number of unique tasks"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=4,
        help="Trajectories per task"
    )
    parser.add_argument(
        "--output",
        default="../data/sample_trajectories.jsonl",
        help="Output file path"
    )

    args = parser.parse_args()
    total = args.num_tasks * args.num_trajectories

    logger.info(f"Generating {args.num_tasks} tasks × {args.num_trajectories} = {total} trajectories")

    trajectories = TrajectoryGenerator.generate(args.num_tasks, args.num_trajectories)
    TrajectoryGenerator.save(trajectories, args.output)
    TrajectoryGenerator.analyze(trajectories)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
