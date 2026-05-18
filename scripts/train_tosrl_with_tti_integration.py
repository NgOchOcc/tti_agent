#!/usr/bin/env python3
"""
Integration example: TOS-RL with TTI trajectory collection.

This script shows how to:
1. Collect trajectories from TTI agent
2. Track mode tokens ([THINK], [OBSERVE], [ANSWER])
3. Apply TOS-RL training on top

Note: This is a template. Replace mock functions with your actual TTI code.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tti.tos_rl import (
    TOSRLTrainer, TOSRLConfig,
    BudgetState, TokenModeFormatter,
    batch_compute_utilities,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TTITOSRLIntegration:
    """Integration of TOS-RL training with TTI trajectory collection."""

    def __init__(
        self,
        config: TOSRLConfig,
        num_epochs: int = 10,
        trajectories_per_task: int = 4,
        tasks_per_epoch: int = 32,
    ):
        """
        Initialize integration.

        Args:
            config: TOS-RL configuration
            num_epochs: Training epochs
            trajectories_per_task: K (group size)
            tasks_per_epoch: Tasks per epoch
        """
        self.config = config
        self.num_epochs = num_epochs
        self.trajectories_per_task = trajectories_per_task
        self.tasks_per_epoch = tasks_per_epoch
        self.trainer = TOSRLTrainer(config)

        logger.info(f"Initialized TOS-RL + TTI integration with config: {config}")

    def create_budget_prompt_suffix(self, budget: BudgetState) -> str:
        """Create budget constraint suffix for LLM prompt."""
        return f"""
Budget Constraints:
- Remaining browser interactions: {int(budget.env_interactions)}
- Remaining tokens: {int(budget.tokens)}
- Remaining time: {budget.time:.1f}s

Choose wisely! Use [THINK] for reasoning, [OBSERVE] for actions, [ANSWER] to stop.
"""

    def collect_trajectory_from_tti(
        self,
        task: Dict[str, Any],
        policy_model: Any,  # Your LLM policy
        initial_budget: BudgetState,
    ) -> Dict[str, Any]:
        """
        Collect a single trajectory from TTI agent.

        Args:
            task: Task specification
            policy_model: Your LLM policy model
            initial_budget: Initial budget constraints

        Returns:
            Trajectory dict with mode tokens tracked
        """

        history = ""
        modes = []
        num_steps = 0
        total_tokens = 0
        num_loops = 0
        num_bad_actions = 0
        success = 0

        budget = BudgetState(
            env_interactions=initial_budget.env_interactions,
            tokens=initial_budget.tokens,
            time=initial_budget.time,
        )

        # Episode loop
        max_iterations = 100
        iteration = 0

        while budget.env_interactions > 0 and iteration < max_iterations:
            iteration += 1

            # Build prompt with budget info
            budget_suffix = self.create_budget_prompt_suffix(budget)
            prompt = f"""Task: {task['description']}

Current History:
{history}

{budget_suffix}

Response:"""

            # Get LLM output
            # In actual TTI, this would be from your trained policy model
            llm_output = self.mock_llm_generate(prompt, task, iteration)

            # Count tokens
            tokens_generated = len(llm_output.split())
            total_tokens += tokens_generated
            budget.tokens -= tokens_generated

            # Parse mode token
            mode, content = TokenModeFormatter.parse_mode_and_content(llm_output)

            if mode is None:
                # Default to OBSERVE if mode not recognized
                mode = "OBSERVE"

            modes.append(mode)

            # Execute mode
            if mode == "THINK":
                # Internal reasoning - no environment interaction
                history += f"\n[THINK] {content}"
                logger.debug(f"  [THINK] Reasoning step")

            elif mode == "OBSERVE":
                # Browser interaction
                action = self.parse_action(content)
                observation = self.mock_execute_action(action, task, history)

                history += f"\n[OBSERVE] {content}\nObservation: {observation}"

                num_steps += 1
                budget.env_interactions -= 1

                # Track loops
                if self.is_repeated_action(action, history):
                    num_loops += 1

                # Track invalid actions
                if not self.is_valid_action(action, task):
                    num_bad_actions += 1

                logger.debug(f"  [OBSERVE] Step {num_steps}, action: {action}")

            elif mode == "ANSWER":
                # Final answer
                answer = self.extract_answer(content)
                history += f"\n[ANSWER] {answer}"

                # Evaluate success
                success = self.evaluate_answer(answer, task)
                logger.debug(f"  [ANSWER] {answer} (success={success})")
                break

            else:
                logger.warning(f"Unknown mode: {mode}")
                break

        # Build trajectory
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

    def collect_group_trajectories(
        self,
        task: Dict[str, Any],
        policy_model: Any,
        num_trajectories: int = 4,
    ) -> List[Dict[str, Any]]:
        """Collect K trajectories from same task."""
        trajectories = []

        for i in range(num_trajectories):
            traj = self.collect_trajectory_from_tti(
                task=task,
                policy_model=policy_model,
                initial_budget=BudgetState(
                    env_interactions=30,
                    tokens=4096,
                    time=300.0,
                ),
            )
            trajectories.append(traj)
            logger.info(f"  Trajectory {i+1}/{num_trajectories}: "
                       f"success={traj['success']}, steps={traj['num_steps']}")

        return trajectories

    def prepare_training_batch(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch for TOS-RL training."""

        # Compute utilities with randomized costs
        utilities, costs = self.trainer.process_trajectories(trajectories)

        # Compute group-relative advantages (no critic!)
        advantages = self.trainer.compute_group_advantages(utilities)

        # Mock log probabilities (in real training, get from model inference)
        num_traj = len(trajectories)
        log_probs = torch.randn(num_traj, dtype=torch.float32)
        log_probs_old = torch.randn(num_traj, dtype=torch.float32)
        log_probs_ref = torch.randn(num_traj, dtype=torch.float32)

        # Mock mode log probs (in real training, get from model)
        mode_log_probs = torch.randn(num_traj, dtype=torch.float32)

        # Mode probabilities [THINK, OBSERVE, ANSWER]
        mode_probs = torch.ones(num_traj, 3) / 3

        batch = {
            "log_probs": log_probs,
            "log_probs_old": log_probs_old,
            "log_probs_ref": log_probs_ref,
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "mode_log_probs": mode_log_probs,
            "mode_probs": mode_probs,
        }

        return batch

    def train_epoch(
        self,
        epoch: int,
        tasks: List[Dict[str, Any]],
        policy_model: Any,
        val_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Train for one epoch."""

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
        logger.info(f"{'='*60}")

        epoch_metrics = {
            "total_loss": [],
            "policy_loss": [],
            "mode_loss": [],
            "success_rate": [],
        }

        num_tasks = len(tasks)

        for task_idx, task in enumerate(tasks):
            logger.info(f"\nTask {task_idx + 1}/{num_tasks}: {task['id']}")

            # Collect K trajectories from same task
            group = self.collect_group_trajectories(
                task=task,
                policy_model=policy_model,
                num_trajectories=self.trajectories_per_task,
            )

            # Prepare batch
            batch = self.prepare_training_batch(group)

            # TOS-RL training step
            metrics = self.trainer.training_step(batch)

            # Track metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])

            success_rate = sum(t['success'] for t in group) / len(group)
            epoch_metrics['success_rate'].append(success_rate)

        # Compute epoch statistics
        epoch_stats = {}
        for key, values in epoch_metrics.items():
            if values:
                avg = np.mean(values)
                epoch_stats[f"train/{key}"] = avg
                logger.info(f"  {key}: {avg:.4f}")

        # Validation
        if val_tasks:
            val_stats = self.evaluate_on_tasks(val_tasks, policy_model)
            epoch_stats.update(val_stats)
            logger.info(
                f"\nValidation - "
                f"Success: {val_stats.get('val/success_rate', 0):.2%}, "
                f"Avg Steps: {val_stats.get('val/mean_steps', 0):.1f}"
            )

        return epoch_stats

    def evaluate_on_tasks(
        self,
        tasks: List[Dict[str, Any]],
        policy_model: Any,
    ) -> Dict[str, float]:
        """Evaluate on a set of tasks."""

        success_rates = []
        num_steps_list = []

        for task in tasks:
            group = self.collect_group_trajectories(
                task=task,
                policy_model=policy_model,
                num_trajectories=1,  # Single trajectory for eval
            )

            success_rate = sum(t['success'] for t in group) / len(group)
            mean_steps = np.mean([t['num_steps'] for t in group])

            success_rates.append(success_rate)
            num_steps_list.append(mean_steps)

        return {
            "val/success_rate": np.mean(success_rates),
            "val/mean_steps": np.mean(num_steps_list),
        }

    def run_training(
        self,
        policy_model: Any,
    ):
        """Run complete training loop."""

        logger.info("Starting TOS-RL + TTI integration training...")

        for epoch in range(self.num_epochs):
            # Sample tasks for this epoch
            train_tasks = self.sample_tasks(self.tasks_per_epoch, split='train')

            # Train epoch
            metrics = self.train_epoch(epoch, train_tasks, policy_model)

            # Sample validation tasks
            val_tasks = self.sample_tasks(16, split='val')

            # Log results
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoints/tosrl_tti_epoch_{epoch + 1}.pt"
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                # In real training: torch.save(policy_model.state_dict(), checkpoint_path)

        logger.info("Training completed!")

    # ========== Mock Functions (Replace with Real TTI Code) ==========

    def mock_llm_generate(
        self, prompt: str, task: Dict[str, Any], iteration: int
    ) -> str:
        """Mock LLM generation. Replace with your actual policy model."""
        import random

        # In real TTI, this would be: output = policy_model.generate(prompt)

        # Mock response with mode token
        modes = ["[THINK]", "[OBSERVE]", "[ANSWER]"]
        mode = random.choice(modes)

        if mode == "[THINK]":
            return "[THINK] Let me reason about this task..."
        elif mode == "[OBSERVE]":
            actions = [
                "[OBSERVE] Click on the search box",
                "[OBSERVE] Type 'product price'",
                "[OBSERVE] Press enter",
                "[OBSERVE] Look for the price element",
            ]
            return random.choice(actions)
        else:
            return "[ANSWER] The task is complete"

    def parse_action(self, content: str) -> str:
        """Parse action from LLM output."""
        return content.strip()

    def mock_execute_action(
        self, action: str, task: Dict[str, Any], history: str
    ) -> str:
        """Mock environment action execution."""
        return f"Action executed: {action}. New state: some observation."

    def is_repeated_action(self, action: str, history: str) -> bool:
        """Check if action is repeated."""
        return history.count(action) > 1

    def is_valid_action(self, action: str, task: Dict[str, Any]) -> bool:
        """Check if action is valid."""
        return len(action) > 0 and "invalid" not in action.lower()

    def extract_answer(self, content: str) -> str:
        """Extract final answer from content."""
        return content.strip()

    def evaluate_answer(self, answer: str, task: Dict[str, Any]) -> int:
        """Evaluate if answer is correct."""
        # Return 1 for correct, 0 for incorrect
        return 1 if answer else 0

    def sample_tasks(
        self, num_tasks: int, split: str = 'train'
    ) -> List[Dict[str, Any]]:
        """Sample tasks from dataset."""
        tasks = []
        for i in range(num_tasks):
            tasks.append({
                "id": f"task_{split}_{i}",
                "description": f"Find information about product {i}",
            })
        return tasks


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TOS-RL with TTI integration")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--tasks-per-epoch", type=int, default=16, help="Tasks per epoch")
    parser.add_argument("--trajectories-per-task", type=int, default=4, help="K trajectories per task")

    args = parser.parse_args()

    # Configure TOS-RL
    config = TOSRLConfig(
        group_size=args.trajectories_per_task,
        max_prefixes_per_task=3,
        branches_per_prefix=2,
        learning_rate=1e-5,
        mode_token_weight=2.0,
        use_branching=False,  # Start without branching
        device=torch.device("cpu"),
    )

    # Create integration
    integration = TTITOSRLIntegration(
        config=config,
        num_epochs=args.epochs,
        trajectories_per_task=args.trajectories_per_task,
        tasks_per_epoch=args.tasks_per_epoch,
    )

    # Mock policy model (replace with your actual TTI agent)
    class MockPolicyModel:
        def generate(self, prompt: str) -> str:
            import random
            modes = ["[THINK]", "[OBSERVE]", "[ANSWER]"]
            mode = random.choice(modes)
            return f"{mode} mock output"

    policy_model = MockPolicyModel()

    # Run training
    integration.run_training(policy_model)

    logger.info("\nIntegration training complete!")
    logger.info("Next steps:")
    logger.info("1. Replace mock functions with real TTI agent code")
    logger.info("2. Connect to actual WebArena environment")
    logger.info("3. Enable prefix branching for improved credit assignment")
    logger.info("4. Evaluate on WebVoyager for transfer")


if __name__ == "__main__":
    main()
