#!/usr/bin/env python3
"""TOS-RL Training Pipeline with Real Data Support."""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrajectoryLoader:
    """Load and validate trajectory data from JSONL files."""

    REQUIRED_FIELDS = {"task_id", "success", "num_steps", "num_tokens", "modes"}

    @staticmethod
    def load(filepath: str) -> List[Dict[str, Any]]:
        """Load and validate trajectory file."""
        trajectories = []
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return []

        logger.info(f"Loading trajectories from {filepath}...")

        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        traj = json.loads(line)
                        if not TrajectoryLoader._validate(traj):
                            missing = TrajectoryLoader.REQUIRED_FIELDS - set(traj.keys())
                            logger.warning(f"Line {line_num}: Missing {missing}")
                            continue
                        trajectories.append(traj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue

            logger.info(f"Loaded {len(trajectories)} valid trajectories")
            return trajectories

        except Exception as e:
            logger.error(f"Error loading trajectories: {e}")
            return []

    @staticmethod
    def _validate(traj: Dict[str, Any]) -> bool:
        """Validate trajectory has required fields."""
        return all(field in traj for field in TrajectoryLoader.REQUIRED_FIELDS)


class DataGrouper:
    """Group trajectories by task_id for training."""

    @staticmethod
    def group(trajectories: List[Dict[str, Any]], group_size: int) -> List[List[Dict[str, Any]]]:
        """Create K-sized groups from trajectories."""
        task_groups = {}

        for traj in trajectories:
            task_id = traj.get("task_id", "unknown")
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(traj)

        groups = []
        for task_id, trajs in task_groups.items():
            for i in range(0, len(trajs), group_size):
                group = trajs[i:i + group_size]
                if len(group) == group_size:
                    groups.append(group)

        logger.info(f"Created {len(groups)} training groups of size {group_size}")
        return groups


class PromptFormatter:
    """Format prompts for TOS-RL LLM input."""

    def __init__(self, tokenizer: Any):
        """Initialize formatter."""
        self.tokenizer = tokenizer
        self.mode_tokens = {"THINK": "<THINK>", "OBSERVE": "<OBSERVE>", "ANSWER": "<ANSWER>"}

    def format_prompt(
        self,
        task_description: str,
        observation_history: Optional[List[str]] = None,
        budget_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format prompt for LLM.

        Args:
            task_description: Task description
            observation_history: List of observations
            budget_info: Budget info (remaining steps, tokens, cost preference)

        Returns:
            Formatted prompt string
        """
        parts = [f"Task: {task_description}"]

        if observation_history:
            obs_text = "\n".join(f"Observation {i+1}: {obs}" for i, obs in enumerate(observation_history))
            parts.append(f"\nObservation History:\n{obs_text}")

        if budget_info:
            budget_text = "\n".join(f"{k}: {v}" for k, v in budget_info.items())
            parts.append(f"\nBudget Info:\n{budget_text}")

        parts.append("\nResponse: ")
        return "".join(parts)

    def extract_mode_token(self, text: str) -> Optional[str]:
        """Extract mode token from LLM output."""
        for mode_name, mode_token in self.mode_tokens.items():
            if mode_token in text:
                return mode_name
        return None


class UtilityComputer:
    """Compute cost-aware utilities from trajectories."""

    @staticmethod
    def compute_utilities(
        trajectories: List[Dict[str, Any]],
        cost_coefficients: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Compute utilities for trajectories.

        U(τ) = R_task - λ_env·N_env - λ_tok·N_tok - λ_loop·N_loop - λ_bad·N_bad
        """
        if cost_coefficients is None:
            # Default uniform distribution of costs
            cost_coefficients = {
                "env": 0.01,
                "tok": 0.001,
                "loop": 0.1,
                "bad": 0.2,
            }

        utilities = []
        for traj in trajectories:
            reward = float(traj.get("success", 0))
            num_steps = int(traj.get("num_steps", 0))
            num_tokens = int(traj.get("num_tokens", 0))
            num_loops = int(traj.get("num_loops", 0))
            num_bad = int(traj.get("num_bad_actions", 0))

            cost = (
                cost_coefficients["env"] * num_steps +
                cost_coefficients["tok"] * num_tokens +
                cost_coefficients["loop"] * num_loops +
                cost_coefficients["bad"] * num_bad
            )
            utility = reward - cost
            utilities.append(utility)

        return np.array(utilities)

    @staticmethod
    def compute_group_advantages(utilities: np.ndarray) -> np.ndarray:
        """
        Compute group-relative advantages.

        Â_i = (U_i - mean(U)) / std(U)
        """
        mean_u = np.mean(utilities)
        std_u = np.std(utilities)
        advantages = (utilities - mean_u) / (std_u + 1e-8)
        return advantages


class TrainingPipeline:
    """Training pipeline for TOS-RL with real data."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        trajectory_file: str,
        epochs: int,
        batch_size: int,
        group_size: int,
        model_name: str = "Qwen/Qwen2.5-4B-Instruct",
    ):
        """Initialize training pipeline."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories = TrajectoryLoader.load(trajectory_file)
        self.epochs = epochs
        self.batch_size = batch_size
        self.group_size = group_size
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_success = 0

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load LLM model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.train()

        # Reference model for KL penalty
        logger.info("Loading reference model for KL penalty")
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(config.get("training", {}).get("learning_rate", 1e-5))
        )

        # Prompt formatter and utility computer
        self.prompt_formatter = PromptFormatter(self.tokenizer)
        self.utility_computer = UtilityComputer()

        logger.info(f"Loaded {len(self.trajectories)} trajectories")
        logger.info(f"Output: {self.output_dir}")

    def run(self) -> None:
        """Execute training pipeline."""
        if not self.trajectories:
            logger.error("No trajectories loaded")
            return

        groups = DataGrouper.group(self.trajectories, self.group_size)
        if not groups:
            logger.error("No valid training groups")
            return

        results = []
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            metrics = self._train_epoch(epoch, groups)
            results.append({"epoch": epoch + 1, "metrics": metrics})
            self._save_checkpoint(epoch, metrics)

        self._save_results(results)
        logger.info("Training completed successfully")

    def _train_epoch(self, epoch: int, groups: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Train single epoch with GRPO."""
        epoch_metrics = {
            "total_loss": [],
            "policy_loss": [],
            "mode_loss": [],
            "kl_loss": [],
            "success_rate": [],
            "mean_utility": [],
        }

        num_groups = min(100, len(groups))  # Limit for speed
        for group_idx, group in enumerate(groups[:num_groups]):
            # Compute utilities and advantages
            utilities = self.utility_computer.compute_utilities(group)
            advantages = self.utility_computer.compute_group_advantages(utilities)

            success_rate = sum(t["success"] for t in group) / len(group)
            mean_utility = float(np.mean(utilities))

            epoch_metrics["success_rate"].append(success_rate)
            epoch_metrics["mean_utility"].append(mean_utility)

            try:
                # Forward pass and loss computation
                losses = self._compute_grpo_loss(group, advantages)

                if losses is not None:
                    total_loss, policy_loss, mode_loss, kl_loss = losses

                    epoch_metrics["total_loss"].append(total_loss.item())
                    epoch_metrics["policy_loss"].append(policy_loss.item())
                    epoch_metrics["mode_loss"].append(mode_loss.item())
                    epoch_metrics["kl_loss"].append(kl_loss.item())

                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

            except Exception as e:
                logger.warning(f"Error training group {group_idx}: {e}")
                continue

            if (group_idx + 1) % 20 == 0:
                avg_loss = float(np.mean(epoch_metrics["total_loss"][-20:])) if epoch_metrics["total_loss"] else 0.0
                logger.info(
                    f"  Group {group_idx + 1}/{num_groups}: "
                    f"loss={avg_loss:.4f}, success={success_rate:.2%}, "
                    f"utility={mean_utility:.4f}"
                )

        # Average metrics
        return {
            key: float(np.mean(values)) if values else 0.0
            for key, values in epoch_metrics.items()
        }

    def _compute_grpo_loss(
        self,
        group: List[Dict[str, Any]],
        advantages: np.ndarray,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute GRPO loss for a group.

        Returns:
            (total_loss, policy_loss, mode_loss, kl_loss)
        """
        # Prepare inputs
        batch_size = len(group)
        device = self.device

        # Compute log probs for each trajectory
        log_probs_list = []
        log_probs_ref_list = []
        mode_log_probs_list = []

        for i, traj in enumerate(group):
            # Create prompt from trajectory
            task_description = traj.get("task_id", "unknown")
            prompt = self.prompt_formatter.format_prompt(task_description)

            try:
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]

                # Forward pass through model (with gradients for policy)
                outputs = self.model(input_ids, output_hidden_states=True)
                logits = outputs.logits

                # Compute log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                # Use last token log prob as representative (mode/action token)
                log_prob = log_probs[0, -1, :].mean()
                log_probs_list.append(log_prob)

                # Reference model log probs (no gradient)
                with torch.no_grad():
                    ref_outputs = self.reference_model(input_ids, output_hidden_states=True)
                    ref_logits = ref_outputs.logits
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_log_prob = ref_log_probs[0, -1, :].mean()
                    log_probs_ref_list.append(ref_log_prob)

                # Mode emphasis: upweight first token (with gradients)
                mode_log_prob = log_probs[0, 0, :].max()
                mode_log_probs_list.append(mode_log_prob)

            except Exception as e:
                logger.warning(f"Error processing trajectory {i}: {e}")
                return None

        # Stack into tensors
        log_probs = torch.stack(log_probs_list)
        log_probs_ref = torch.stack(log_probs_ref_list)
        mode_log_probs = torch.stack(mode_log_probs_list)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

        # Compute policy loss (PPO-style)
        ratios = torch.exp(log_probs - log_probs_ref)
        clip_ratio = 0.2
        surr1 = ratios * advantages_tensor
        surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages_tensor
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Compute mode emphasis loss
        mode_weight = 2.0
        mode_loss = -mode_weight * torch.mean(mode_log_probs * advantages_tensor)

        # Compute KL penalty
        kl_weight = 0.1
        kl_div = log_probs - log_probs_ref
        kl_loss = kl_weight * torch.mean(kl_div)

        # Total loss
        total_loss = policy_loss + mode_loss + kl_loss

        return total_loss, policy_loss, mode_loss, kl_loss

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        # Save per-epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Update best checkpoint
        current_success = metrics.get("success_rate", 0)
        if current_success > self.best_success:
            self.best_success = current_success
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Updated best checkpoint: {best_path}")

        # Keep last checkpoint
        last_path = self.checkpoint_dir / "last_model.pt"
        torch.save(checkpoint, last_path)

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save training results."""
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {results_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TOS-RL Training with Real Data")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--trajectory-file", required=True, help="Path to JSONL trajectory file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--group-size", type=int, default=32, help="Group size (K for GRPO)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--dataset", default="webarena", help="Dataset name (webarena|webvoyager)")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-4B-Instruct",
        help="Model name (e.g., Qwen/Qwen2.5-4B-Instruct or Qwen/Qwen2.5-8B-Instruct)"
    )

    args = parser.parse_args()

    logger.info(f"=== TOS-RL Training Pipeline ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Group size (K): {args.group_size}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded: {config['dataset']['name']}")

    # Run pipeline
    pipeline = TrainingPipeline(
        config=config,
        output_dir=args.output_dir,
        trajectory_file=args.trajectory_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        model_name=args.model,
    )
    pipeline.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
