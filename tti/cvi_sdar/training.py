"""
Unified training loop for CVI-SDAR.

Integrates all components:
- Mode policy π_φ
- Critics V, Q, V_ans
- CVI-gated self-distillation
- RL objective (PPO)
- Prefix mining and branch generation
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .mode_policy import ModePolicy, PPOModePolicy
from .critics import CriticNetwork, CriticLosses
from .distillation import CVIDistillationLoss, CVIGate
from .prefix_mining import PrefixMiner, BranchGenerator
from .utils import compute_cvi_gap, select_mode_by_cvi

logger = logging.getLogger(__name__)


@dataclass
class CVISARConfig:
    """Configuration for CVI-SDAR training."""

    # Architecture
    state_embedding_dim: int = 768
    mode_hidden_dims: Tuple = (128,)
    critic_hidden_dims: Tuple = (256, 128)
    num_modes: int = 6
    dropout: float = 0.1

    # Learning rates
    lr_mode: float = 1e-4
    lr_critics: float = 1e-4
    lr_distillation: float = 1e-4

    # Loss weights
    weight_rl: float = 1.0
    weight_critic: float = 0.5
    weight_ans: float = 0.5
    weight_distillation: float = 0.2
    weight_kl: float = 0.1
    weight_entropy: float = 0.01

    # PPO parameters
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 3
    batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Distillation parameters
    distillation_beta: float = 2.0
    distillation_eta: float = 0.5
    distillation_rho: float = 0.0

    # Curriculum
    curriculum_steps: int = 1000
    initial_cost_scale: float = 0.0
    final_cost_scale: float = 1.0

    # Prefix mining
    prefix_sampling_strategy: str = "mixed"
    prefixes_per_trajectory: int = 5
    branches_per_prefix: int = 3

    device: torch.device = torch.device("cpu")


class CVISARTrainer:
    """
    Main trainer for CVI-SDAR.

    Orchestrates training loop combining RL, critics, and distillation.
    """

    def __init__(
        self,
        config: CVISARConfig,
        reference_policy: Optional[torch.nn.Module] = None,
    ):
        """
        Args:
            config: Training configuration
            reference_policy: Reference policy for KL penalty (optional)
        """
        self.config = config
        self.device = config.device
        self.reference_policy = reference_policy
        self.global_step = 0

        # Initialize models
        self._init_models()

        # Initialize optimizers
        self._init_optimizers()

        # Initialize loss functions
        self._init_losses()

        # Initialize miners and generators
        self.prefix_miner = PrefixMiner(
            sampling_strategy=config.prefix_sampling_strategy,
            max_prefixes_per_trajectory=config.prefixes_per_trajectory,
        )
        self.branch_generator = BranchGenerator(
            num_modes=config.num_modes,
            max_branch_length=None,
        )

    def _init_models(self):
        """Initialize all neural network models."""
        self.mode_policy = ModePolicy(
            input_dim=self.config.state_embedding_dim,
            hidden_dims=self.config.mode_hidden_dims,
            dropout=self.config.dropout,
            num_modes=self.config.num_modes,
        ).to(self.device)

        self.critics = {
            "v": CriticNetwork(
                input_dim=self.config.state_embedding_dim,
                hidden_dims=self.config.critic_hidden_dims,
                dropout=self.config.dropout,
                output_dim=1,
            ).to(self.device),
            "q": CriticNetwork(
                input_dim=self.config.state_embedding_dim,
                hidden_dims=self.config.critic_hidden_dims,
                dropout=self.config.dropout,
                output_dim=self.config.num_modes,
            ).to(self.device),
            "v_ans": CriticNetwork(
                input_dim=self.config.state_embedding_dim,
                hidden_dims=self.config.critic_hidden_dims,
                dropout=self.config.dropout,
                output_dim=1,
            ).to(self.device),
        }

    def _init_optimizers(self):
        """Initialize optimizers."""
        self.opt_mode = optim.Adam(
            self.mode_policy.parameters(),
            lr=self.config.lr_mode
        )

        self.opt_critics = optim.Adam(
            list(self.critics["v"].parameters()) +
            list(self.critics["q"].parameters()) +
            list(self.critics["v_ans"].parameters()),
            lr=self.config.lr_critics
        )

    def _init_losses(self):
        """Initialize loss functions."""
        self.distillation_loss = CVIDistillationLoss(
            num_modes=self.config.num_modes,
            beta=self.config.distillation_beta,
            eta=self.config.distillation_eta,
            rho=self.config.distillation_rho,
        ).to(self.device)

        self.critic_losses = CriticLosses()

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        prefix_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of trajectories with:
                - "state_embeddings": [batch_size, state_dim]
                - "modes": [batch_size] mode indices
                - "returns": [batch_size] returns
                - "old_log_probs": [batch_size] old log probs
                - "advantages": [batch_size] advantages
            prefix_data: Optional prefix and branch data for distillation

        Returns:
            Dictionary of loss values and metrics
        """
        self.global_step += 1
        losses = {}

        # Extract batch data
        state_embeddings = batch["state_embeddings"].to(self.device)
        modes = batch["modes"].to(self.device)
        returns = batch["returns"].to(self.device)
        old_log_probs = batch.get("old_log_probs").to(self.device)
        advantages = batch.get("advantages").to(self.device)

        # =======================
        # 1. CRITIC TRAINING
        # =======================
        self.opt_critics.zero_grad()

        v_pred = self.critics["v"](state_embeddings).squeeze(-1)
        q_pred = self.critics["q"](state_embeddings)

        critic_loss = self.critic_losses.critic_loss(
            v_pred, q_pred, returns, modes
        )

        # Answer loss (if available)
        ans_loss = torch.tensor(0.0, device=self.device)
        if "answer_utilities" in batch:
            v_ans_pred = self.critics["v_ans"](state_embeddings).squeeze(-1)
            answer_utilities = batch["answer_utilities"].to(self.device)
            ans_loss = self.critic_losses.answer_loss(v_ans_pred, answer_utilities)

        total_critic_loss = (
            self.config.weight_critic * critic_loss +
            self.config.weight_ans * ans_loss
        )

        total_critic_loss.backward()
        self.opt_critics.step()

        losses["critic_loss"] = critic_loss.item()
        losses["ans_loss"] = ans_loss.item()

        # =======================
        # 2. RL TRAINING (PPO)
        # =======================
        self.opt_mode.zero_grad()

        # Get current mode log probs
        mode_probs, mode_logits = self.mode_policy(state_embeddings, return_logits=True)
        current_log_probs = torch.log(
            mode_probs[torch.arange(len(modes)), modes] + 1e-8
        )

        # PPO clipped objective
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.ppo_clip_ratio,
            1 + self.config.ppo_clip_ratio
        ) * advantages

        ppo_loss = -torch.mean(torch.min(surr1, surr2))

        # Entropy regularization
        entropy = -torch.sum(mode_probs * torch.log(mode_probs + 1e-8), dim=-1).mean()
        entropy_loss = -self.config.weight_entropy * entropy

        total_rl_loss = (
            self.config.weight_rl * ppo_loss +
            entropy_loss
        )

        total_rl_loss.backward()
        self.opt_mode.step()

        losses["ppo_loss"] = ppo_loss.item()
        losses["entropy"] = entropy.item()

        # =======================
        # 3. SELF-DISTILLATION (if available)
        # =======================
        if prefix_data is not None:
            distill_loss = self._train_distillation(
                prefix_data, state_embeddings
            )
            losses["distill_loss"] = distill_loss.item()

        return losses

    def _train_distillation(
        self,
        prefix_data: Dict[str, Any],
        state_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training step for distillation component.

        Args:
            prefix_data: Prefix and branch data
            state_embeddings: State embeddings for affected prefixes

        Returns:
            Distillation loss
        """
        # This would typically:
        # 1. Extract teacher modes and CVI advantages from branches
        # 2. Compute mode logits for prefixes
        # 3. Call distillation_loss function
        # 4. Backprop

        # Simplified implementation:
        teacher_modes = prefix_data.get("teacher_modes")
        cvi_advantages = prefix_data.get("cvi_advantages")

        if teacher_modes is None or cvi_advantages is None:
            return torch.tensor(0.0, device=self.device)

        teacher_modes = torch.tensor(teacher_modes, device=self.device)
        cvi_advantages = torch.tensor(cvi_advantages, device=self.device)

        # Mode logits
        mode_probs, mode_logits = self.mode_policy(
            state_embeddings[:len(teacher_modes)],
            return_logits=True
        )

        # Distillation loss
        distill_output = self.distillation_loss(
            mode_logits,
            teacher_modes,
            cvi_advantages,
        )

        loss = distill_output["total_loss"]

        return self.config.weight_distillation * loss

    def evaluate(
        self,
        val_trajectories: List[Dict[str, Any]],
        get_state_embedding: callable,
    ) -> Dict[str, float]:
        """
        Evaluation on validation trajectories.

        Args:
            val_trajectories: List of validation trajectories
            get_state_embedding: Function to extract state embeddings from trajectories

        Returns:
            Dictionary of evaluation metrics
        """
        self.mode_policy.eval()
        for critic in self.critics.values():
            critic.eval()

        metrics = {
            "avg_v_estimate": 0,
            "avg_q_estimate": 0,
            "avg_v_ans_estimate": 0,
            "avg_cvi_gap": 0,
        }

        num_samples = 0

        with torch.no_grad():
            for traj in val_trajectories[:10]:  # Sample of val set
                state_emb = get_state_embedding(traj)
                if state_emb is None:
                    continue

                state_emb = torch.tensor(
                    state_emb, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                v = self.critics["v"](state_emb).item()
                q = self.critics["q"](state_emb)[0]
                v_ans = self.critics["v_ans"](state_emb).item()

                metrics["avg_v_estimate"] += v
                metrics["avg_q_estimate"] += q.mean().item()
                metrics["avg_v_ans_estimate"] += v_ans
                metrics["avg_cvi_gap"] += (q.max().item() - v_ans)

                num_samples += 1

        if num_samples > 0:
            for key in metrics:
                metrics[key] /= num_samples

        self.mode_policy.train()
        for critic in self.critics.values():
            critic.train()

        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "mode_policy": self.mode_policy.state_dict(),
            "critics": {k: v.state_dict() for k, v in self.critics.items()},
            "opt_mode": self.opt_mode.state_dict(),
            "opt_critics": self.opt_critics.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.mode_policy.load_state_dict(checkpoint["mode_policy"])
        for k, v in checkpoint["critics"].items():
            self.critics[k].load_state_dict(v)

        self.opt_mode.load_state_dict(checkpoint["opt_mode"])
        self.opt_critics.load_state_dict(checkpoint["opt_critics"])

        self.global_step = checkpoint.get("global_step", 0)

        logger.info(f"Loaded checkpoint from {path}")
