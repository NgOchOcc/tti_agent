"""
Unit tests for CVI-SDAR components.

Tests:
1. Critics output correct shapes
2. Mode policy produces valid distributions
3. CVI gap computation is correct
4. Distillation gate works properly
5. Training loop completes without errors
6. Inference controller makes valid decisions
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tti.cvi_sdar import (
    CriticNetwork,
    ModePolicy,
    CVIDistillationLoss,
    PrefixMiner,
    BranchGenerator,
    CVISARTrainer,
    CVISARConfig,
    CVISARInferenceController,
    compute_cvi_gap,
    select_mode_by_cvi,
    Budget,
)


class TestCritics:
    """Test critic networks."""

    @pytest.fixture
    def setup(self):
        """Setup for critic tests."""
        self.input_dim = 768
        self.num_modes = 6
        self.batch_size = 4

        self.critic_v = CriticNetwork(
            input_dim=self.input_dim,
            hidden_dims=(128,),
            output_dim=1,
        )

        self.critic_q = CriticNetwork(
            input_dim=self.input_dim,
            hidden_dims=(128,),
            output_dim=self.num_modes,
        )

    def test_critic_v_output_shape(self, setup):
        """Test V critic output shape."""
        state_emb = torch.randn(self.batch_size, self.input_dim)
        output = self.critic_v(state_emb)
        assert output.shape == (self.batch_size,), f"Expected shape ({self.batch_size},), got {output.shape}"

    def test_critic_q_output_shape(self, setup):
        """Test Q critic output shape."""
        state_emb = torch.randn(self.batch_size, self.input_dim)
        output = self.critic_q(state_emb)
        assert output.shape == (self.batch_size, self.num_modes), f"Expected shape ({self.batch_size}, {self.num_modes}), got {output.shape}"

    def test_critic_gradients(self, setup):
        """Test that gradients flow through critics."""
        state_emb = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        output = self.critic_v(state_emb)
        loss = output.mean()
        loss.backward()

        assert state_emb.grad is not None
        assert state_emb.grad.norm() > 0

    def test_critic_eval_mode(self, setup):
        """Test eval mode (no gradients)."""
        self.critic_v.eval()

        with torch.no_grad():
            state_emb = torch.randn(self.batch_size, self.input_dim)
            output1 = self.critic_v(state_emb)
            output2 = self.critic_v(state_emb)

            # Should be deterministic in eval mode
            assert torch.allclose(output1, output2)


class TestModePolicy:
    """Test mode policy."""

    @pytest.fixture
    def setup(self):
        """Setup for mode policy tests."""
        self.input_dim = 768
        self.num_modes = 6
        self.batch_size = 4

        self.policy = ModePolicy(
            input_dim=self.input_dim,
            hidden_dims=(128,),
            num_modes=self.num_modes,
        )

    def test_mode_probs_sum_to_one(self, setup):
        """Test that mode probabilities sum to 1."""
        state_emb = torch.randn(self.batch_size, self.input_dim)
        probs = self.policy(state_emb)

        assert probs.shape == (self.batch_size, self.num_modes)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(self.batch_size))

    def test_mode_probs_valid_range(self, setup):
        """Test that probabilities are in [0, 1]."""
        state_emb = torch.randn(self.batch_size, self.input_dim)
        probs = self.policy(state_emb)

        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_mode_sampling(self, setup):
        """Test mode sampling."""
        state_emb = torch.randn(self.batch_size, self.input_dim)
        modes, log_probs = self.policy.sample_mode(state_emb)

        assert modes.shape == (self.batch_size,)
        assert log_probs.shape == (self.batch_size,)
        assert (modes >= 0).all()
        assert (modes < self.num_modes).all()
        assert (log_probs < 0).all()  # Log probs should be negative

    def test_deterministic_sampling(self, setup):
        """Test deterministic mode selection."""
        state_emb = torch.randn(self.batch_size, self.input_dim)
        modes, _ = self.policy.sample_mode(state_emb, deterministic=True)

        assert (modes >= 0).all()
        assert (modes < self.num_modes).all()


class TestCVIUtils:
    """Test CVI utility functions."""

    def test_cvi_gap_computation(self):
        """Test CVI gap computation."""
        batch_size = 4
        num_modes = 6

        q_values = torch.randn(batch_size, num_modes)
        v_ans = torch.randn(batch_size)

        cvi_gaps = compute_cvi_gap(q_values, v_ans)

        assert cvi_gaps.shape == (batch_size, num_modes)

        # Manually verify for first sample
        expected = q_values[0] - v_ans[0]
        assert torch.allclose(cvi_gaps[0], expected, atol=1e-5)

    def test_mode_selection_by_cvi(self):
        """Test mode selection based on CVI gaps."""
        batch_size = 4
        num_modes = 6

        q_values = torch.randn(batch_size, num_modes)
        v_ans = torch.randn(batch_size)

        modes, gaps = select_mode_by_cvi(q_values, v_ans, delta_threshold=0.5)

        assert modes.shape == (batch_size,)
        assert gaps.shape == (batch_size,)
        assert (modes >= 0).all()
        assert (modes < num_modes).all()


class TestDistillation:
    """Test distillation components."""

    @pytest.fixture
    def setup(self):
        """Setup for distillation tests."""
        self.batch_size = 4
        self.num_modes = 6

        self.distill_loss_fn = CVIDistillationLoss(num_modes=self.num_modes)

    def test_distillation_loss_computation(self, setup):
        """Test distillation loss computation."""
        mode_logits = torch.randn(self.batch_size, self.num_modes)
        teacher_modes = torch.randint(0, self.num_modes, (self.batch_size,))
        cvi_advantages = torch.randn(self.batch_size)

        output = self.distill_loss_fn(
            mode_logits=mode_logits,
            teacher_modes=teacher_modes,
            cvi_advantages=cvi_advantages,
        )

        assert "mode_sd_loss" in output
        assert "total_loss" in output
        assert "gate" in output
        assert output["gate"].shape == (self.batch_size,)

    def test_distillation_gate_range(self, setup):
        """Test that distillation gates are in [0, 1]."""
        mode_logits = torch.randn(self.batch_size, self.num_modes)
        teacher_modes = torch.randint(0, self.num_modes, (self.batch_size,))
        cvi_advantages = torch.randn(self.batch_size)

        output = self.distill_loss_fn(
            mode_logits=mode_logits,
            teacher_modes=teacher_modes,
            cvi_advantages=cvi_advantages,
        )

        gates = output["gate"].detach().numpy()
        assert (gates >= 0).all()
        assert (gates <= 1).all()


class TestPrefixMining:
    """Test prefix mining."""

    def test_prefix_mining_basic(self):
        """Test basic prefix mining."""
        miner = PrefixMiner(max_prefixes_per_trajectory=3)

        trajectory = {
            "task_id": "test_task",
            "observations": [{"url": f"url_{i}"} for i in range(10)],
            "actions": ["action_" + str(i) for i in range(9)],
            "success": True,
            "num_tokens": 100,
        }

        prefixes = miner.mine_prefixes(trajectory)

        assert len(prefixes) <= 3
        assert all(p.step_index >= 0 for p in prefixes)
        assert all(p.step_index < 9 for p in prefixes)


class TestBranchGenerator:
    """Test branch generation."""

    def test_branch_generation(self):
        """Test branch generation from prefixes."""
        from tti.cvi_sdar import Prefix

        generator = BranchGenerator(num_modes=6)

        prefix = Prefix(
            trajectory_id="test",
            step_index=0,
            history={},
            budget={},
        )

        trajectory = {
            "actions": ["a1", "a2", "a3"],
            "observations": [{"url": "url1"}, {"url": "url2"}],
            "success": True,
            "num_tokens": 100,
            "tokens_before_prefix": 50,
        }

        branches = generator.generate_branches(prefix, trajectory, sampled_modes=[0, 1, 2])

        assert len(branches) == 3


class TestTrainer:
    """Test CVI-SDAR trainer."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = CVISARConfig(
            state_embedding_dim=256,
            num_modes=6,
            device=torch.device("cpu"),
        )

        trainer = CVISARTrainer(config)

        assert trainer.mode_policy is not None
        assert "v" in trainer.critics
        assert "q" in trainer.critics
        assert "v_ans" in trainer.critics

    def test_training_step(self):
        """Test a single training step."""
        config = CVISARConfig(
            state_embedding_dim=256,
            num_modes=6,
            batch_size=4,
            device=torch.device("cpu"),
        )

        trainer = CVISARTrainer(config)

        batch = {
            "state_embeddings": torch.randn(4, 256),
            "modes": torch.randint(0, 6, (4,)),
            "returns": torch.randn(4),
            "old_log_probs": torch.randn(4),
            "advantages": torch.randn(4),
        }

        losses = trainer.train_step(batch)

        assert "ppo_loss" in losses
        assert "critic_loss" in losses
        assert all(v >= 0 for v in losses.values())


class TestInference:
    """Test inference controller."""

    def test_inference_decision(self):
        """Test CVI inference decision making."""
        config = CVISARConfig(
            state_embedding_dim=256,
            num_modes=6,
            device=torch.device("cpu"),
        )

        trainer = CVISARTrainer(config)

        controller = CVISARInferenceController(
            mode_policy=trainer.mode_policy,
            critics=trainer.critics,
            delta_threshold=0.0,
            device=config.device,
        )

        state_emb = torch.randn(256)
        decision = controller.get_cvi_decision(state_emb)

        assert 0 <= decision.mode < 6
        assert decision.cvi_gaps.shape == (6,)
        assert isinstance(decision.should_continue, bool)
        assert decision.confidence >= 0


# Pytest fixtures
@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for tests."""
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    yield

    # Teardown (if needed)
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
