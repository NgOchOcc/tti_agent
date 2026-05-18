# TOS-RL + TTI Integration Checklist

A step-by-step checklist for integrating TOS-RL training with your existing TTI codebase.

## Phase 1: Preparation

### Code Review
- [ ] Read `TOSRL_INTEGRATION_WITH_TTI.md` completely
- [ ] Review `tti/tos_rl/README.md` API reference
- [ ] Understand the 7 core TOS-RL modules:
  - [ ] `objectives.py` - Cost-aware utility computation
  - [ ] `optimization.py` - GRPO loss and advantages
  - [ ] `branching.py` - Prefix-level branching
  - [ ] `training.py` - Training orchestrator
  - [ ] `inference.py` - Inference algorithm
  - [ ] `utils.py` - Budget, costs, formatting utilities
  - [ ] `__init__.py` - Package exports

### Environment Setup
- [ ] Ensure PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
- [ ] Create virtual environment if needed: `python -m venv venv_tosrl`
- [ ] Install dependencies (PyTorch, NumPy, etc.)
- [ ] Verify TOS-RL modules can be imported:
  ```bash
  python -c "from tti.tos_rl import TOSRLTrainer, TOSRLConfig; print('✓ TOS-RL imported')"
  ```

### Documentation
- [ ] Review `TOSRL_IMPLEMENTATION_GUIDE.md`
- [ ] Review `TOSRL_IMPLEMENTATION_COMPLETE.md`
- [ ] Review `tti/tos_rl/README.md`
- [ ] Review `TOS_RL_FINAL_SUMMARY.md`
- [ ] Read `scripts/train_tos_rl_example.py` as reference

## Phase 2: Integration - Trajectory Collection

### Understand Current TTI Collection
- [ ] Locate your trajectory collection code in TTI
- [ ] Identify the agent execution loop
- [ ] Find where trajectories are stored/logged
- [ ] Understand what metrics are currently tracked

### Modify Trajectory Collection
- [ ] Add mode token tracking to LLM output:
  ```python
  from tti.tos_rl import TokenModeFormatter
  mode, content = TokenModeFormatter.parse_mode_and_content(llm_output)
  ```

- [ ] Track the following per trajectory:
  - [ ] `success`: int (1 if task succeeded, 0 otherwise)
  - [ ] `num_steps`: int (number of OBSERVE actions)
  - [ ] `num_tokens`: int (total tokens generated)
  - [ ] `num_loops`: int (repeated actions/observations)
  - [ ] `num_bad_actions`: int (invalid actions)
  - [ ] `modes`: list of str (["THINK", "OBSERVE", "OBSERVE", "ANSWER"])

- [ ] Update LLM prompting to enforce mode tokens:
  ```python
  prompt = f"""Task: {task_description}

  History:
  {history}

  Respond with one of: [THINK] for reasoning, [OBSERVE] for action, [ANSWER] to stop.

  Your response:"""
  ```

- [ ] Include budget information in prompts:
  ```python
  from tti.tos_rl.utils import create_budget_prompt_suffix
  budget_text = create_budget_prompt_suffix(budget)
  ```

### Test Collection
- [ ] Run trajectory collection on a few WebArena tasks
- [ ] Verify mode tokens are being parsed correctly:
  ```bash
  python -c "
  from tti.tos_rl import TokenModeFormatter
  output = '[OBSERVE] Click the button'
  mode, content = TokenModeFormatter.parse_mode_and_content(output)
  print(f'Mode: {mode}, Content: {content}')
  "
  ```
- [ ] Verify trajectory dict format is correct
- [ ] Log sample trajectories to verify quality

## Phase 3: Integration - Training

### Configure TOS-RL
- [ ] Create `TOSRLConfig` with your hyperparameters:
  ```python
  from tti.tos_rl import TOSRLConfig
  config = TOSRLConfig(
      group_size=4,              # K trajectories per task
      max_prefixes_per_task=3,
      branches_per_prefix=3,
      learning_rate=1e-5,
      mode_token_weight=2.0,     # Upweight mode tokens
      use_branching=False,       # Start without branching
  )
  ```
- [ ] Set reasonable hyperparameter values for your setup
- [ ] Document your config choices

### Implement Batch Preparation
- [ ] Create function to group K trajectories by task_id
- [ ] Implement log probability extraction from model:
  ```python
  def get_log_probs_from_trajectories(trajectories, model):
      # Return log_probs for each trajectory
      pass
  ```
- [ ] Implement mode log probability extraction
- [ ] Implement mode probability extraction (for entropy term)
- [ ] Test batch preparation on sample trajectories

### Integrate Training Step
- [ ] Create `TOSRLTrainer` instance with config
- [ ] In your training loop, add TOS-RL training:
  ```python
  from tti.tos_rl import TOSRLTrainer

  trainer = TOSRLTrainer(config)

  # For each group of K trajectories
  utilities, costs = trainer.process_trajectories(group)
  advantages = trainer.compute_group_advantages(utilities)
  batch = prepare_batch(utilities, advantages)
  metrics = trainer.training_step(batch)
  ```
- [ ] Add metrics logging
- [ ] Verify loss is decreasing over training steps

### Test Training
- [ ] Run training on small dataset (e.g., 10 tasks, 2 epochs)
- [ ] Verify no errors occur
- [ ] Check loss values are reasonable
- [ ] Save/load checkpoints
- [ ] Monitor mode distribution (all three modes being used?)

## Phase 4: Integration - Inference

### Implement Inference Controller
- [ ] Import `TOSRLInference` and `BudgetState`:
  ```python
  from tti.tos_rl import TOSRLInference, BudgetState
  ```

- [ ] Create inference wrapper:
  ```python
  inference = TOSRLInference(max_steps=30, max_tokens=4096)

  result = inference.run_episode(
      task="Find the price",
      llm_fn=model.generate,
      initial_budget=BudgetState(
          env_interactions=30,
          tokens=4096
      )
  )
  ```

- [ ] Handle different cost preferences:
  ```python
  # High efficiency
  budget_efficient = BudgetState(env_interactions=10, tokens=2000)

  # Balanced
  budget_balanced = BudgetState(env_interactions=30, tokens=4096)

  # High success
  budget_generous = BudgetState(env_interactions=50, tokens=8192)
  ```

### Test Inference
- [ ] Run inference on validation tasks
- [ ] Verify mode tokens are parsed correctly
- [ ] Check budget is being respected
- [ ] Monitor success rate
- [ ] Monitor average steps/tokens used

## Phase 5: Evaluation

### Baseline Comparisons
- [ ] Collect base agent performance (no RL)
- [ ] Collect TTI baseline performance (fixed horizon)
- [ ] Record TOS-RL performance (your implementation)
- [ ] Compare on metrics:
  - [ ] Success rate
  - [ ] Average steps per task
  - [ ] Average tokens per task
  - [ ] Loop rate (% repeated actions)
  - [ ] Invalid action rate

### Metrics to Track
- [ ] **Success-Cost Pareto Curve**: Plot success rate vs average steps at different cost settings
- [ ] **Mode Distribution**: % of THINK vs OBSERVE vs ANSWER by task difficulty
- [ ] **Training Stability**: Loss curves, advantage distribution
- [ ] **Generalization**: Transfer from WebArena to WebVoyager

### Analysis
- [ ] Hypothesis 1: Same success at lower cost?
  - [ ] Calculate success-cost AUC
  - [ ] Compare with baselines

- [ ] Hypothesis 2: Learned stopping behavior?
  - [ ] Analyze easy vs hard tasks
  - [ ] Check mode distribution differences

- [ ] Hypothesis 3: No mode collapse?
  - [ ] Verify all three modes are used
  - [ ] Check mode distribution across tasks

## Phase 6: Optimization

### Enable Prefix Branching (Optional)
- [ ] Once basic training works, enable branching:
  ```python
  config.use_branching = True
  config.max_prefixes_per_task = 3
  config.branches_per_prefix = 3
  ```

- [ ] Run training with branching enabled
- [ ] Compare results with/without branching
- [ ] If it helps, keep enabled; otherwise disable

### Hyperparameter Tuning
- [ ] Experiment with different K values (group_size):
  - [ ] K=2 (faster, noisier)
  - [ ] K=4 (balanced)
  - [ ] K=8 (slower, smoother)

- [ ] Tune mode_token_weight:
  - [ ] Lower (1.0): slower mode learning
  - [ ] Higher (3.0): faster mode learning

- [ ] Adjust cost coefficient ranges:
  ```python
  config.lambda_env_range = (0.001, 0.1)      # Cost per step
  config.lambda_tok_range = (0.0001, 0.01)    # Cost per token
  ```

### Performance Tuning
- [ ] Profile training to find bottlenecks
- [ ] Enable batch processing where possible
- [ ] Use GPU if available
- [ ] Implement gradient accumulation if needed

## Phase 7: Documentation

### Code Documentation
- [ ] Add docstrings to integration code
- [ ] Document your trajectory collection modifications
- [ ] Document your training loop integration
- [ ] Document inference deployment

### Results Documentation
- [ ] Create results table comparing baselines
- [ ] Plot success-cost Pareto curves
- [ ] Analyze mode distributions by task type
- [ ] Write summary of findings

### Integration Guide
- [ ] Document any custom modifications
- [ ] Create deployment instructions
- [ ] Document troubleshooting steps
- [ ] List known limitations

## Phase 8: Production Deployment

### Checkpoint Management
- [ ] Implement checkpoint saving:
  ```python
  torch.save({
      'model': model.state_dict(),
      'config': config,
      'epoch': epoch,
  }, checkpoint_path)
  ```

- [ ] Implement checkpoint loading
- [ ] Test checkpoint restoration

### Monitoring
- [ ] Set up metrics logging (TensorBoard, WandB, etc.)
- [ ] Monitor key metrics in real-time:
  - [ ] Success rate
  - [ ] Average steps
  - [ ] Mode distribution
  - [ ] Loss curves

- [ ] Set up alerts for anomalies

### Production Readiness
- [ ] Code review of integration
- [ ] Full test suite passes
- [ ] Documentation is complete
- [ ] Performance is acceptable
- [ ] Resource usage is reasonable

## Optional Enhancements

### Curriculum Learning
- [ ] Start with high success, low efficiency requirement
- [ ] Gradually increase efficiency requirement
- [ ] Tracks mode evolution over curriculum

### Multi-Task Learning
- [ ] Train single policy on multiple task types
- [ ] Include task category in prompt
- [ ] Analyze task-specific mode preferences

### Robustness Testing
- [ ] Test on out-of-distribution tasks
- [ ] Test with different LLM models
- [ ] Test with perturbed environment
- [ ] Measure transfer to WebVoyager

## Troubleshooting Guide

### Common Issues

**Issue**: Mode tokens not being parsed
- **Check**: Ensure LLM output starts with `[THINK]`, `[OBSERVE]`, or `[ANSWER]`
- **Fix**: Adjust prompting to enforce format

**Issue**: Always choosing one mode (mode collapse)
- **Check**: Cost coefficients too low?
- **Fix**: Increase `lambda_env`, `lambda_tok` ranges

**Issue**: Always stopping early
- **Check**: Cost coefficients too high?
- **Fix**: Decrease coefficients or sample wider range

**Issue**: No improvement from training
- **Check**: Are K trajectories grouped correctly?
- **Fix**: Verify group_size and task_id matching

**Issue**: Memory out of memory errors
- **Check**: Is batch_size too large?
- **Fix**: Reduce group_size or max_prefixes_per_task

**Issue**: Poor success rate
- **Check**: Is model initialized from good checkpoint?
- **Fix**: Start from supervised baseline, not random

## Success Criteria

You'll know integration is successful when:

- ✅ Trajectories are collected with proper mode token tracking
- ✅ Training loop runs without errors
- ✅ Loss decreases over training
- ✅ Success rate improves over baseline
- ✅ Mode distribution is diverse (all three modes used)
- ✅ Inference produces reasonable results
- ✅ Cost-aware behavior is learned
- ✅ Transfer to new tasks works

## Next Steps After Integration

1. **Ablation Studies**:
   - Train without mode emphasis
   - Train without prefix branching
   - Train with different K values
   - Train with different cost ranges

2. **Extended Evaluation**:
   - Run on full WebArena test set
   - Transfer to WebVoyager
   - Compare with CVI-SDAR
   - Compare with other baselines

3. **Publication**:
   - Prepare paper with results
   - Create comparison figures
   - Document novelty over CVI-SDAR
   - Release code and data

## Resources

- **Integration Guide**: `TOSRL_INTEGRATION_WITH_TTI.md`
- **Integration Example**: `scripts/train_tosrl_with_tti_integration.py`
- **TOS-RL API**: `tti/tos_rl/README.md`
- **Implementation Details**: `TOSRL_IMPLEMENTATION_GUIDE.md`
- **Example Training**: `scripts/train_tos_rl_example.py`

## Questions?

Refer to:
1. `tti/tos_rl/README.md` for API questions
2. `TOSRL_IMPLEMENTATION_GUIDE.md` for implementation questions
3. `TOSRL_INTEGRATION_WITH_TTI.md` for integration questions
4. Code comments in `tti/tos_rl/` for implementation details

---

**Status**: Ready for integration
**Last Updated**: 2026-05-18
**Version**: 1.0.0
