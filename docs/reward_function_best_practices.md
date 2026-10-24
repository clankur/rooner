# Reward and GRPO weighting

The runner uses task-specific binary rewards and group-normalized advantages.

## Live task reward

[`verify_harvest`](../rsenv/rlvr/tasks.py) awards success when an episode harvests the requested number of normal logs and retains them in inventory. Each credited action must identify a tree from the preceding observation, add logs, and add 250 engine XP tenths per log to woodcutting.

The verifier checks the reset receipt, episode/world identity, observation order, and continuity between actions. Unattributed inventory or XP changes, bank mutations, and pending actions invalidate the evidence.

- Verified success: reward `1.0`.
- Valid episode ending before success: reward `0.0`, including a final answer or an exhausted budget.
- Infrastructure failure or invalid evidence: reward `None`; training rejects the group.

Partial progress is reported to the policy but does not contribute reward. Tool syntax alone earns no reward.

## Group normalization and token weighting

[`build_examples`](../rsenv/rlvr/learning.py) requires a group of `G >= 2` episodes with the same task, starting-state fingerprint, and policy version. For rewards `r_i`, it computes:

```text
mean_reward = sum(r_i) / G
reward_variance = sum((r_i - mean_reward)^2) / G
advantage_i = (r_i - mean_reward) / (sqrt(reward_variance) + 1e-8)
per_token_advantage_i = advantage_i / (generated_tokens_i * G)
```

Each turn uses its recorded prompt and generated token IDs. Inputs and targets are shifted by one token. Prompt targets receive zero advantage; generated targets receive `per_token_advantage_i` and their sampled behavior log-probabilities. Earlier assistant turns appearing in a later prompt are masked there.

Tinker's PPO loss sums token contributions with ratio clipping at `[0.8, 1.2]`. Dividing by each episode's generated-token count and the group size gives each episode equal weight. The adapter sends these weights directly; it does not apply another mean reduction.

Zero-variance groups skip optimization. The runner stops after twenty consecutive such groups. Tests check prompt masking and total episode weight for different trajectory lengths and group sizes. Hosted optimizer behavior still requires service validation.
