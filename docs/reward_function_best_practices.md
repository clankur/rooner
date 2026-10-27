# Reward and GRPO weighting

The runner uses task-specific binary rewards and group-normalized advantages.

## Live task reward

`verify_task` evaluates acquisition, skill, equipment, and location goals over authoritative evidence. Goals may be unordered or sequential, with item-preservation and final-health constraints. See [task evaluation](task-evaluation.md) for task authoring and timing.

The verifier checks episode/world identity, reset receipts, paused decision boundaries, and continuity. Harvesting requires matched item and XP changes from an observed target. Combat permits its companion Hitpoints XP. Infrastructure failures receive `None`; valid failures and exhausted gameplay budgets receive `0.0`; verified completion receives `1.0`. Progress is reported to the policy but is not a shaped reward.

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

Zero-variance groups skip optimization. The runner stops after twenty consecutive such groups. Tests check prompt masking and total episode weight for different trajectory lengths and group sizes. A [hosted forward check](live-rlvr-validation.md) reproduced Tinker's PPO loss from generated-token probabilities and these weights.
