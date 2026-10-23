# Reward Function Best Practices (General + GRPO)

A reference compiled for the pooner GRPO RuneScape agent. Organized from general
RL reward design → GRPO-specific concerns → how it maps to pooner.

---

## Part 1 — General reward function best practices

Foundational principles that apply to any RL setup.

1. **Reward the outcome you actually want, not a proxy.** The single biggest
   failure mode is *specification gaming* — the agent maximizes your written
   reward while violating your intent. Every shaped term is a potential exploit.

2. **Prefer verifiable/programmatic rewards over learned ones where possible.**
   A deterministic checker (did the bot reach the bank? did XP increase? did the
   item count go up?) can't be gamed the way a learned reward model can, and
   removes a whole class of reward-model drift.

3. **Use potential-based reward shaping (PBRS) when you add intermediate
   signals.** Ng, Harada & Russell (1999) proved that shaping of the form
   `F(s,s') = γΦ(s') − Φ(s)` is *policy-invariant* — it speeds learning without
   changing the optimal policy. This is the theoretically safe way to densify a
   sparse reward. Any non-potential-based bonus risks creating a new optimum
   (e.g., the agent loops to farm the bonus).

4. **Match density to the task.** Sparse 0/1 terminal rewards are hard to learn
   from but hard to hack; dense shaping accelerates learning but invites hacking.
   The standard resolution: keep the *true* objective as a sparse terminal
   reward, add shaping only as policy-invariant guidance.

5. **Normalize / bound rewards.** Unbounded or wildly-scaled rewards destabilize
   advantage estimates. (pooner already does this — min-max clamp to [0,1].)

6. **Curriculum as an alternative to shaping.** Instead of hand-crafting
   intermediate rewards, start with easier task variants and ramp difficulty.
   Often cleaner than shaping for agentic tasks.

7. **Audit for reward hacking continuously.** Hold out diverse, realistic evals
   (including full agentic rollouts, not just the metric you optimize).
   Penalizing detected hacking during training cut misaligned generalization by
   >75% in Anthropic's 2025 work.

### References

- [Ng, Harada, Russell (1999) — Policy Invariance Under Reward Transformations (potential-based shaping, the canonical paper)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
- [Reward shaping notes — *Mastering Reinforcement Learning*](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)
- [The Sparse Reward Problem: Shape Signals Without Cheating](https://medium.com/@bhagyarana80/the-sparse-reward-problem-shape-signals-without-cheating-faa1962cf339)
- [Shaping Sparse Rewards in RL: A Semi-supervised Approach (arXiv 2501.19128)](https://arxiv.org/html/2501.19128v1)
- [Anthropic — From shortcuts to sabotage: emergent misalignment from reward hacking](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)
- [Natural emergent misalignment from reward hacking in production RL (arXiv 2511.18397)](https://arxiv.org/html/2511.18397v1)

---

## Part 2 — GRPO-specific reward best practices

GRPO drops PPO's critic and computes advantages by *group-relative
normalization*: sample G completions per prompt, then
`Â_i = (r_i − mean(r)) / std(r)`. This changes what matters in reward design.

1. **Rewards only matter *relative to the group*, not in absolute scale.**
   Because advantages are mean-centered per prompt, an affine shift of your
   reward is invisible to GRPO. What matters is the *spread* of rewards across
   the G samples for one prompt.

2. **Guarantee intra-group reward variance.** If all G completions get the same
   reward (all fail, or all succeed), `std = 0` → zero advantage → zero gradient
   for that prompt. Sparse all-or-nothing rewards on hard prompts produce dead
   batches. Mitigations:
   - Tune prompt difficulty so some samples succeed and some fail (this is where
     curriculum pays off most).
   - Add partial-credit gradations so completions differentiate.
   - GRPO-λ dynamically switches to length-agnostic 0/1 rewards when the group's
     correctness ratio is low, and re-enables efficiency penalties when high.

3. **Beware the std-normalization length bias — consider Dr. GRPO.** The
   "Understanding R1-Zero-Like Training" paper showed GRPO's length + std
   normalization terms systematically inflate response length, *especially for
   wrong answers*. **Dr. GRPO** removes those normalization terms for an unbiased
   optimizer and better token efficiency. Worth knowing if you add any
   length-related reward term, since the optimizer can interact with it.

4. **Verifiable rewards (RLVR) are the GRPO sweet spot.** DeepSeek-R1-Zero used
   pure verifiable rewards (correctness + format) with GRPO — no learned reward
   model — precisely because verifiable signals are hard to game at scale. For a
   game env, environment-derived signals (XP delta, item delta, task completion,
   valid tool-call format) are exactly this.

5. **Separate and weight reward components explicitly.** The common RLVR recipe
   is `accuracy reward + format reward` (e.g., did it emit a parseable tool call
   / correct structure). Keep them as named components so you can schedule/anneal
   them — format reward matters early, accuracy later.

6. **Reward variance is itself a tunable lever.** Recent work (GRPOVI) shows
   *deliberately increasing* reward variance improves GRPO sample efficiency —
   the opposite intuition from "smooth out rewards." Group-relative methods are
   starved by low variance.

### References

- [DeepSeekMath — original GRPO paper (arXiv 2402.03300)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 — GRPO + verifiable rewards at scale (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948)
- [Understanding R1-Zero-Like Training / **Dr. GRPO** (arXiv 2503.20783)](https://arxiv.org/abs/2503.20783) · [code](https://github.com/sail-sg/understand-r1-zero)
- [Cameron Wolfe — Group Relative Policy Optimization (GRPO) deep dive](https://cameronrwolfe.substack.com/p/grpo)
- [Sebastian Raschka — The State of RL for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [Unsloth — Reinforcement Learning (RL) Guide (practical GRPO reward-function tips)](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)
- [Stable RL for Efficient Reasoning / GRPO-λ (arXiv 2505.18086)](https://arxiv.org/pdf/2505.18086)
- [Accelerating RLHF with Reward Variance Increase / GRPOVI (arXiv 2505.23247)](https://arxiv.org/pdf/2505.23247)
- [RLVR: GRPO's Effective Loss, Dynamics, and Success Amplification (arXiv 2503.06639)](https://arxiv.org/html/2503.06639v1)
- [The Good, The Bad, and The Hybrid: Reward Structure Showdown in Reasoning Training (arXiv 2511.13016)](https://arxiv.org/pdf/2511.13016)

---

## How this maps to pooner specifically

Given the plan-then-execute RuneScape agent:

- **Lean on verifiable game-state rewards** (XP/item/level deltas, task
  completion, valid tool-call parse) rather than a learned reward model — the
  simulator already gives ground truth.
- **The [0,1] min-max normalization** is good for stability, but GRPO
  mean-centers per group anyway — the *clamp* matters more for taming outliers
  than the absolute range. Watch that clamping doesn't collapse intra-group
  variance and starve gradients.
- **Difficulty/curriculum is the highest-leverage knob** for keeping group
  reward variance alive — sample scenarios where some of the G rollouts succeed
  and some fail.
- If a **length or step-count penalty** is ever added (tempting for "do it in
  fewer actions"), check the Dr. GRPO length-bias interaction first.
- Keep **format reward separate** from task reward so malformed tool calls are
  penalized distinctly from bad-but-valid play.
