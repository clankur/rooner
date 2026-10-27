# Task calibration — 2026-10-27

The base model succeeded in **30 of 35 valid episodes (85.7%)**. One additional episode was excluded by the verifier. Basic mining was consistently solved; Strength and Defence each succeeded once in three trials. These results justify a more focused combat/recovery evaluation, but do not yet justify a training run: four failures involved token budgets, and the verifier rejection had incomplete evidence.

## Results

| Task | Model successes / valid trials | Excluded trials |
|---|---:|---:|
| Gather one log | 3/3 | 0 |
| Gather logs, retain tool | 3/3 | 0 |
| Equip axe, then gather logs | 2/3 | 0 |
| Mine copper | 3/3 | 0 |
| Mine tin | 3/3 | 0 |
| Mine mixed ores | 3/3 | 0 |
| Reach Attack level 2 | 3/3 | 0 |
| Reach Strength level 2 | 1/3 | 0 |
| Reach Defence level 2 | 1/3 | 0 |
| Equip pickaxe, mine, return | 3/3 | 0 |
| Mine ores in order, return | 2/2 | 1 |
| Equip sword, level Attack, return | 3/3 | 0 |

Family macro success rates were 88.9% woodcutting, 100% mining, 55.6% combat, and 100% sequences among valid trials. The sequence score excludes the rejected trial. Per-task Wilson 95% intervals are wide: approximately 44–100% for 3/3, 21–94% for 2/3, and 6–79% for 1/3. Treat these as diagnostic outcomes rather than reliable rankings.

The baseline took **1,411.70 seconds (23.53 minutes)**. Episode duration p50/p95 was 75.54/289.43 seconds; sampling latency per decision was 2.39/2.85 seconds; game ticks per episode were 92/356. Records contain 38,443 generated tokens and 9,313,783 prompt tokens, including repeated history and the unsent context-limit prompt. These are recorded token counts, not billed usage. There were 19 failed action attempts and one invalid decision; failed attempts can occur inside successful episodes.

## Setup

Twelve tasks cover woodcutting, copper/tin mining, melee leveling, and ordered combinations. Each task ran once with the scripted reference policy and three times with base Qwen/Qwen3.5-4B. Both policies use the same explicit actions and authoritative reward checks. No optimizer ran.

The model used temperature 1, thinking disabled, 512 generated tokens per decision, and a 65,536-token context budget. There were three workers. Each episode owned a fresh world using SDK revision `66ed33320e719e92883d5fbfc05a0b091b54373f`. The game paused between decisions and advanced at normal 600 ms ticks during actions. Sampling seeds derive from task IDs and repeat indices; game RNG is uncontrolled.

```sh
uv run python -m rsenv.rlvr scripted --manifest configs/rlvr-calibration.json --episodes 1 --workers 3 --output outputs/calibration-reference-verified
uv run python -m rsenv.rlvr evaluate --manifest configs/rlvr-calibration.json --episodes 3 --workers 3 --output outputs/calibration-qwen-verified
```

The reference policy completed all twelve tasks in 384.04 seconds, with no infrastructure failures. All twelve persisted reward decisions replayed successfully. This establishes observed feasibility for each fixture, not guaranteed success under every RNG trajectory.

## Trace findings

- In the second `logs-equip-first` trial, the model continued gathering and waiting as HP fell from 10 to 4, 3, 2, then 0. Its final decision was to wait for combat to end. It did not retreat.
- In the first `level-strength` trial, the model selected a duck and repeatedly waited with no XP gain. The episode ended before another sample because its 65,260-token prompt plus the 512-token allowance exceeded the context budget.
- In the second `level-defence` trial, the model spent its fourth response reasoning about combat styles and exhausted 512 output tokens before issuing a complete action. The second Strength trial also exhausted its output allowance.
- In the third Defence trial, the model exhausted its output allowance while incorrectly reinterpreting `xp_tenths`, despite the observation providing the exact next-level threshold. Overall, the four original `generation_limit` outcomes comprise three output truncations and one context-budget stop. Future runs report `context_limit` separately.
- An `ordered-ores` trial was excluded with `unattributed_items`. The original recorder did not retain the rejected action's after-state, so the precise gained item cannot be recovered from the episode record. Inspection of the pinned mining script exposed an unhandled legal outcome: incidental uncut gems without ore XP.

The verifier now allows only the four uncut gems listed by the pinned mining table during observed rock harvesting, bounded by elapsed ticks. Gems do not count toward ore goals or earn ore XP. Other unexplained gains remain invalid. Action evidence is now saved before verification, including transitions that the verifier rejects. These changes were not applied to the running baseline's loaded Python modules.

A separate three-trial `ordered-ores/v2` follow-up with the corrected verifier completed in 77.20 seconds: **two successes, one 512-token output truncation, zero infrastructure failures**. All three rewards replayed. No gem appeared in these follow-up observations, so the rare-drop allowance is validated by targeted tests and pinned game-source inspection rather than a live reproduction. The original rejected episode remains excluded; these follow-ups are not pooled into the baseline table. Artifacts are in `outputs/calibration-mining-followup/`; its `run.json` contains the single-task manifest and configuration.

## Next evaluation

Keep the gathering/mining cases as controls. Prioritize combat style changes, recognizing stalled attacks, choosing reachable targets, and retreating or healing under damage. Add short combinations of these requirements with reference-verified fixtures; see the [candidate task list](task-evaluation.md#expand-from-evidence).

Before attributing all failures to a need for post-training, repeat the affected combat cases with a predeclared larger output allowance and inspect whether the underlying decisions improve. Keep these follow-ups separate from this baseline. Then freeze a held-out set with disjoint starting positions and objective combinations. A later training experiment should compare base and trained policies under identical budgets and report success separately from latency and cost.

## Validation and artifacts

The four-episode smoke passed in 48.48 seconds: equipment mutation, two successful gathers, a zero-reward control, matching reset fingerprints, distinct worlds, and four replayed episodes. All 85 offline tests pass, covering goal ordering, cumulative resource retention, pinned XP thresholds, explicit tool arguments, incidental gems, and retention of rejected evidence. Python lint/format checks, bridge builds, and whitespace checks pass.

All 35 valid baseline rewards and all twelve reference rewards replay under the final verifier. The baseline command exited nonzero because it included one infrastructure-invalid episode; it completed all 36 scheduled trials.

Full local artifacts are under `outputs/calibration-reference-verified/`, `outputs/calibration-qwen-verified/`, and `outputs/calibration-smoke-verified/`. Each run contains task/source metadata and persisted episode evidence. Output directories are ignored by git. Earlier debugging runs are excluded from these results: they exposed a bad mining spawn, a login-initialization error, and an ordered-goal verifier that unnecessarily required another action when the destination was already reached.

The calibration set informed these findings and is not a held-out test. Three trials per task give wide uncertainty intervals; model latency, RNG, token budgets, and decision failures should be inspected separately. Billed cost was not measured.
