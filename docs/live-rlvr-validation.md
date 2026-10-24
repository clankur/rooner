# Local RLVR validation — 2026-10-23

The live environment path is exercised on this macOS workspace. Hosted validation was initially blocked by Tinker billing. The 2026-10-24 hosted experiment below exercised real sampling and optimizer updates. No learning improvement is claimed.

## Passed checks

- 42 offline tests: reset/verifier invariants, adversarial evidence, action parsing, protocol correlation, timeouts, token alignment and masks, group rejection, checkpoint failure recovery, SDK payloads, finite-loss guards, and partial startup cleanup.
- Focused Ruff check, Python compilation, TypeScript bridge bundle, and whitespace checks.
- Actual Qwen/Qwen3.5-4B tokenizer with Transformers 5.9.0 rendered a multi-turn tool conversation successfully. This caught and fixed the tokenizer's dictionary-return default.
- 100 live reset–equipment-mutation cycles across four workers: 100 valid episodes, zero infrastructure failures, one canonical initial-state fingerprint. Each probe equipped the axe and then ended intentionally without satisfying the gathering goal. Reset p50/p95/p99: 7.65/8.89/8.95 seconds.
- Two initial five-log episodes plus twelve profiling episodes completed with authoritative item and woodcutting-XP evidence. All twelve profiling episodes passed verifier replay.
- All 100 reset-probe records passed verifier replay.
- A policy that immediately claimed completion in a live world received reward zero.
- SIGTERM during a two-worker live collection cleaned up all 24 identified owned processes in 3.47 seconds; no identified owned processes remained.
- Tinker setup stopped after 30.9 seconds on the billing pause, instead of waiting indefinitely.

## Scripted five-log throughput

Each row contains four episodes, all successful with zero infrastructure failures. Runs used fresh worlds, 600 ms game ticks, XP multiplier 1, and no hosted model calls.

| Workers | Wall seconds | Successful episodes/hour | Reset p95 seconds | Observed seconds/tick p95 |
|---|---:|---:|---:|---:|
| 1 | 296.46 | 48.57 | 6.80 | 0.6012 |
| 2 | 174.77 | 82.39 | 7.48 | 0.6014 |
| 4 | 104.72 | 137.51 | 9.07 | 0.6012 |

These are small, stochastic local measurements. They demonstrate functioning concurrency, not policy learning or precise long-run capacity. Each row's Wilson 95% success interval is approximately [0.51, 1.00]. Memory observations cover the engine process; browser/gateway resource accounting is not included.

## Artifacts and remaining gates

Machine-readable evidence is under `outputs/rlvr-validation/`: `validation.json`, `reset-100/`, `profile/`, `failing-control/`, `cancellation/`, and `tinker-preflight.json`. Output directories are intentionally ignored by git. Episode JSONL includes immutable sampled-token records and authoritative before/after action states; run manifests identify the source and configuration used by each run.

At this stage, billing blocked real sampling, a non-degenerate GRPO update, sampler refresh, optimizer-state resume, and checkpoint evaluation. The later hosted experiment below covers sampling, updates, and sampler refresh. Clean-machine provisioning and broader reset fault/mutation coverage (including banking) remain independent acceptance work. The successful reset soak covers equipment mutation, while gathering runs cover XP changes and depleted trees. Game RNG remains uncontrolled.

See [Live RLVR setup and operation](live-rlvr.md) for commands and contracts.

## Live smoke follow-up — 2026-10-24

- `uv run python -m rsenv.rlvr smoke --manifest configs/rlvr-smoke.json --workers 2` passed in 41.99 seconds with four valid episodes and zero infrastructure errors. Both one-log gathering episodes succeeded; equipment mutation and the zero-reward control passed. All four initial fingerprints matched across distinct worlds, and all four persisted episodes passed verifier replay.
- The earlier run used a temporary Puppeteer cache. Running without that override exposed a missing Chrome installation in the default cache. Installing the pinned Chrome there fixed the exact command above. Browser executable availability is now checked before world creation, with an actionable installation message.
- The focused suite now has 43 passing tests, including a regression check that a missing browser prevents world preparation. Focused lint and whitespace checks pass.
- This smoke exercises the real environment without a model or optimizer. It does not validate hosted learning or exhaustive tool and transport-failure coverage.

## Single-flow validation — 2026-10-24

- `uv sync --locked` installed the project environment without Torch, Hydra, runq, or wandb.
- `uv run --locked python -m pytest rsenv/tests -q`: 51 passed, covering the live verifier, parser, transport, GRPO weighting, and Tinker adapter.
- The four-episode smoke passed in **42.97 seconds** with two successful gathers, the expected zero-reward controls, matching reset fingerprints, distinct worlds, four replayed episodes, and no infrastructure failures. Artifacts are in `outputs/pr9-live-only-smoke/`.
- The pinned browser dependencies installed with `bun install --frozen-lockfile`. Python lint/format checks, bridge bundling, setup-script syntax, and whitespace checks passed.
- This run reused the installed pinned game SDK. Clean-machine provisioning and hosted learning remain unvalidated.

## Hosted experiment — 2026-10-24

```sh
uv run --locked python -m rsenv.rlvr train --manifest configs/rlvr-smoke.json --workers 2 --group-size 4 --iterations 2 --output outputs/tinker-small-20261024-fixed
uv run --locked python -m rsenv.rlvr replay --manifest configs/rlvr-smoke.json --records outputs/tinker-small-20261024-fixed/episodes.jsonl
```

The first attempt exposed a missing Jinja2 runtime dependency and an outdated `ServiceClient.close()` call. Jinja2 is now a direct locked dependency, and cleanup passes the session outcome required by Tinker 0.35.0. Regression tests exercise actual Transformers template rendering and validate cleanup against the installed SDK signature. All 55 tests pass; focused lint, formatting, and whitespace checks pass.

The corrected run used Qwen/Qwen3.5-4B, seed 42, LoRA rank 16, and learning rate `1e-5`:

- Eight valid episodes in eight separate worlds, one initial-state fingerprint, and zero infrastructure failures.
- Five successful gathers; three episodes exhausted the six-turn budget. Rewards were `[1, 0, 1, 1]` and `[1, 1, 0, 0]` in completion order.
- All 33 sampled decisions parsed as supported actions; 2,589 generated tokens.
- Two optimizer updates with finite losses (`0.00057916`, `0.00050427`), saved optimizer state, and refreshed sampler weights. The second batch used the first update's sampler.
- Replay verified all eight rewards and both training-conversion hashes. The CLI exited successfully, no pending update remained, and no owned game or training processes remained.

A forward-only diagnostic restored the initial optimizer checkpoint and recomputed the first batch. Across 954 generated tokens, the mean trainer/sampler probability ratio was `0.999406`, mean absolute log-ratio `0.011758`, and clipped fraction `0.006289`. The independently computed masked PPO loss (`0.0005791544`) matched the service loss (`0.0005791485`) within `6e-9`, using the [documented PPO objective](https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/). The service's aggregate KL/clipping diagnostics were much larger; this comparison indicates they include prompt positions with placeholder behavior probabilities. They should not be interpreted as generated-token policy drift.

Artifacts are in `outputs/tinker-small-20261024-fixed/`: `experiment-summary.json`, `probability-check.json`, `check_probabilities.py`, `checkpoint.json`, and `episodes.jsonl`. The earlier failed attempt is preserved in `outputs/tinker-small-20261024/`.

This validates hosted sampling, nonzero-signal updates, sampler refresh, checkpoint saving, and checkpoint restoration for a forward pass. It does not establish learning improvement: batch success rates were 75% and 50%, with only four episodes each. Full training resume, held-out checkpoint evaluation, and billed cost measurement remain unchecked. The live SDK was copied from the existing pinned installation; this was not a clean-machine provisioning test.
