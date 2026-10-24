# Local RLVR validation — 2026-10-23

The live environment path is exercised on this macOS workspace. Hosted learning is implemented but remains unvalidated because Tinker returned HTTP 402, blocking access on billing status. No hosted optimizer update or learning improvement is claimed.

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

Still required: a funded Tinker account to verify real sampling, a non-degenerate GRPO update, sampler refresh, optimizer-state resume, and checkpoint evaluation. Clean-machine provisioning and broader reset fault/mutation coverage (including banking) remain independent acceptance work. The successful reset soak covers equipment mutation, while gathering runs cover XP changes and depleted trees. Game RNG remains uncontrolled.

See [Live RLVR setup and operation](live-rlvr.md) for commands and contracts.

## Live smoke follow-up — 2026-10-24

- `uv run python -m rsenv.rlvr smoke --manifest configs/rlvr-smoke.json --workers 2` passed in 41.99 seconds with four valid episodes and zero infrastructure errors. Both one-log gathering episodes succeeded; equipment mutation and the zero-reward control passed. All four initial fingerprints matched across distinct worlds, and all four persisted episodes passed verifier replay.
- The earlier run used a temporary Puppeteer cache. Running without that override exposed a missing Chrome installation in the default cache. Installing the pinned Chrome there fixed the exact command above. Browser executable availability is now checked before world creation, with an actionable installation message.
- The focused suite now has 43 passing tests, including a regression check that a missing browser prevents world preparation. Focused lint and whitespace checks pass.
- This smoke exercises the real environment without a model or optimizer. It does not validate hosted learning or exhaustive tool and transport-failure coverage.
