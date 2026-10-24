# Live RuneScape RLVR

The `rsenv.rlvr` runner uses isolated real 2004scape worlds through `VerifiedBridgeClient`, a `BridgeClient` subclass. Task specifications provide training goals, starting fixtures, budgets, and reward verifiers. The initial goal is to harvest and retain five normal logs. Mining requires its own action/evidence implementation before it can be added.

## Runtime

Use Python 3.11+, Bun **1.3.14**, rs-sdk revision **66ed33320e719e92883d5fbfc05a0b091b54373f**, Puppeteer **24.43.1**, and Tinker **0.35.0**. The lightweight Python path avoids local torch and runq:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-rlvr.txt
bash rsenv/bridge/rlvr/setup.sh
.venv/bin/python -m rsenv.rlvr doctor
```

The pinned setup script installs `rsenv/bridge/rs-sdk`, dependencies, packed content, the bot webclient, and Chrome. It refuses an existing checkout at a different revision. Clean-machine execution remains an acceptance check. The doctor checks the pinned revision, required build artifacts, and Puppeteer's configured browser executable. It is a preflight check, not proof that a browser can log in. Existing `rsenv/bridge/setup.sh` is a legacy floating-revision installer. If using a custom `PUPPETEER_CACHE_DIR`, keep that variable set for subsequent runs. To install a missing browser in the default cache, run `(cd rsenv/bridge && bun x puppeteer browsers install chrome)` without a cache override.

Each episode starts a fresh engine, gateway, browser, and player save directory. Shared assets are read-only symlinks; engine source patches are applied to private copies. Normal server files and accounts are not modified. Ports are assigned per world. Logs and fixtures remain under the run's `worlds/` directory after processes stop.

## Validation and collection

Run from the repository root. These commands start their own servers; no shared server is needed.

For a small MacBook-friendly **real environment** smoke test:

```sh
.venv/bin/python -m rsenv.rlvr smoke --manifest configs/rlvr-smoke.json --workers 2
```

This runs four cold-start episodes: equip an axe, gather one log in each of two independent worlds, and end without gathering as a zero-reward control. It asserts the expected outcomes, identical initial fingerprints, distinct world IDs, equipment mutation, and reward replay from saved records. The production rollout closes each browser/server lease. `smoke.json` records phase latency and pass/fail; any failed assertion exits nonzero. Use `--workers 1` to reduce concurrent memory use. The default task is the same one-log fixture when `--manifest` is omitted; `--episodes` does not change this fixed four-episode suite. No model download, GPU, training, or Tinker account is needed.

The smoke checks the real engine, browser login, bridge, action parsing/execution, observations, resets, reward verification, and recording/replay. It does not cover every tool or injected transport failure, independently prove process-leak absence, or test model token masks, hosted optimization, and checkpoint/resume. The focused unit suite covers failure contracts separately. Gameplay remains stochastic despite fixed starting state; a gameplay timeout fails this smoke rather than being silently retried.

```sh
.venv/bin/python -m pytest rsenv/tests/rlvr -q
.venv/bin/python -m rsenv.rlvr scripted --episodes 4 --workers 2 --output outputs/scripted
.venv/bin/python -m rsenv.rlvr scripted --episodes 2 --fail-policy --output outputs/failing-control
.venv/bin/python -m rsenv.rlvr profile --episodes 4 --output outputs/profile
.venv/bin/python -m rsenv.rlvr reset-check --episodes 100 --workers 4 --output outputs/resets
.venv/bin/python -m rsenv.rlvr replay --records outputs/scripted/episodes.jsonl
```

`reset-check` collects fresh episodes, equips the axe, verifies that mutation completed, and checks initial fingerprints. These probes intentionally end without completing the gathering goal. It does not by itself exercise every possible XP/bank mutation or prove engine-wide deterministic replay; scripted gathering runs additionally exercise XP and depleted trees. `profile` measures 1/2/4 workers. Inspect both valid and infrastructure-invalid episode counts; success rates exclude invalid episodes and include Wilson intervals. Costs are explicitly unknown unless billing data is supplied; token counts are recorded.

## Hosted sampling and GRPO

Set `TINKER_API_KEY` in the environment. The account must have billing access.

```sh
.venv/bin/python -m rsenv.rlvr evaluate --episodes 4 --workers 2 --output outputs/base
.venv/bin/python -m rsenv.rlvr train --manifest configs/rlvr-train.json --workers 2 --group-size 4 --iterations 25 --output outputs/train
.venv/bin/python -m rsenv.rlvr train --resume outputs/train/checkpoint.json --iterations 25
.venv/bin/python eval.py --checkpoint 'tinker://SAMPLER_PATH' --manifest configs/rlvr-heldout.json --episodes 4 --output outputs/heldout
```

The default model is `Qwen/Qwen3.5-4B`, LoRA rank 16, learning rate 1e-5, PPO clipping [0.8, 1.2], temperature 1, no top-k/top-p filtering, no thinking, and 512 generated tokens per turn. No KL term or dense reward is used. Support is checked against the account's service capabilities. Hosted sampling and optimization still require a successful live service validation; unit tests are not that validation.

Every group shares a task, reset fingerprint, and policy checkpoint. Any infrastructure-invalid member rejects the entire group, with at most three full collection attempts. Malformed tools, final answers before success, and exhausted gameplay budgets remain ordinary zero-reward outcomes. Zero-variance groups skip optimization; twenty consecutive such groups stop training. Do not fabricate reward variation to force an update.

## Evidence and recovery

The verifier requires newly acquired canonical log IDs, matching woodcutting XP, a real tree target, fresh correlated observations, and retained inventory. It rechecks the reset receipt. Reset administration and the private authoritative state endpoint are unavailable to policy tools. Each completed action is cancelled before the next model decision. Known unsuccessful actions return failure; uncertain side effects poison the episode.

Task seeds and sampling seeds are recorded. Gameplay RNG is uncontrolled: exact initial fixtures do not imply deterministic trajectories. The held-out manifest varies quantity only; it is a small evaluation probe, not evidence of broad task generalization.

Samples are recorded before execution, and transitions/episodes are appended with fsync. Training uses each turn's original prompt tokens, generated tokens, and behavior log-probabilities. Prompt/observation positions have zero advantage; trajectory weights normalize for generated length. Earlier assistant messages occurring in later prompts are masked there.

The advantage normalization accounts for Tinker's summed token loss; see its [PPO loss contract](https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/). Offline replay checks recorded training conversion hashes as well as reward decisions.

`checkpoint.json` records the committed optimizer checkpoint separately from the sampler checkpoint. `pending-update.json` identifies an uncertain remote branch. Resume forks from the last committed optimizer state and abandons that branch; it never retries the unknown optimizer operation on the same trainer. New collection uses freshly exported sampling weights after each update.

See [local validation results](live-rlvr-validation.md) for the completed 100-cycle equipment-reset soak and 1/2/4-worker gathering runs. Remaining acceptance work includes clean-machine provisioning, the broader reset fault/mutation matrix, browser/gateway resource accounting, and an actual hosted sampling/update/resume run. SFT and broader task curricula follow the verified learning loop.
