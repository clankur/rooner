# Live RuneScape RLVR

`rsenv.rlvr` runs tasks in isolated 2004scape worlds through `VerifiedBridgeClient`. Each task defines a starting fixture, action and episode budgets, and a reward verifier. The initial task is to harvest and retain normal logs.

## Setup

Use Python 3.11+, Bun **1.3.14**, rs-sdk revision **66ed33320e719e92883d5fbfc05a0b091b54373f**, Puppeteer **24.43.1**, and Tinker **0.35.0**. Run commands from the repository root. `uv sync` installs the pinned Python dependencies from the project lockfile.

```sh
uv sync
bash rsenv/bridge/rlvr/setup.sh
uv run python -m rsenv.rlvr doctor
```

Setup installs the pinned SDK, game assets, dependencies, and Chrome. It rejects an existing SDK checkout at another revision. `doctor` checks these artifacts and browser availability.

If Chrome is missing, run `(cd rsenv/bridge && bun x puppeteer browsers install chrome)`. Use the same `PUPPETEER_CACHE_DIR` for installation and execution when setting a custom cache.

Each episode owns a fresh engine, gateway, browser, and player save directory. Shared assets are symlinked; source patches apply to a private engine copy. Episode logs and fixtures remain in the run's `worlds/` directory after shutdown.

## Environment checks

The smoke command starts four episodes: equip an axe, gather one log in each of two worlds, and finish without gathering as a zero-reward control. It checks equipment mutation, rewards, matching initial fingerprints, distinct worlds, and replay of persisted evidence. It needs no model weights or Tinker account.

```sh
uv run python -m rsenv.rlvr smoke --manifest configs/rlvr-smoke.json --workers 2
uv run python -m pytest rsenv/tests -q
```

`smoke.json` records outcomes and phase latency; failures exit nonzero. `--workers 1` reduces concurrent memory use. The smoke suite always runs four episodes; `--episodes` applies to collection commands.

```sh
uv run python -m rsenv.rlvr scripted --episodes 4 --workers 2 --output outputs/scripted
uv run python -m rsenv.rlvr scripted --episodes 2 --fail-policy --output outputs/failing-control
uv run python -m rsenv.rlvr profile --episodes 4 --output outputs/profile
uv run python -m rsenv.rlvr reset-check --episodes 100 --workers 4 --output outputs/resets
uv run python -m rsenv.rlvr replay --records outputs/scripted/episodes.jsonl
```

`reset-check` equips the axe in each fresh episode and checks initial fingerprints. `profile` measures collection with 1, 2, and 4 workers. Reports separate gameplay outcomes from infrastructure failures and include latency, token counts, and engine memory. Success rates exclude infrastructure-invalid episodes.

## Hosted sampling and training

Set `TINKER_API_KEY` for an account with billing access.

```sh
uv run python -m rsenv.rlvr evaluate --episodes 4 --workers 2 --output outputs/base
uv run python -m rsenv.rlvr train --manifest configs/rlvr-train.json --workers 2 --group-size 4 --iterations 25 --output outputs/train
uv run python -m rsenv.rlvr train --resume outputs/train/checkpoint.json --iterations 25
uv run python -m rsenv.rlvr evaluate --checkpoint 'tinker://SAMPLER_PATH' --manifest configs/rlvr-heldout.json --episodes 4 --output outputs/heldout
```

Defaults are `Qwen/Qwen3.5-4B`, LoRA rank 16, learning rate `1e-5`, PPO clipping `[0.8, 1.2]`, temperature 1, and 512 generated tokens per turn. Thinking, top-k/top-p filtering, and KL penalties are disabled. Evaluation accepts Tinker sampler checkpoints.

Each group shares a task, reset fingerprint, and policy checkpoint. An infrastructure-invalid member rejects the group, with up to three collection attempts. Valid episodes that end before success receive zero reward. Zero-variance groups skip optimization; twenty consecutive such groups stop training. See [reward and weighting](reward_function_best_practices.md) for the equations and masks.

## Evidence and recovery

The verifier checks new log IDs, matching woodcutting XP, a tree target, observation order, and retained inventory. Each action is cancelled before the next decision. Unknown action outcomes invalidate the episode. Policy tools cannot access fixture administration.

Records include task and sampling seeds. Starting fixtures are reproducible; gameplay RNG remains uncontrolled. The held-out manifest varies log quantity only.

Samples are saved before action execution. Each record append is flushed and synced to disk. Replay checks reward decisions and training-conversion hashes using the recorded prompts, generated tokens, and behavior log-probabilities.

`checkpoint.json` stores the committed optimizer and sampler paths. An update is committed after optimizer-state saving and sampler refresh succeed. `pending-update.json` records an interrupted update; resume forks from the last committed optimizer state.

[Validation results](live-rlvr-validation.md) cover live gathering, equipment resets, replay, and cancellation cleanup. Hosted sampling/update/resume remains unvalidated after a billing error. Clean-machine setup, broader reset mutations, and exhaustive transport failures remain open checks.
