# rooner

A RuneScape RLVR experiment using the [LostCity/2004scape](https://2004.lostcity.rs/) game engine and hosted Tinker sampling and training. The policy follows a task goal through tool calls. Each episode runs in a fresh world, and the verifier checks game-state evidence before awarding a binary reward.

## Setup

Install Python 3.11+, [uv](https://docs.astral.sh/uv/), and [Bun 1.3.14](https://bun.sh/). From the repository root:

```sh
uv sync
bash rsenv/bridge/rlvr/setup.sh
uv run python -m rsenv.rlvr doctor
```

The setup script installs the pinned game SDK, assets, gateway, web client, and browser. Each episode starts and stops its own game processes.

## Validate the environment

```sh
uv run python -m pytest rsenv/tests -q
uv run python -m rsenv.rlvr smoke --manifest configs/rlvr-smoke.json --workers 2
```

The smoke runs four episodes: equip an axe, gather a log in each of two worlds, and finish without gathering as a zero-reward control. It checks resets, rewards, and replay of recorded evidence. It needs no model weights or Tinker account.

## Train and evaluate

Set `TINKER_API_KEY` for an account with billing access.

```sh
uv run python -m rsenv.rlvr train --manifest configs/rlvr-train.json --workers 2 --group-size 4 --iterations 25 --output outputs/train
uv run python -m rsenv.rlvr train --resume outputs/train/checkpoint.json --iterations 50
uv run python -m rsenv.rlvr evaluate --checkpoint 'tinker://SAMPLER_PATH' --manifest configs/rlvr-heldout.json --episodes 10
```

Training uses verified episode groups, masked token advantages, and equal trajectory weights. Optimizer state and sampler weights are checkpointed after each update. A small hosted run exercised sampling, optimization, and checkpointing; see [validation results](docs/live-rlvr-validation.md). It does not establish learning improvement.

## Code

- `rsenv/rlvr/contracts.py`, `tasks.py`: task fixtures, episode records, and reward verification.
- `rsenv/rlvr/worlds.py`, `bridge.py`, `rsenv/bridge/rlvr/`: world lifecycle and game execution.
- `rsenv/tools.py`, `tool_parser.py`, `rsenv/rlvr/rollout.py`: tool schemas, parsing, and the episode loop.
- `rsenv/rlvr/learning.py`, `tinker_backend.py`, `runner.py`: GRPO examples, service calls, and training.
- `rsenv/rlvr/records.py`, `cli.py`: persisted evidence, replay, and commands.
- `configs/rlvr-*.json`: smoke, training, and held-out task manifests.

See [setup and operation](docs/live-rlvr.md), [reward and weighting](docs/reward_function_best_practices.md), and [validation results](docs/live-rlvr-validation.md).
