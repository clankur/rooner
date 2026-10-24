# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

rooner ([clankur/rooner](https://github.com/clankur/rooner)) is a PyTorch GRPO training codebase for teaching a small LLM (Qwen3 / Gemma 4) to play RuneScape on the LostCity/2004scape server. The model plans then executes: it generates a high-level strategy, then drives action-by-action execution using tool calls, observing game state between actions. The defining design value is **explicitness**: the algorithm, state transitions, and computational costs should be visible in source. Minimize the conceptual machinery needed to understand and change the program.

## Code Standards

### Write directly

State what something is, does, or demonstrates. Keep documentation and explanations concise and factual. Include comparisons, caveats, and limitations only when they answer a likely reader question or affect a decision. Describe verification by naming the behavior checked and the observed result.

### Own the core; build on primitives

Prefer direct implementations of core algorithms using low-level, well-supported libraries such as PyTorch. Keep the mechanics visible and editable. Before adopting a framework or abstraction, weigh the code it removes against the concepts, indirection, and constraints it introduces.

Implement a small operation locally when doing so materially improves clarity or control. Use established libraries for complex supporting functionality outside the project's core. Introduce abstractions for established concepts or repeated complexity; require a concrete current use for each extension point.

### Preserve linear readability

Keep the main algorithm readable in execution order, with closely related steps together. Extract a helper when it names a meaningful operation or isolates substantial complexity. Each layer of indirection should contribute meaning and keep the flow easy to follow.

Use intermediate names that correspond to mathematical quantities. Keep normalization, masking, clipping, and reduction order visible so the implementation can be checked against the equations.

### Make costs and decisions visible at point of use

Keep expensive allocations, copies, device transfers, synchronization, and recomputation explicit at their call sites. Place casts, validation, transfers, and reductions beside the operations that require them. Make deliberate choices about retaining large intermediates and preserving precision during sensitive reductions.

Give primitives narrow, predictable behavior. Keep additional communication, mutation, and I/O explicit so callers can reason about their effects locally.

### Represent state as data

Bundle related state in named records and pass it explicitly. Prefer immutable configuration. Make state transitions apparent in function inputs and outputs, including any mutations in core algorithms.

### Comments explain decisions

Leave a comment only when the next reader would otherwise be confused or repeat a known mistake.

### Type annotations

All functions must have full type signatures. Use dataclasses to bundle related return values instead of bare tuples. Express meaningful domain constraints such as tensor dimensions, dtype, units, and ownership where useful. Validate assumptions that types cannot enforce near the operations that depend on them.

### Einops for shape transforms

Use `einops.rearrange` / `einops.einsum` instead of `.reshape`, `.unsqueeze`, `torch.einsum`. Named dimensions read as documentation.

### Reproducibility and numerical validation

Keep random seeds and generators explicit. Where supported, derive independent random streams from stable semantic names so unrelated initialization changes do not perturb them.

When changing numerical behavior, validate the relevant mathematical property with a small controlled example and state what was checked. Choose checks appropriate to the change, such as masking correctness, reduction scaling, or finite gradients. For GRPO, validate objective correctness and reward behavior using checks appropriate to the experiment.

## Common commands

### Install

Requires Python >= 3.11. CPU dev:

```
uv sync --extra cpu
```

For GPU use `uv sync --extra gpu`.

### Local CPU smoke test

```
RUNQ_EXPERIMENT_ID=local uv run python -m train --config-name=local_test ++paths.model_name=smoke_000
```

### Run on remote GPU (fractal)

```
uv run python -m train --config-name=gpu_4090 ++paths.model_name=my_experiment
```

Without `RUNQ_EXPERIMENT_ID`, this auto-submits to the runq queue via `execute_remotely()`.

### Hydra overrides

```
uv run python -m train --config-name=gpu_4090 ++paths.model_name=grpo_001 ++grpo.learning_rate=3e-5
```

### Lint and format

```
uvx ruff check
uvx ruff format
```

### Docs (Quarto)

```
quarto preview docs
quarto render docs
```

### Tests

Run verifier, transport, training-conversion, and legacy reward tests without model weights:

```
uv run --no-project --with-requirements requirements-rlvr.txt python -m pytest rsenv/tests/rlvr rsenv/tests/test_reward.py -q
```

For the real-game smoke, follow `docs/live-rlvr.md`. Hosted training requires Tinker billing access.

## Architecture

- `train.py` — Model loading (torchao int4), GRPO algorithm, training loop, wandb logging, runq remote execution, Hydra entrypoint. Readable linearly.
- `rsenv/` — Game state, tool parsing, simulator and live clients, trajectory rollout, and shaped rewards.
- `rsenv/rlvr/` — Isolated live episodes, task verification, GRPO example conversion, Tinker sampling/training, and checkpoint records.
- `configs/` — Hydra YAML configs. `base.yaml` is the schema; other configs inherit and override model/hardware settings.

## Remote infrastructure

- **runq** — Self-hosted experiment queue at https://github.com/clankur/runq. Server runs on fractal (192.168.4.85:8080).
- **fractal** — GPU box (RTX 4090, Ubuntu 22.04, driver 570, CUDA 12.8). SSH alias `fractal`, user `clankur`.
- **wandb** — Metrics logged to `clankur-personal/pooner` project.
