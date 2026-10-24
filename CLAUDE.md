# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

rooner ([clankur/rooner](https://github.com/clankur/rooner)) is a GRPO training experiment for teaching an LLM to play RuneScape on the LostCity/2004scape server. The policy follows task goals through tool calls in isolated live worlds. Game-state evidence determines rewards; Tinker provides hosted sampling and optimization. The defining design value is **explicitness**: the algorithm, state transitions, and computational costs should be visible in source. Minimize the conceptual machinery needed to understand and change the program.

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

Requires Python >= 3.11, uv, and Bun 1.3.14. Run from the repository root.

### Install and check the game runtime

```
uv sync
bash rsenv/bridge/rlvr/setup.sh
uv run python -m rsenv.rlvr doctor
uv run python -m rsenv.rlvr smoke --manifest configs/rlvr-smoke.json --workers 2
```

### Train

Set `TINKER_API_KEY` for an account with billing access.

```
uv run python -m rsenv.rlvr train --manifest configs/rlvr-train.json --workers 2 --group-size 4 --iterations 25 --output outputs/train
```

### Tests, lint, and format

```
uv run python -m pytest rsenv/tests -q
uvx ruff check
uvx ruff format
```

### Experiment reports (Quarto)

```
uv run --group docs quarto preview docs
uv run --group docs quarto render docs
```

## Architecture

- `rsenv/rlvr/contracts.py`, `tasks.py` — Task specifications, episode evidence, and reward verification.
- `rsenv/rlvr/worlds.py`, `bridge.py` — Private game worlds and bounded action transport.
- `rsenv/bridge/rlvr/` — Pinned runtime setup, authoritative engine observations, and tool execution.
- `rsenv/tools.py`, `tool_parser.py`, `rsenv/rlvr/rollout.py` — Tool schemas, parsing, and episode control flow.
- `rsenv/rlvr/learning.py`, `tinker_backend.py`, `runner.py` — Token masks, GRPO weighting, service calls, and checkpointed training.
- `rsenv/rlvr/records.py`, `cli.py` — Durable records, replay, and command entry point.
- `configs/rlvr-*.json` — Smoke, training, and held-out task manifests.
