# Evaluate tasks before training

`configs/rlvr-calibration.json` contains twelve pilot tasks: woodcutting, copper/tin mining, melee skill leveling, and ordered sequences. The difficulty labels are hypotheses, not measured difficulty. This is a calibration set; it is not a held-out benchmark.

The [initial calibration report](calibration-2026-10-27.md) includes base-model outcomes, reference feasibility, failure traces, and the next evaluation decision.

## Run the pilot

```sh
uv run python -m rsenv.rlvr scripted --manifest configs/rlvr-calibration.json --episodes 1 --workers 3 --output outputs/reference
uv run python -m rsenv.rlvr evaluate --manifest configs/rlvr-calibration.json --episodes 3 --workers 3 --output outputs/baseline
uv run python -m rsenv.rlvr replay --manifest configs/rlvr-calibration.json --records outputs/baseline/episodes.jsonl
```

The reference policy establishes feasibility through the same action interface. It has no fixture-control access and does not change rewards. The model run uses the base Qwen/Qwen3.5-4B policy, temperature 1, and no optimizer. `--episodes` is repetitions per task. Workers are shared across the whole manifest; task IDs determine sampling seeds independently of manifest order. Game RNG remains uncontrolled, so these are repeated trials, not deterministic paired trajectories.

`benchmark.json` reports each task and family, with success rates, Wilson intervals, failure reasons, invalid decisions, failed actions, generated tokens, sampling latency, and game ticks. Family scores average task success rates so a family cannot dominate just by having more cases. Per-task throughput uses summed episode durations, while overall throughput uses elapsed run time.

Inspect terminal limits in the traces: `generation_limit` means a 512-token output was truncated; `context_limit` means the next prompt plus output allowance exceeded the 65,536-token context budget. The latter records no generated tokens. The initial calibration used `generation_limit` for both; its report distinguishes them using token counts. `observed_seconds_per_tick` includes paused thinking time; use game ticks and sampling latency separately when comparing decision quality and speed.

Three trials are diagnostic, not a precise success-rate estimate:

- All successes: possible ceiling; repeat and add meaningful planning requirements.
- Mixed results: inspect decisions and RNG effects; potentially useful training signal.
- No successes: check the reference policy and traces before labeling the task difficult.
- Infrastructure failures: fix the environment; do not count these as zero-reward gameplay.

Review these results before choosing a training run. Freeze a separate held-out set after calibration, with disjoint starting conditions and goal combinations. Do not tune task selection against that final test set. Merely increasing resource quantities does not create a convincing generalization test.

## Task definitions

Tasks contain starting fixtures, objectives, ordering, constraints, supported tools, and budgets. The rollout and trainer do not dispatch on task IDs or families. Objectives are a tagged union:

| Kind | Fields | Success condition |
|---|---|---|
| `acquire` | `item_id`, `quantity` | New items relative to the starting inventory, retained at completion |
| `skill` | `skill`, `level` | Earn enough authoritative XP to reach the specified level |
| `equip` | `item_id` | Item appears in equipment |
| `reach` | `position`, `radius` | Reach the specified plane and coordinate tolerance |

`ordered: true` checks one objective at a time. New acquisition or skill progress must occur after its stage starts. Acquired items must still be retained at task completion. Unordered objectives are checked together in the final state. `preserve_items` requires keeping starting quantities across inventory and equipment throughout the episode; `min_final_hp` constrains completion health.

For example, equipping a pickaxe, mining copper, then returning to a position requires only data:

```json
{
  "id": "mine-return/v2",
  "family": "sequence",
  "difficulty": "control",
  "split": "calibration",
  "reset": {
    "position": [3286, 3363, 0],
    "inventory": [{"slot": 0, "id": 1265, "count": 1}]
  },
  "required_resource_items": [436],
  "ordered": true,
  "objectives": [
    {"kind": "equip", "item_id": 1265},
    {"kind": "acquire", "item_id": 436, "quantity": 1},
    {"kind": "reach", "position": [3286, 3363, 0], "radius": 1}
  ]
}
```

Required nearby resources/NPCs validate fixture prerequisites at reset. They do not imply the entire objective is feasible; run the reference policy. Fixture levels must agree with XP under the pinned engine's curve, which uses `level / 10`, not standard RuneScape's `level / 7`. Ten Hitpoints starts at 9,800 engine XP tenths.

## Actions, evidence, and timing

The policy chooses explicit tree/rock coordinates, NPC indices, inventory slots, and melee style. `harvest` uses the shared content table to support both trees and ore. Walking uses game pathfinding; it does not automatically open doors. Current fixtures avoid routes requiring door interactions.

Model observations name each supported skill and show its earned level, XP, and next-level threshold. Authoritative records retain the full engine arrays. The policy emits one tool call per decision; the verifier ends successful episodes.

Simulation pauses between decisions. Normal 600 ms ticks advance only during bounded action windows. The engine enforces the tick cap; model latency is measured separately. Combat state persists across turns, including while eating or waiting. Harvesting and walking stop at their action boundary. Wall-clock timeouts remain infrastructure safeguards.

The shared verifier checks episode identity, paused boundaries, state continuity, inventory changes, and XP attribution. Harvest yields require the matching item and XP delta for the selected observed resource. Combat permits Attack/Strength/Defence and Hitpoints XP; skill goals use the engine XP curve rather than temporary stat boosts. Privileged reset and clock controls are inaccessible to policy tools.

Protocol v2 and `goals-v2` replace the original woodcutting-only contracts. Historical v1 records require their corresponding commit; there is no compatibility flow.

## Expand from evidence

Use the pilot to choose combinations that require decisions rather than longer repetition. Candidates for the next set:

| Task | Decision being evaluated | Additional mechanic/evidence needed |
|---|---|---|
| Raise Attack, then Defence, while preserving food | Switch style at the right time; manage combat | Current goals and tools |
| Mine a mixed order, then deliver it to a different location | Select resources and plan a route | A reference-verified destination |
| Gather with distractor tools and irrelevant resources | Select usable equipment and targets | Additional fixture variants |
| Finish a combat goal from low health with limited food | Decide when to heal or retreat | Reference-check survivability |
| Mine copper and tin, smelt bronze, smith an item | Plan dependencies and resource ratios | Furnace/anvil actions and transformation checks |
| Fish, cook, and provision a combat trip | Plan consumables across skills | Fishing/cooking actions and consumption checks |
| Fill an order larger than inventory capacity | Schedule bank trips and preserve tools | Banking actions and bank transfer checks |
| Buy tools and supplies within a coin budget | Allocate scarce money | Shop actions, prices, and transaction checks |
| Recover from depleted resources or blocked routes | Revise a plan from observed failure | Controlled obstacle fixtures; door interaction where required |
| Complete objectives across distant areas | Choose route and objective order | Verified routes and held-out starting positions |

For each mechanic, retain a short control that isolates tool competence. Combine mechanics only after their reference trajectories pass. Hold out goal combinations, starting positions, and quantities rather than changing wording alone. Keep the initial pilot results separate from later sets selected using those results.

Add a supported mechanic and its evidence checks once, then author task combinations as data. Do not add task-ID branches or an arbitrary expression/plugin system.
