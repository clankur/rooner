"""Shared evidence validation and composable task goals."""

from rsenv.rlvr.content import HARVESTS, SKILLS, level_for_xp
from rsenv.rlvr.contracts import (
    AcquireGoal,
    ActionOutcome,
    EquipGoal,
    Goal,
    InfrastructureError,
    Observation,
    ReachGoal,
    ResetReceipt,
    SkillGoal,
    TaskSpec,
    VerificationResult,
    fingerprint,
)


def owned(state: Observation, item_id: int) -> int:
    return sum(i.count for i in (*state.inventory, *state.equipment, *state.bank) if i.id == item_id)


def verify_reset(task: TaskSpec, state: Observation) -> ResetReceipt:
    spec = task.reset
    expected = dict(
        position=spec.position,
        xp_tenths=spec.xp_tenths,
        levels=spec.levels,
        inventory=spec.inventory,
        equipment=(),
        bank=(),
        hp=spec.hp,
        pending=False,
        paused=True,
        tutorial=spec.tutorial,
        in_game=True,
    )
    mismatches = [name for name, value in expected.items() if getattr(state, name) != value]
    available = {HARVESTS[r.id]["item_id"] for r in state.resources if r.id in HARVESTS}
    if set(task.required_resource_items) - available or set(task.required_npcs) - {n.name for n in state.npcs}:
        mismatches.append("required_entities_missing")
    if mismatches:
        raise InfrastructureError(f"reset_mismatch:{','.join(mismatches)}")
    return ResetReceipt(
        spec_hash=fingerprint(spec.model_dump(mode="json")),
        state_hash=fingerprint(state.canonical()),
        observation=state,
    )


def goal_met(goal: Goal, baseline: Observation, state: Observation) -> bool:
    if isinstance(goal, AcquireGoal):
        return state.count(goal.item_id) - baseline.count(goal.item_id) >= goal.quantity
    if isinstance(goal, SkillGoal):
        index = SKILLS[goal.skill]
        return level_for_xp(baseline.xp_tenths[index]) < goal.level <= level_for_xp(state.xp_tenths[index])
    if isinstance(goal, ReachGoal):
        return (
            state.position[2] == goal.position[2]
            and max(abs(state.position[i] - goal.position[i]) for i in (0, 1)) <= goal.radius
        )
    if isinstance(goal, EquipGoal):
        return any(i.id == goal.item_id for i in state.equipment)
    raise TypeError(f"Unknown goal: {goal}")


def verify_actions(receipt: ResetReceipt, actions: list[ActionOutcome]) -> None:
    previous = receipt.observation
    seen: set[str] = set()
    for action in actions:
        before, after = action.before, action.after
        if action.request_id in seen:
            raise InfrastructureError("duplicate_action_evidence")
        seen.add(action.request_id)
        for state in (before, after):
            if (state.episode_id, state.world_id) != (previous.episode_id, previous.world_id):
                raise InfrastructureError("foreign_episode_evidence")
            if len(state.xp_tenths) != 21 or not state.paused:
                raise InfrastructureError("incomplete_or_unpaused_state")
        if (
            before.sequence < previous.sequence
            or after.sequence <= before.sequence
            or before.tick != previous.tick
            or after.tick < before.tick
            or (after.tick == before.tick and after.canonical() != before.canonical())
        ):
            raise InfrastructureError("stale_action_evidence")
        if before.canonical() != previous.canonical() or after.bank != before.bank:
            raise InfrastructureError("unattributed_progress")
        xp_delta = [b - a for a, b in zip(before.xp_tenths, after.xp_tenths, strict=True)]
        if min(xp_delta) < 0:
            raise InfrastructureError("xp_regressed")
        combat = before.combat_active or after.combat_active or action.name == "attackNpc"
        permitted_skills = {0, 1, 2, 3} if combat else set()
        item_ids = {i.id for i in (*before.inventory, *after.inventory)}
        gained = {i: after.count(i) - before.count(i) for i in item_ids if after.count(i) > before.count(i)}
        if action.name == "harvest" and action.target is not None:
            rule = HARVESTS.get(action.target.id)
            if not rule or action.target not in before.resources:
                raise InfrastructureError("unverified_harvest_target")
            skill = rule["skill_index"]
            quantity = gained.get(rule["item_id"], 0)
            if xp_delta[skill] != quantity * rule["xp_tenths"] or (quantity and action.status != "completed"):
                raise InfrastructureError("unverified_harvest")
            permitted_skills.add(skill)
            gained.pop(rule["item_id"], None)
            # Mining can award uncut gems without ore or XP (pinned mining.rs2).
            incidental = sum(gained.pop(item, 0) for item in rule["incidental_item_ids"])
            if incidental > after.tick - before.tick:
                raise InfrastructureError("unverified_incidental_items")
        if action.name == "attackNpc" and (action.target is None or action.target not in before.npcs):
            if any(xp_delta):
                raise InfrastructureError("unverified_combat_target")
        if action.name == "equipItem":
            all_items = {i.id for i in (*before.inventory, *before.equipment, *after.inventory, *after.equipment)}
            if any(owned(before, i) != owned(after, i) for i in all_items):
                raise InfrastructureError("equipment_changed_item_totals")
        if gained and action.name != "equipItem":
            raise InfrastructureError("unattributed_items")
        if any(xp and index not in permitted_skills for index, xp in enumerate(xp_delta)):
            raise InfrastructureError("unrelated_xp_change")
        previous = after


def verify_task(task: TaskSpec, receipt: ResetReceipt, actions: list[ActionOutcome]) -> VerificationResult:
    baseline = receipt.observation
    if verify_reset(task, baseline) != receipt:
        raise InfrastructureError("invalid_reset_receipt")
    verify_actions(receipt, actions)
    state = baseline
    stage = 0
    stage_start = baseline
    preserved = True
    evidence: list[str] = []
    for action in actions:
        state = action.after
        preserved &= all(owned(state, i) >= owned(baseline, i) for i in task.preserve_items)
        while task.ordered and stage < len(task.objectives) and goal_met(task.objectives[stage], stage_start, state):
            evidence.append(action.request_id)
            stage += 1
            stage_start = state
    progress = stage if task.ordered else sum(goal_met(goal, baseline, state) for goal in task.objectives)
    required_items: dict[int, int] = {}
    for goal in task.objectives:
        if isinstance(goal, AcquireGoal):
            previous_quantity = required_items.get(goal.item_id, 0)
            required_items[goal.item_id] = (
                previous_quantity + goal.quantity if task.ordered else max(previous_quantity, goal.quantity)
            )
    retained = all(state.count(item) - baseline.count(item) >= quantity for item, quantity in required_items.items())
    success = (
        progress == len(task.objectives) and preserved and retained and state.hp >= task.min_final_hp and state.in_game
    )
    if not task.ordered:
        evidence = [a.request_id for a in actions]
    return VerificationResult(
        success=success,
        progress=progress,
        reason="goal_complete" if success else "constraint_failed" if not preserved else "goal_incomplete",
        evidence=tuple(evidence),
        version=task.verifier_version,
    )
