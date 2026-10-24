"""Task-specific verification over authoritative episode evidence."""

from rsenv.rlvr.contracts import (
    ActionOutcome,
    InfrastructureError,
    Observation,
    ResetReceipt,
    TaskSpec,
    VerificationResult,
    fingerprint,
)


def verify_reset(task: TaskSpec, state: Observation) -> ResetReceipt:
    spec = task.reset
    expected = {
        "position": spec.position,
        "xp_tenths": spec.xp_tenths,
        "levels": spec.levels,
        "inventory": spec.inventory,
        "equipment": (),
        "bank": (),
        "hp": spec.hp,
        "pending": False,
        "tutorial": spec.tutorial,
        "in_game": True,
    }
    mismatches = [name for name, value in expected.items() if getattr(state, name) != value]
    if state.count(task.resource_id) or len(state.resources) < task.quantity:
        mismatches.append("available_resources")
    if mismatches:
        raise InfrastructureError(f"reset_mismatch:{','.join(mismatches)}")
    return ResetReceipt(
        spec_hash=fingerprint(spec.model_dump(mode="json")),
        state_hash=fingerprint(state.canonical()),
        observation=state,
    )


def verify_harvest(task: TaskSpec, receipt: ResetReceipt, actions: list[ActionOutcome]) -> VerificationResult:
    baseline = receipt.observation
    if verify_reset(task, baseline) != receipt:
        raise InfrastructureError("invalid_reset_receipt")
    previous = baseline
    gained = 0
    seen: set[str] = set()
    evidence: list[str] = []
    for action in actions:
        before, after = action.before, action.after
        if action.request_id in seen:
            raise InfrastructureError("duplicate_action_evidence")
        seen.add(action.request_id)
        for state in (before, after):
            if (state.episode_id, state.world_id) != (baseline.episode_id, baseline.world_id):
                raise InfrastructureError("foreign_episode_evidence")
            if len(state.xp_tenths) != 21 or not state.in_game:
                raise InfrastructureError("incomplete_state")
        if (
            before.sequence < previous.sequence
            or after.sequence <= before.sequence
            or before.tick < previous.tick
            or after.tick <= before.tick
        ):
            raise InfrastructureError("stale_action_evidence")
        # An ongoing action or an outside mutation between decisions destroys attribution.
        if (
            before.inventory != previous.inventory
            or before.xp_tenths != previous.xp_tenths
            or before.equipment != previous.equipment
            or before.bank != previous.bank
            or after.bank != before.bank
        ):
            raise InfrastructureError("unattributed_progress")
        if after.pending:
            raise InfrastructureError("action_still_pending")
        delta = after.count(task.resource_id) - before.count(task.resource_id)
        xp = after.xp_tenths[task.skill_index] - before.xp_tenths[task.skill_index]
        if delta or xp:
            if (
                action.name != "chopTree"
                or action.status != "completed"
                or not action.target
                or action.target not in before.resources
                or action.target.name != "Tree"
                or delta <= 0
                or xp != delta * task.xp_per_item_tenths
            ):
                raise InfrastructureError("unverified_harvest")
            gained += delta
            evidence.append(action.request_id)
        for idx in range(21):
            if idx != task.skill_index and before.xp_tenths[idx] != after.xp_tenths[idx]:
                raise InfrastructureError("unrelated_xp_change")
        previous = after
    retained = previous.count(task.resource_id) - baseline.count(task.resource_id)
    success = gained >= task.quantity and retained >= task.quantity
    return VerificationResult(
        success=success,
        progress=min(gained, retained),
        reason="goal_complete" if success else "goal_incomplete",
        evidence=tuple(evidence),
        version=task.verifier_version,
    )
