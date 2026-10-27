from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rsenv.rlvr.content import level_for_xp
from rsenv.rlvr.contracts import (
    AcquireGoal,
    ActionOutcome,
    EquipGoal,
    InfrastructureError,
    Item,
    Observation,
    Resource,
    SkillGoal,
    TaskSpec,
)
from rsenv.rlvr.tasks import verify_reset, verify_task


def initial(task: TaskSpec) -> Observation:
    spec = task.reset
    return Observation(
        episode_id="episode",
        world_id="world",
        sequence=1,
        tick=10,
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
        resources=(Resource(id=2090, x=3228, z=3147, name="Copper rocks"),),
        npcs=(Resource(id=41, index=2, x=3231, z=3295, name="Chicken", kind="npc", hp=3),),
        in_game=True,
    )


def mine(before: Observation, request_id: str = "1") -> ActionOutcome:
    xp = list(before.xp_tenths)
    xp[14] += 175
    after = before.model_copy(
        update={
            "sequence": before.sequence + 2,
            "tick": before.tick + 5,
            "xp_tenths": tuple(xp),
            "inventory": (*before.inventory, Item(slot=len(before.inventory), id=436, count=1)),
        }
    )
    return ActionOutcome(
        request_id=request_id,
        name="harvest",
        status="completed",
        reason="harvested",
        before=before,
        after=after,
        elapsed_seconds=3,
        target=before.resources[0],
    )


def test_mining_uses_shared_goals_and_evidence() -> None:
    task = TaskSpec(objectives=(AcquireGoal(item_id=436, quantity=1),))
    state = initial(task)
    assert verify_task(task, verify_reset(task, state), [mine(state)]).success


@pytest.mark.parametrize("item_id", [1617, 1619, 1621, 1623])
def test_incidental_mining_gems_do_not_count_as_ore(item_id: int) -> None:
    task = TaskSpec(objectives=(AcquireGoal(item_id=436, quantity=1),))
    state = initial(task)
    action = mine(state)
    gem = Item(slot=1, id=item_id, count=1)
    action = action.model_copy(
        update={
            "status": "failed",
            "reason": "no_yield",
            "after": action.after.model_copy(
                update={"inventory": (*state.inventory, gem), "xp_tenths": state.xp_tenths}
            ),
        }
    )
    result = verify_task(task, verify_reset(task, state), [action])
    assert not result.success and result.progress == 0


def test_incidental_mining_allowance_rejects_other_items() -> None:
    task = TaskSpec(objectives=(AcquireGoal(item_id=436, quantity=1),))
    state = initial(task)
    action = mine(state)
    action = action.model_copy(
        update={
            "after": action.after.model_copy(
                update={"inventory": (*action.after.inventory, Item(slot=2, id=995, count=1))}
            )
        }
    )
    with pytest.raises(InfrastructureError, match="unattributed_items"):
        verify_task(task, verify_reset(task, state), [action])


def test_ordered_goals_do_not_credit_early_progress() -> None:
    task = TaskSpec(objectives=(EquipGoal(item_id=1351), AcquireGoal(item_id=436, quantity=1)), ordered=True)
    state = initial(task)
    first = mine(state)
    equipped = first.after.model_copy(
        update={
            "sequence": 5,
            "tick": 16,
            "equipment": (Item(slot=3, id=1351, count=1),),
            "inventory": (Item(slot=1, id=436, count=1),),
        }
    )
    second = ActionOutcome(
        request_id="2",
        name="equipItem",
        status="completed",
        reason="equipped",
        before=first.after,
        after=equipped,
        elapsed_seconds=1,
    )
    receipt = verify_reset(task, state)
    assert verify_task(task, receipt, [first, second]).progress == 1
    third = mine(equipped, "3")
    assert verify_task(task, receipt, [first, second, third]).success


def test_attack_goal_allows_hitpoints_xp_and_pending_combat() -> None:
    task = TaskSpec(objectives=(SkillGoal(skill="attack", level=2),))
    state = initial(task)
    xp = list(state.xp_tenths)
    xp[0], xp[3] = 840, xp[3] + 280
    after = state.model_copy(
        update={"sequence": 2, "tick": 20, "xp_tenths": tuple(xp), "pending": True, "combat_active": True}
    )
    action = ActionOutcome(
        request_id="1",
        name="attackNpc",
        status="completed",
        reason="action_window_complete",
        before=state,
        after=after,
        target=state.npcs[0],
        elapsed_seconds=6,
    )
    assert verify_task(task, verify_reset(task, state), [action]).success
    fake = after.model_copy(update={"xp_tenths": state.xp_tenths, "levels": (99, *state.levels[1:])})
    assert not verify_task(task, verify_reset(task, state), [action.model_copy(update={"after": fake})]).success


def test_changes_while_paused_invalidate_evidence() -> None:
    task = TaskSpec(objectives=(AcquireGoal(item_id=436, quantity=1),))
    state = initial(task)
    first = mine(state)
    changed = first.after.model_copy(update={"hp": 9})
    with pytest.raises(InfrastructureError, match="unattributed"):
        verify_task(task, verify_reset(task, state), [first, mine(changed, "2")])


def test_inventory_gain_cannot_be_disguised_as_equipment_swap() -> None:
    task = TaskSpec(objectives=(AcquireGoal(item_id=436, quantity=1),))
    state = initial(task)
    action = mine(state).model_copy(update={"name": "equipItem", "target": None})
    with pytest.raises(InfrastructureError):
        verify_task(task, verify_reset(task, state), [action])


def test_food_constraint_is_gameplay_failure() -> None:
    task = TaskSpec(objectives=(AcquireGoal(item_id=436, quantity=1),), preserve_items=(1351,))
    state = initial(task)
    action = mine(state)
    action = action.model_copy(
        update={"after": action.after.model_copy(update={"inventory": (Item(slot=1, id=436, count=1),)})}
    )
    result = verify_task(task, verify_reset(task, state), [action])
    assert not result.success and result.reason == "constraint_failed"


@pytest.mark.parametrize("xp,level", [(0, 1), (799, 1), (800, 2), (1670, 3), (9800, 10), (11320, 11)])
def test_engine_xp_units(xp: int, level: int) -> None:
    assert level_for_xp(xp) == level


def test_calibration_has_distinct_tasks_and_families() -> None:
    tasks = [TaskSpec.model_validate(t) for t in json.loads(Path("configs/rlvr-calibration.json").read_text())]
    assert len(tasks) == len({t.id for t in tasks}) == 12
    assert {t.family for t in tasks} == {"woodcutting", "mining", "combat", "sequence"}
    assert all(t.split == "calibration" for t in tasks)
    assert any(t.ordered for t in tasks) and any(t.preserve_items for t in tasks)


def test_goals_and_reset_must_be_consistent() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(objectives=())
    with pytest.raises(ValidationError):
        TaskSpec(reset={"levels": [99] * 21})


def test_pause_evidence_is_required() -> None:
    raw = initial(TaskSpec()).model_dump()
    raw.pop("paused")
    with pytest.raises(ValidationError):
        Observation.model_validate(raw)


def test_reset_requires_declared_resources() -> None:
    task = TaskSpec(required_resource_items=(1511,))
    with pytest.raises(InfrastructureError, match="required_entities_missing"):
        verify_reset(task, initial(task))


def test_ordered_repeated_item_goals_require_cumulative_retention() -> None:
    task = TaskSpec(
        objectives=(AcquireGoal(item_id=436, quantity=1), AcquireGoal(item_id=436, quantity=1)), ordered=True
    )
    state = initial(task)
    first = mine(state)
    second = mine(first.after, "2")
    receipt = verify_reset(task, state)
    assert verify_task(task, receipt, [first, second]).success
    after_loss = second.after.model_copy(update={"sequence": 7, "tick": 22, "inventory": first.after.inventory})
    loss = ActionOutcome(
        request_id="3",
        name="wait",
        status="completed",
        reason="action_window_complete",
        before=second.after,
        after=after_loss,
        elapsed_seconds=1,
    )
    assert not verify_task(task, receipt, [first, second, loss]).success


def test_reference_policy_uses_mining_for_a_mining_level_goal() -> None:
    from rsenv.rlvr.rollout import ScriptedPolicy, parse_action, visible_state

    task = TaskSpec(objectives=(SkillGoal(skill="mining", level=2),))
    state = initial(task)
    messages = [
        {"role": "system", "content": "Play"},
        {"role": "user", "content": task.goal + "\n" + visible_state(state)},
    ]
    turn = ScriptedPolicy().sample(messages, task, 42, 10)
    action = parse_action(turn, task)
    assert action.name == "harvest" and action.arguments["id"] == 2090


def test_ordered_location_already_reached_needs_no_dummy_action() -> None:
    from rsenv.rlvr.contracts import ReachGoal

    task = TaskSpec(
        objectives=(AcquireGoal(item_id=436, quantity=1), ReachGoal(position=(3200, 3220, 0))), ordered=True
    )
    state = initial(task)
    result = verify_task(task, verify_reset(task, state), [mine(state)])
    assert result.success and result.progress == 2


def test_visible_skills_use_engine_thresholds() -> None:
    from rsenv.rlvr.content import xp_for_level
    from rsenv.rlvr.rollout import visible_state

    state = initial(TaskSpec())
    visible = json.loads(visible_state(state))
    assert "xp_tenths" not in visible and "levels" not in visible
    assert visible["skills"]["attack"] == {"level": 1, "xp_tenths": 0, "next_level_xp_tenths": 800}
    assert visible["skills"]["hitpoints"] == {"level": 10, "xp_tenths": 9800, "next_level_xp_tenths": 11320}
    for level in range(2, 100):
        threshold = xp_for_level(level)
        assert level_for_xp(threshold) == level
        assert level_for_xp(threshold - 1) == level - 1
