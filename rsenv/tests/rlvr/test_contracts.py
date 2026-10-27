from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from rsenv.rlvr.contracts import (
    ActionOutcome,
    EpisodeResult,
    InfrastructureError,
    Item,
    Observation,
    Resource,
    SampledTurn,
    TaskSpec,
)
from rsenv.rlvr.learning import TrainingExample, build_examples
from rsenv.rlvr.rollout import ScriptedPolicy, parse_action, rollout
from rsenv.rlvr.tasks import verify_reset, verify_task


@pytest.fixture
def initial() -> Observation:
    spec = TaskSpec().reset
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
        hp=10,
        pending=False,
        paused=True,
        tutorial=1000,
        in_game=True,
        resources=tuple(Resource(id=1276, x=3200 + i, z=3220, name="Tree") for i in range(6)),
    )


def harvest(before: Observation, count: int = 1, request_id: str = "1") -> ActionOutcome:
    xp = list(before.xp_tenths)
    xp[8] += 250 * count
    inv = before.inventory + tuple(Item(slot=len(before.inventory) + i, id=1511, count=1) for i in range(count))
    after = before.model_copy(
        update={"sequence": before.sequence + 1, "tick": before.tick + 1, "xp_tenths": tuple(xp), "inventory": inv}
    )
    return ActionOutcome(
        request_id=request_id,
        name="harvest",
        status="completed",
        reason="harvested",
        before=before,
        after=after,
        target=before.resources[0],
        elapsed_seconds=1,
    )


def turn(text: str, stop: str = "stop") -> SampledTurn:
    return SampledTurn(
        prompt_ids=(1, 2, 3),
        output_ids=(4, 5),
        logprobs=(-1.0, -2.0),
        text=text,
        stop_reason=stop,
        policy_version="policy",
        renderer="test",
        seed=1,
    )


def test_environment_import_is_model_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rsenv.rlvr.bridge,sys; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'tinker' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_reset_fingerprint_and_repeated_isolation(initial: Observation) -> None:
    for i in range(100):
        changed_ids = initial.model_copy(update={"episode_id": str(i), "world_id": str(i), "tick": 50 + i})
        assert verify_reset(TaskSpec(), changed_ids).state_hash == verify_reset(TaskSpec(), initial).state_hash
    for field, value in [
        ("equipment", (Item(slot=0, id=1351, count=1),)),
        ("hp", 9),
        ("pending", True),
        ("bank", (Item(slot=0, id=1511, count=1),)),
        ("position", (3201, 3220, 0)),
        ("paused", False),
    ]:
        with pytest.raises(InfrastructureError):
            verify_reset(TaskSpec(), initial.model_copy(update={field: value}))


def test_goal_boundary_and_retention(initial: Observation) -> None:
    receipt = verify_reset(TaskSpec(), initial)
    assert not verify_task(TaskSpec(), receipt, []).success
    assert not verify_task(TaskSpec(), receipt, [harvest(initial, 4)]).success
    assert verify_task(TaskSpec(), receipt, [harvest(initial, 5)]).success
    with pytest.raises(InfrastructureError, match="duplicate"):
        action = harvest(initial)
        verify_task(TaskSpec(), receipt, [action, action])


@pytest.mark.parametrize(
    "attack", ["wrong_item", "wrong_skill", "stale", "foreign", "unpaused", "preloaded", "false_sdk"]
)
def test_verifier_rejects_adversarial_evidence(initial: Observation, attack: str) -> None:
    receipt = verify_reset(TaskSpec(), initial)
    action = harvest(initial, 5)
    if attack == "wrong_item":
        action = action.model_copy(update={"after": action.after.model_copy(update={"inventory": initial.inventory})})
    elif attack == "wrong_skill":
        xp = list(action.after.xp_tenths)
        xp[1] += 1
        action = action.model_copy(update={"after": action.after.model_copy(update={"xp_tenths": tuple(xp)})})
    elif attack == "stale":
        action = action.model_copy(update={"after": action.after.model_copy(update={"tick": initial.tick})})
    elif attack == "foreign":
        action = action.model_copy(update={"after": action.after.model_copy(update={"episode_id": "other"})})
    elif attack == "unpaused":
        action = action.model_copy(update={"after": action.after.model_copy(update={"paused": False})})
    elif attack == "preloaded":
        action = action.model_copy(update={"before": action.after})
    else:
        action = action.model_copy(update={"status": "failed"})
    with pytest.raises(InfrastructureError):
        verify_task(TaskSpec(), receipt, [action])


@pytest.mark.parametrize(
    "text,status",
    [
        ('<tool_call>{"name":"harvest","arguments":{"id":1276,"x":3200,"z":3220}}</tool_call>', "action"),
        (
            "<tool_call><function=harvest><parameter=id>1276</parameter><parameter=x>3200</parameter><parameter=z>3220</parameter></function></tool_call>",
            "action",
        ),
        ("<tool_call>{broken</tool_call>", "invalid"),
        ('<tool_call>{"name":[],"arguments":{}}</tool_call>', "invalid"),
        ('<tool_call>{"name":"adminCommand","arguments":{}}</tool_call>', "invalid"),
        ('<tool_call>{"name":"walkTo","arguments":{"x":"bad","z":2}}</tool_call>', "invalid"),
        ('<tool_call>{"name":"harvest","arguments":[]}</tool_call>', "invalid"),
        (
            '<tool_call>{"name":"harvest","arguments":{"id":1276,"x":3200,"z":3220,"unexpected":1}}</tool_call>',
            "invalid",
        ),
        ('<tool_call>{"name":"harvest","arguments":{"x":3200,"z":3220}}</tool_call>', "invalid"),
        ('<tool_call>{"name":"equipItem","arguments":{"slot":28}}</tool_call>', "invalid"),
        ('<tool_call>{"name":"walkTo","arguments":{"x":-1,"z":3220}}</tool_call>', "invalid"),
        ('<tool_call>{"name":"walkTo","arguments":{"x":16384,"z":3220}}</tool_call>', "invalid"),
        ("", "invalid"),
        ("Done.", "final"),
    ],
)
def test_parser_records_rejections(text: str, status: str) -> None:
    assert parse_action(turn(text), TaskSpec()).status == status
    assert parse_action(turn(text, "length"), TaskSpec()).status == "truncated"


def test_multiple_calls_do_not_execute() -> None:
    text = '<tool_call>{"name":"harvest","arguments":{"id":1276,"x":3200,"z":3220}}</tool_call>'
    assert parse_action(turn(text * 2), TaskSpec()).status == "invalid"


def test_context_and_output_limits_are_distinct() -> None:
    assert parse_action(turn("", "context_limit"), TaskSpec()).reason == "context_limit"
    assert parse_action(turn("", "max_tokens"), TaskSpec()).reason == "generation_limit"


class FakeClient:
    def __init__(self, initial: Observation, broken: bool = False) -> None:
        self.episode_id = initial.episode_id
        self.state = initial
        self.closed = False
        self.broken = broken
        self.index = 0

    def start(self) -> Observation:
        if self.broken:
            raise InfrastructureError("reset_failed")
        return self.state

    def act(self, name: str, arguments: dict, timeout: float, max_ticks: int = 35) -> ActionOutcome:
        self.index += 1
        action = harvest(self.state, request_id=str(self.index))
        self.state = action.after
        return action

    def stop(self) -> None:
        self.closed = True


def test_rollout_stops_at_verified_success(initial: Observation) -> None:
    client = FakeClient(initial)
    result = rollout(client, TaskSpec(), ScriptedPolicy(), 42)
    assert result.status == "success" and result.reward == 1
    assert len(result.transitions) == 5 and client.closed


def test_infrastructure_errors_have_no_reward(initial: Observation) -> None:
    client = FakeClient(initial, broken=True)
    result = rollout(client, TaskSpec(), ScriptedPolicy(), 42)
    assert result.status == "infra_invalid" and result.reward is None and client.closed


def test_rejected_action_evidence_is_retained(initial: Observation) -> None:
    class InvalidEvidenceClient(FakeClient):
        def act(self, name: str, arguments: dict, timeout: float, max_ticks: int = 35) -> ActionOutcome:
            action = super().act(name, arguments, timeout, max_ticks)
            return action.model_copy(update={"after": action.after.model_copy(update={"episode_id": "foreign"})})

    result = rollout(InvalidEvidenceClient(initial), TaskSpec(), ScriptedPolicy(), 42)
    assert result.status == "infra_invalid" and result.reward is None
    assert result.reason == "foreign_episode_evidence"
    assert len(result.transitions) == 1
    assert result.transitions[0].outcome.after.episode_id == "foreign"


def episodes(initial: Observation) -> list[EpisodeResult]:
    result = rollout(FakeClient(initial), TaskSpec(), ScriptedPolicy(), 42)
    sampled = turn("test")
    transition = result.transitions[0].model_copy(update={"turn": sampled})
    return [
        result.model_copy(
            update={
                "episode_id": f"ep{i}",
                "policy_version": "policy",
                "reward": float(i),
                "status": "success" if i else "failure",
                "verification": result.verification.model_copy(update={"success": bool(i)}),
                "transitions": (transition,) if i == 0 else (transition, transition),
            }
        )
        for i in range(2)
    ]


def test_exact_context_masks_and_trajectory_weighting(initial: Observation) -> None:
    batch = build_examples(episodes(initial), 2)
    assert batch[0].input_ids == (1, 2, 3, 4)
    assert batch[0].targets == (2, 3, 4, 5)
    assert batch[0].behavior_logprobs == (0, 0, -1, -2)
    assert batch[0].advantages[:2] == (0, 0)
    assert sum(batch[0].advantages) == pytest.approx(-sum(batch[1].advantages) - sum(batch[2].advantages))


def test_degenerate_and_invalid_groups(initial: Observation) -> None:
    group = episodes(initial)
    with pytest.raises(InfrastructureError):
        build_examples(group[:1], 1)
    assert (
        build_examples(
            [
                e.model_copy(
                    update={
                        "reward": 0.0,
                        "status": "failure",
                        "verification": e.verification.model_copy(update={"success": False}),
                    }
                )
                for e in group
            ],
            2,
        )
        == []
    )
    with pytest.raises(InfrastructureError):
        build_examples(group * 2, 4)
    with pytest.raises(InfrastructureError):
        build_examples([group[0], group[1].model_copy(update={"status": "infra_invalid", "reward": None})], 2)
    corrupted = (
        group[0]
        .transitions[0]
        .model_copy(update={"turn": turn("test").model_copy(update={"logprobs": (math.nan, -1)})})
    )
    with pytest.raises(InfrastructureError):
        build_examples([group[0].model_copy(update={"transitions": (corrupted,)}), group[1]], 2)


def test_entire_group_retried_after_infrastructure_failure(
    initial: Observation, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rsenv.rlvr import runner
    from rsenv.rlvr.records import Recorder

    good = episodes(initial)
    invalid = good[1].model_copy(update={"status": "infra_invalid", "reward": None})
    calls = []

    def collect(*args: object) -> list[EpisodeResult]:
        calls.append(args)
        return [good[0], invalid] if len(calls) == 1 else good

    monkeypatch.setattr(runner, "collect", collect)
    assert runner.collect_group(TaskSpec(), ScriptedPolicy(), 2, 2, 42, tmp_path, Recorder(tmp_path)) == good
    assert len(calls) == 2 and calls[0][2] == calls[1][2] == 2


def test_checkpoint_commit_waits_for_sampler_refresh(
    initial: Observation, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from rsenv.rlvr import runner

    class Backend:
        model = "test"
        rank = 16

        def __init__(self) -> None:
            self.updates = 0

        def sampling_policy(self, name: str) -> ScriptedPolicy:
            if self.updates:
                raise InfrastructureError("sampler_export_failed")
            policy = ScriptedPolicy()
            policy.renderer = "scripted-v2"
            policy.max_context = 65536
            return policy

        def save_state(self, name: str) -> str:
            return "optimizer-initial" if not self.updates else "optimizer-new"

        def update(self, examples: list[TrainingExample], lr: float) -> dict[str, float]:
            self.updates += 1
            return {"loss": 0.1}

    backend = Backend()
    monkeypatch.setattr(runner, "collect_group", lambda *args: episodes(initial))
    with pytest.raises(InfrastructureError, match="sampler_export_failed"):
        runner.train(backend, [TaskSpec()], tmp_path, group_size=2, iterations=1)
    committed = json.loads((tmp_path / "checkpoint.json").read_text())
    assert committed["updates"] == 0 and committed["training_path"] == "optimizer-initial"
    assert (tmp_path / "pending-update.json").exists() and backend.updates == 1


def test_reward_and_receipt_corruption_rejected(initial: Observation) -> None:
    group = episodes(initial)
    with pytest.raises(InfrastructureError, match="unverified_group_reward"):
        build_examples([group[0].model_copy(update={"reward": 1.0}), group[1]], 2)
    receipt = verify_reset(TaskSpec(), initial).model_copy(update={"state_hash": "forged"})
    with pytest.raises(InfrastructureError, match="invalid_reset_receipt"):
        verify_task(TaskSpec(), receipt, [])


def test_bridge_rejects_wrong_action_name(initial: Observation, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from rsenv.rlvr.bridge import VerifiedBridgeClient

    client = VerifiedBridgeClient(SimpleNamespace(world_id="world"), TaskSpec(), "episode")
    monkeypatch.setattr(client, "rpc", lambda *args, **kwargs: harvest(initial).model_dump(mode="json"))
    with pytest.raises(InfrastructureError, match="bridge_action_name_mismatch"):
        client.act("walkTo", {"x": 3200, "z": 3220}, 1)
    assert client.poisoned


def test_unattributed_bank_mutation_invalidates_episode(initial: Observation) -> None:
    action = harvest(initial)
    action = action.model_copy(
        update={"after": action.after.model_copy(update={"bank": (Item(slot=0, id=1511, count=100),)})}
    )
    with pytest.raises(InfrastructureError, match="unattributed_progress"):
        verify_task(TaskSpec(), verify_reset(TaskSpec(), initial), [action])


@pytest.mark.parametrize("group_size", [2, 4])
def test_group_and_token_normalization(initial: Observation, group_size: int) -> None:
    pair = episodes(initial)
    group = [pair[i % 2].model_copy(update={"episode_id": f"episode-{i}"}) for i in range(group_size)]
    batch = build_examples(group, group_size)
    offset = 0
    for episode in group:
        turns = batch[offset : offset + len(episode.transitions)]
        offset += len(episode.transitions)
        expected_advantage = -1.0 if episode.reward == 0.0 else 1.0
        assert sum(sum(turn.advantages) for turn in turns) == pytest.approx(expected_advantage / group_size)
        for example in turns:
            assert example.advantages[:2] == (0.0, 0.0)
            assert len(example.input_ids) == len(example.targets) == len(example.advantages)
