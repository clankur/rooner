"""Bounded goal-following episodes; parsing never erases rejected attempts."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from pydantic import ValidationError

from rsenv.rlvr.contracts import (
    ActionOutcome,
    EpisodeResult,
    InfrastructureError,
    Observation,
    ParsedAction,
    SampledTurn,
    TaskSpec,
    Transition,
    fingerprint,
)
from rsenv.rlvr.records import Recorder
from rsenv.rlvr.tasks import verify_harvest, verify_reset
from rsenv.tool_parser import extract_tool_calls
from rsenv.tools import TOOL_MODELS


class Policy(Protocol):
    version: str

    def sample(self, messages: list[dict[str, str]], task: TaskSpec, seed: int, timeout: float) -> SampledTurn: ...


class EpisodeClient(Protocol):
    episode_id: str

    def start(self) -> Observation: ...
    def act(self, name: str, arguments: dict[str, Any], timeout: float) -> ActionOutcome: ...
    def stop(self) -> None: ...


def parse_action(turn: SampledTurn, task: TaskSpec) -> ParsedAction:
    if turn.stop_reason in ("length", "max_tokens"):
        return ParsedAction(status="truncated", reason="generation_limit")
    calls = extract_tool_calls(turn.text)
    if not calls:
        if not turn.text.strip() or "<tool_call" in turn.text:
            return ParsedAction(status="invalid", reason="empty_or_malformed_tool_call")
        return ParsedAction(status="final", reason="model_final_answer")
    if len(calls) != 1 or turn.text.count("<tool_call>") != 1:
        return ParsedAction(status="invalid", reason="one_tool_call_required")
    name, args = calls[0]
    if not isinstance(name, str) or name not in task.allowed_tools:
        return ParsedAction(status="invalid", reason="unsupported_tool")
    if not isinstance(args, dict):
        return ParsedAction(status="invalid", reason="arguments_must_be_object")
    try:
        model = TOOL_MODELS[name]
        if set(args) - model.model_fields.keys():
            raise ValueError("extra arguments")
        arguments = model.model_validate(args).model_dump(mode="json", exclude_none=True)
        if name == "chopTree" and arguments.get("tree_type", "Tree") != "Tree":
            raise ValueError("unsupported tree")
        if name == "equipItem" and arguments["item"] != "Bronze axe":
            raise ValueError("unsupported equipment")
        if name == "walkTo" and not all(0 <= arguments[k] < 16384 for k in ("x", "z")):
            raise ValueError("coordinates outside world")
    except (ValidationError, ValueError, TypeError):
        return ParsedAction(status="invalid", reason="invalid_arguments")
    return ParsedAction(status="action", name=name, arguments=arguments)


def visible_state(state: Observation) -> str:
    return json.dumps(
        state.model_dump(mode="json", exclude={"bank", "tutorial", "sequence", "world_id", "episode_id", "runtime"}),
        sort_keys=True,
    )


class ScriptedPolicy:
    def __init__(self, fail: bool = False, reset_probe: bool = False) -> None:
        self.fail = fail
        self.reset_probe = reset_probe
        if reset_probe:
            self.version = "reset-probe-v1"
        elif fail:
            self.version = "scripted-fail-v1"
        else:
            self.version = "scripted-gather-v1"

    def sample(self, messages: list[dict[str, str]], task: TaskSpec, seed: int, timeout: float) -> SampledTurn:
        if self.reset_probe:
            if len(messages) == 2:
                text = '<tool_call>{"name":"equipItem","arguments":{"item":"Bronze axe"}}</tool_call>'
            else:
                text = "Reset mutation complete."
        elif self.fail:
            text = "I am finished."
        else:
            text = '<tool_call>{"name":"chopTree","arguments":{"tree_type":"Tree"}}</tool_call>'
        return SampledTurn(
            prompt_ids=(),
            output_ids=(),
            logprobs=(),
            text=text,
            stop_reason="stop",
            policy_version=self.version,
            renderer="scripted-v1",
            seed=seed,
        )


def rollout(
    client: EpisodeClient, task: TaskSpec, policy: Policy, seed: int, recorder: Recorder | None = None
) -> EpisodeResult:
    started = time.monotonic()
    reset_seconds = 0.0
    receipt = None
    verification = None
    transitions: list[Transition] = []
    actions: list[ActionOutcome] = []
    status, reason = "failure", "model_final_answer"
    reward: float | None = 0.0
    try:
        state = client.start()
        receipt = verify_reset(task, state)
        reset_seconds = time.monotonic() - started
        deadline = time.monotonic() + task.episode_seconds
        messages = [
            {
                "role": "system",
                "content": "You control a RuneScape character. Follow the user's goal. "
                'Issue exactly one tool call per turn using <tool_call>{"name":"tool","arguments":{}}</tool_call>. '
                "Wait for the result before choosing another action. Never claim success without game evidence.",
            },
            {"role": "user", "content": task.goal + "\n" + visible_state(state)},
        ]
        for index in range(task.max_turns):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status, reason = "truncated", "episode_deadline"
                break
            sample_start = time.monotonic()
            turn = policy.sample(messages, task, seed + index, remaining)
            sample_seconds = time.monotonic() - sample_start
            if turn.policy_version != policy.version:
                raise InfrastructureError("policy_changed_during_episode")
            parsed = parse_action(turn, task)
            if recorder:
                recorder.append(
                    "sample",
                    {
                        "episode_id": client.episode_id,
                        "turn_index": index,
                        "turn": turn.model_dump(mode="json"),
                        "parsed": parsed.model_dump(mode="json"),
                    },
                )
            outcome = None
            if time.monotonic() >= deadline:
                parsed = ParsedAction(status="truncated", reason="episode_deadline")
            if parsed.status == "action":
                outcome = client.act(
                    parsed.name, parsed.arguments, min(task.action_seconds, deadline - time.monotonic())
                )
                state = outcome.after
                actions.append(outcome)
            verification = verify_harvest(task, receipt, actions)
            transition = Transition(
                turn=turn, parsed=parsed, outcome=outcome, observation=state, sampling_seconds=sample_seconds
            )
            transitions.append(transition)
            if recorder:
                recorder.append("transition", {"episode_id": client.episode_id, **transition.model_dump(mode="json")})
            if verification.success:
                status, reason, reward = "success", "goal_complete", 1.0
                break
            if state.hp <= 0:
                status, reason = "failure", "death"
                break
            if parsed.status in ("final", "truncated"):
                status = "failure" if parsed.status == "final" else "truncated"
                reason = parsed.reason
                break
            messages.append({"role": "assistant", "content": turn.text})
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "result": outcome.reason if outcome else parsed.reason,
                            "state": json.loads(visible_state(state)),
                            "progress": verification.progress,
                        }
                    ),
                }
            )
        else:
            status, reason = "truncated", "turn_limit"
    except Exception as exc:
        status, reward = "infra_invalid", None
        reason = str(exc) if isinstance(exc, InfrastructureError) else f"{type(exc).__name__}: {exc}"
    finally:
        try:
            client.stop()
        except Exception:
            status, reason, reward = "infra_invalid", "cleanup_failed", None
    result = EpisodeResult(
        episode_id=client.episode_id,
        task_hash=fingerprint(task.model_dump(mode="json")),
        policy_version=policy.version,
        seed=seed,
        status=status,
        reason=reason,
        reward=reward,
        receipt=receipt,
        transitions=tuple(transitions),
        verification=verification,
        elapsed_seconds=time.monotonic() - started,
        reset_seconds=reset_seconds,
    )
    if recorder:
        recorder.append("episode", result.model_dump(mode="json"))
    return result
