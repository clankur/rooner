"""Bounded goal-following episodes; parsing never erases rejected attempts."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from pydantic import ValidationError

from rsenv.rlvr.content import HARVESTS, SKILLS, level_for_xp, xp_for_level
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
from rsenv.rlvr.tasks import verify_reset, verify_task
from rsenv.tool_parser import extract_tool_calls
from rsenv.tools import TOOL_MODELS


class Policy(Protocol):
    version: str

    def sample(self, messages: list[dict[str, str]], task: TaskSpec, seed: int, timeout: float) -> SampledTurn: ...


class EpisodeClient(Protocol):
    episode_id: str

    def start(self) -> Observation: ...
    def act(self, name: str, arguments: dict[str, Any], timeout: float, max_ticks: int = 35) -> ActionOutcome: ...
    def stop(self) -> None: ...


def parse_action(turn: SampledTurn, task: TaskSpec) -> ParsedAction:
    if turn.stop_reason == "context_limit":
        return ParsedAction(status="truncated", reason="context_limit")
    if turn.stop_reason in ("length", "max_tokens"):
        return ParsedAction(status="truncated", reason="generation_limit")
    calls = extract_tool_calls(turn.text)
    if not calls:
        if not turn.text.strip() or "<tool_call" in turn.text:
            return ParsedAction(status="invalid", reason="empty_or_malformed_tool_call")
        return ParsedAction(status="final", reason="model_final_answer")
    if len(calls) != 1 or turn.text.count("<tool_call>") != 1:
        return ParsedAction(status="invalid", reason="one_tool_call_required")
    name = calls[0].name
    args = calls[0].arguments
    if not isinstance(name, str) or name not in task.allowed_tools:
        return ParsedAction(status="invalid", reason="unsupported_tool")
    if not isinstance(args, dict):
        return ParsedAction(status="invalid", reason="arguments_must_be_object")
    try:
        model = TOOL_MODELS[name]
        if set(args) - model.model_fields.keys():
            raise ValueError("extra arguments")
        arguments = model.model_validate(args).model_dump(mode="json", exclude_none=True)
    except (ValidationError, ValueError, TypeError):
        return ParsedAction(status="invalid", reason="invalid_arguments")
    return ParsedAction(status="action", name=name, arguments=arguments)


def visible_state(state: Observation) -> str:
    visible = state.model_dump(
        mode="json",
        exclude={"bank", "tutorial", "sequence", "world_id", "episode_id", "runtime", "levels", "xp_tenths"},
    )
    visible["skills"] = {}
    for name, index in SKILLS.items():
        level = level_for_xp(state.xp_tenths[index])
        visible["skills"][name] = {
            "level": level,
            "xp_tenths": state.xp_tenths[index],
            "next_level_xp_tenths": xp_for_level(level + 1) if level < 99 else None,
        }
    return json.dumps(visible, sort_keys=True)


class ScriptedPolicy:
    def __init__(self, fail: bool = False, reset_probe: bool = False) -> None:
        self.fail = fail
        self.reset_probe = reset_probe
        if reset_probe:
            self.version = "reset-probe-v2"
        elif fail:
            self.version = "scripted-fail-v2"
        else:
            self.version = "scripted-reference-v2"

    def sample(self, messages: list[dict[str, str]], task: TaskSpec, seed: int, timeout: float) -> SampledTurn:
        content = messages[-1]["content"]
        if messages[-1]["role"] == "tool":
            result = json.loads(content)
            state = result["state"]
            stage = result["progress"] if task.ordered else 0
        else:
            state = json.loads(content.rsplit("\n", 1)[-1])
            stage = 0
        name, arguments = "wait", {"ticks": 10}
        inventory = state["inventory"]
        starting_state = json.loads(messages[1]["content"].rsplit("\n", 1)[-1])
        if self.fail or (self.reset_probe and len(messages) > 2):
            text = "Finished."
        else:
            if self.reset_probe:
                name, arguments = "equipItem", {"slot": inventory[0]["slot"]}
            else:
                goals = task.objectives[stage : stage + 1] if task.ordered else task.objectives
                for goal in goals:
                    if goal.kind == "acquire":
                        required = goal.quantity + sum(
                            i["count"] for i in starting_state["inventory"] if i["id"] == goal.item_id
                        )
                        if task.ordered:
                            required += sum(
                                g.quantity
                                for g in task.objectives[:stage]
                                if g.kind == "acquire" and g.item_id == goal.item_id
                            )
                        if sum(i["count"] for i in inventory if i["id"] == goal.item_id) >= required:
                            continue
                        resources = [
                            r for r in state["resources"] if HARVESTS.get(r["id"], {}).get("item_id") == goal.item_id
                        ]
                        if resources:
                            target = min(
                                resources,
                                key=lambda r: abs(r["x"] - state["position"][0]) + abs(r["z"] - state["position"][1]),
                            )
                            name, arguments = "harvest", {k: target[k] for k in ("id", "x", "z")}
                        break
                    if goal.kind == "reach":
                        if max(abs(state["position"][i] - goal.position[i]) for i in (0, 1)) <= goal.radius:
                            continue
                        name, arguments = "walkTo", {"x": goal.position[0], "z": goal.position[1]}
                        break
                    if goal.kind == "equip":
                        if any(i["id"] == goal.item_id for i in state["equipment"]):
                            continue
                        item = next((i for i in inventory if i["id"] == goal.item_id), None)
                        if item:
                            name, arguments = "equipItem", {"slot": item["slot"]}
                        break
                    if goal.kind == "skill":
                        if state["skills"][goal.skill]["level"] >= goal.level:
                            continue
                        if goal.skill in ("woodcutting", "mining"):
                            resources = [
                                r
                                for r in state["resources"]
                                if HARVESTS.get(r["id"], {}).get("skill_index") == SKILLS[goal.skill]
                            ]
                            if resources:
                                target = min(
                                    resources,
                                    key=lambda r: (
                                        abs(r["x"] - state["position"][0]) + abs(r["z"] - state["position"][1])
                                    ),
                                )
                                name, arguments = "harvest", {k: target[k] for k in ("id", "x", "z")}
                            break
                        sword = next((i for i in inventory if i["id"] == 1277), None)
                        food = next((i for i in inventory if i["id"] == 2309), None)
                        style = {"attack": 0, "strength": 1, "defence": 3}.get(goal.skill, 0)
                        npcs = [n for n in state["npcs"] if n["name"] in ("Chicken", "Cow", "Goblin")]
                        if food and state["hp"] <= 5:
                            name, arguments = "eatFood", {"slot": food["slot"]}
                        elif sword:
                            name, arguments = "equipItem", {"slot": sword["slot"]}
                        elif state["combat_style"] != style:
                            name, arguments = "setCombatStyle", {"style": style}
                        elif not state["combat_active"] and npcs:
                            target = min(
                                npcs,
                                key=lambda n: abs(n["x"] - state["position"][0]) + abs(n["z"] - state["position"][1]),
                            )
                            name, arguments = "attackNpc", {"index": target["index"]}
                        break
            text = "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
        return SampledTurn(
            prompt_ids=(),
            output_ids=(),
            logprobs=(),
            text=text,
            stop_reason="stop",
            policy_version=self.version,
            renderer="scripted-v2",
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
                "content": "You control a RuneScape character. Follow the user's goal. The game pauses while you think. "
                "Select explicit targets from resources and npcs. Use the observed skill levels and XP thresholds. "
                'Issue exactly one tool call per turn using <tool_call>{"name":"tool","arguments":{}}</tool_call>. '
                "Output only the tool call, without prose. Wait for the result before choosing another action. "
                "The environment ends the episode when the goal is complete.",
            },
            {"role": "user", "content": task.goal + "\n" + visible_state(state)},
        ]
        for index in range(task.max_turns):
            if state.tick - receipt.observation.tick >= task.game_ticks:
                status, reason = "truncated", "game_tick_limit"
                break
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
                    parsed.name,
                    parsed.arguments,
                    min(task.action_seconds, deadline - time.monotonic()),
                    min(35, task.game_ticks - (state.tick - receipt.observation.tick)),
                )
                state = outcome.after
                actions.append(outcome)
            transition = Transition(
                turn=turn, parsed=parsed, outcome=outcome, observation=state, sampling_seconds=sample_seconds
            )
            transitions.append(transition)
            if recorder:
                recorder.append("transition", {"episode_id": client.episode_id, **transition.model_dump(mode="json")})
            verification = verify_task(task, receipt, actions)
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
