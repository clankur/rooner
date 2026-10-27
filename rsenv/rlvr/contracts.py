"""Versioned, model-free live episode contracts. XP is in engine tenths."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rsenv.rlvr.content import ITEM_NAMES, SKILLS, level_for_xp
from rsenv.tools import TOOL_MODELS


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InfrastructureError(RuntimeError):
    """The episode must not contribute a gameplay reward."""


class Item(Contract):
    slot: int = Field(ge=0)
    id: int = Field(ge=0)
    count: int = Field(gt=0)


class Resource(Contract):
    id: int
    x: int
    z: int
    name: str
    kind: Literal["location", "npc"] = "location"
    index: int | None = None
    hp: int | None = None


class Observation(Contract):
    episode_id: str
    world_id: str
    sequence: int = Field(ge=1)
    tick: int = Field(ge=0)
    position: tuple[int, int, int]
    xp_tenths: tuple[int, ...]
    levels: tuple[int, ...]
    inventory: tuple[Item, ...]
    equipment: tuple[Item, ...]
    bank: tuple[Item, ...]
    hp: int
    pending: bool
    tutorial: int
    resources: tuple[Resource, ...]
    in_game: bool
    paused: bool
    npcs: tuple[Resource, ...] = ()
    combat_active: bool = False
    combat_style: int = 0
    item_names: dict[str, str] = Field(default_factory=dict)
    runtime: dict[str, float] = Field(default_factory=dict)

    def count(self, item_id: int) -> int:
        return sum(item.count for item in self.inventory if item.id == item_id)

    def canonical(self) -> dict[str, Any]:
        return self.model_dump(
            exclude={"episode_id", "world_id", "sequence", "tick", "runtime", "resources", "npcs", "item_names"},
            mode="json",
        )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ResetSpec(Contract):
    version: str = "fixture-v2"
    position: tuple[int, int, int] = (3200, 3220, 0)
    inventory: tuple[Item, ...] = (Item(slot=0, id=1351, count=1),)
    xp_tenths: tuple[int, ...] = Field(
        default=tuple(9800 if i == 3 else 0 for i in range(21)), min_length=21, max_length=21
    )
    levels: tuple[int, ...] = Field(default=tuple(10 if i == 3 else 1 for i in range(21)), min_length=21, max_length=21)
    hp: int = 10
    tutorial: int = 1000

    @model_validator(mode="after")
    def consistent_fixture(self) -> ResetSpec:
        if (
            self.hp > self.levels[3]
            or self.hp < 1
            or min(self.xp_tenths) < 0
            or min(self.levels) < 1
            or any(level_for_xp(xp) != level for xp, level in zip(self.xp_tenths, self.levels, strict=True))
            or len({i.slot for i in self.inventory}) != len(self.inventory)
            or any(i.slot >= 28 for i in self.inventory)
            or not all(0 <= c < 16384 for c in self.position[:2])
            or not 0 <= self.position[2] < 4
        ):
            raise ValueError("Inconsistent or unsupported reset fixture")
        return self


class AcquireGoal(Contract):
    kind: Literal["acquire"] = "acquire"
    item_id: int = Field(ge=0)
    quantity: int = Field(ge=1, le=28)


class SkillGoal(Contract):
    kind: Literal["skill"] = "skill"
    skill: Literal["attack", "defence", "strength", "woodcutting", "mining"]
    level: int = Field(ge=2, le=99)


class ReachGoal(Contract):
    kind: Literal["reach"] = "reach"
    position: tuple[int, int, int]
    radius: int = Field(default=1, ge=0, le=8)


class EquipGoal(Contract):
    kind: Literal["equip"] = "equip"
    item_id: int = Field(ge=0)


Goal = Annotated[AcquireGoal | SkillGoal | ReachGoal | EquipGoal, Field(discriminator="kind")]


class TaskSpec(Contract):
    id: str = "gather_logs_5/v2"
    family: str = "woodcutting"
    difficulty: Literal["control", "standard", "challenge"] = "control"
    split: Literal["smoke", "calibration", "train", "heldout"] = "smoke"
    objectives: tuple[Goal, ...] = (AcquireGoal(item_id=1511, quantity=5),)
    ordered: bool = False
    preserve_items: tuple[int, ...] = ()
    required_resource_items: tuple[int, ...] = ()
    required_npcs: tuple[str, ...] = ()
    min_final_hp: int = Field(default=1, ge=1)
    verifier_version: Literal["goals-v2"] = "goals-v2"
    reset: ResetSpec = ResetSpec()
    allowed_tools: tuple[str, ...] = (
        "harvest",
        "walkTo",
        "equipItem",
        "attackNpc",
        "setCombatStyle",
        "eatFood",
        "wait",
    )
    max_turns: int = Field(default=24, gt=0)
    game_ticks: int = Field(default=400, gt=0)
    episode_seconds: float = Field(default=900, gt=0)
    action_seconds: float = Field(default=30, gt=0, le=30)

    @model_validator(mode="after")
    def validate_goals(self) -> TaskSpec:
        if not self.allowed_tools or set(self.allowed_tools) - TOOL_MODELS.keys():
            raise ValueError("Unknown or empty task tool set")
        if not self.objectives:
            raise ValueError("A task needs at least one objective")
        for goal in self.objectives:
            if isinstance(goal, SkillGoal) and goal.level <= self.reset.levels[SKILLS[goal.skill]]:
                raise ValueError("Skill goals must require new progress")
        return self

    @property
    def goal(self) -> str:
        descriptions = []
        for goal in self.objectives:
            if isinstance(goal, AcquireGoal):
                descriptions.append(
                    f"Gather {goal.quantity} new {ITEM_NAMES.get(goal.item_id, str(goal.item_id))} and retain them in inventory"
                )
            elif isinstance(goal, SkillGoal):
                descriptions.append(f"Reach {goal.skill.title()} level {goal.level}")
            elif isinstance(goal, ReachGoal):
                descriptions.append(f"Reach within {goal.radius} tiles of {goal.position}")
            else:
                descriptions.append(f"Equip {ITEM_NAMES.get(goal.item_id, str(goal.item_id))}")
        text = ("Complete in order: " if self.ordered else "Complete all: ") + "; ".join(descriptions) + "."
        text += f" Finish alive with at least {self.min_final_hp} HP. Budget: {self.max_turns} decisions, {self.game_ticks} game ticks."
        if self.preserve_items:
            text += (
                " Preserve starting quantities of: "
                + ", ".join(ITEM_NAMES.get(i, str(i)) for i in self.preserve_items)
                + "."
            )
        return text


class ResetReceipt(Contract):
    spec_hash: str
    state_hash: str
    observation: Observation
    transition_determinism: Literal["stochastic"] = "stochastic"


class ActionOutcome(Contract):
    request_id: str
    name: str
    status: Literal["completed", "failed"]
    reason: str
    before: Observation
    after: Observation
    elapsed_seconds: float
    target: Resource | None = None


class SampledTurn(Contract):
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    text: str
    stop_reason: str
    policy_version: str
    renderer: str
    seed: int
    sampling: dict[str, Any] = Field(default_factory=dict)


class ParsedAction(Contract):
    status: Literal["action", "invalid", "final", "truncated"]
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class VerificationResult(Contract):
    success: bool
    progress: int
    reason: str
    evidence: tuple[str, ...]
    version: str


class Transition(Contract):
    turn: SampledTurn
    parsed: ParsedAction
    outcome: ActionOutcome | None
    observation: Observation
    sampling_seconds: float


class EpisodeResult(Contract):
    episode_id: str
    task_hash: str
    policy_version: str
    seed: int
    status: Literal["success", "failure", "truncated", "infra_invalid"]
    reason: str
    reward: float | None
    receipt: ResetReceipt | None
    transitions: tuple[Transition, ...]
    verification: VerificationResult | None
    elapsed_seconds: float
    reset_seconds: float
