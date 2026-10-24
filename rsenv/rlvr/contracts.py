"""Versioned, model-free live episode contracts. XP is in engine tenths."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    runtime: dict[str, float] = Field(default_factory=dict)

    def count(self, item_id: int) -> int:
        return sum(item.count for item in self.inventory if item.id == item_id)

    def canonical(self) -> dict[str, Any]:
        return self.model_dump(exclude={"episode_id", "world_id", "sequence", "tick", "runtime"}, mode="json")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ResetSpec(Contract):
    version: str = "logs-lumbridge-v1"
    position: tuple[int, int, int] = (3200, 3220, 0)
    inventory: tuple[Item, ...] = (Item(slot=0, id=1351, count=1),)
    xp_tenths: tuple[int, ...] = Field(
        default=tuple(11540 if i == 3 else 0 for i in range(21)), min_length=21, max_length=21
    )
    levels: tuple[int, ...] = Field(default=tuple(10 if i == 3 else 1 for i in range(21)), min_length=21, max_length=21)
    hp: int = 10
    tutorial: int = 1000

    @model_validator(mode="after")
    def consistent_fixture(self) -> ResetSpec:
        if (
            self.hp != self.levels[3]
            or self.hp < 1
            or min(self.xp_tenths) < 0
            or min(self.levels) < 1
            or len({i.slot for i in self.inventory}) != len(self.inventory)
            or any(i.slot >= 28 for i in self.inventory)
            or not all(0 <= c < 16384 for c in self.position[:2])
            or not 0 <= self.position[2] < 4
        ):
            raise ValueError("Inconsistent or unsupported reset fixture")
        return self


class TaskSpec(Contract):
    id: str = "gather_logs_5/v1"
    resource_id: Literal[1511] = 1511
    quantity: int = Field(default=5, ge=1, le=27)
    skill_index: Literal[8] = 8
    xp_per_item_tenths: Literal[250] = 250
    verifier_version: Literal["harvest-v1"] = "harvest-v1"
    reset: ResetSpec = ResetSpec()
    allowed_tools: tuple[Literal["chopTree", "walkTo", "equipItem"], ...] = ("chopTree", "walkTo", "equipItem")
    max_turns: int = Field(default=20, gt=0)
    episode_seconds: float = Field(default=300, gt=0)
    action_seconds: float = Field(default=30, gt=0, le=30)

    @property
    def goal(self) -> str:
        return f"Harvest {self.quantity} new normal logs from trees and retain at least {self.quantity} in inventory."


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
