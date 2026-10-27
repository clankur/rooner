"""Explicit game actions; targets come from the current observation."""

from typing import Any

from pydantic import BaseModel, Field


class Harvest(BaseModel):
    """Harvest the selected visible tree or rock; stops after a yield or bounded attempt."""

    id: int = Field(ge=0, description="Location type ID from resources")
    x: int = Field(ge=0, lt=16384)
    z: int = Field(ge=0, lt=16384)


class WalkTo(BaseModel):
    """Walk to coordinates using game pathfinding; blocked routes may require another destination."""

    x: int = Field(ge=0, lt=16384)
    z: int = Field(ge=0, lt=16384)


class EquipItem(BaseModel):
    """Equip the selected inventory item."""

    slot: int = Field(ge=0, lt=28)


class AttackNpc(BaseModel):
    """Attack the selected visible NPC for up to 10 game ticks. Combat persists between turns."""

    index: int = Field(ge=0, description="NPC index from npcs")


class SetCombatStyle(BaseModel):
    """Choose melee training style: 0 Attack, 1 Strength, 2 controlled (sword) or Defence (unarmed), 3 Defence (sword/axe)."""

    style: int = Field(ge=0, le=3)


class EatFood(BaseModel):
    """Eat food from the selected inventory slot."""

    slot: int = Field(ge=0, lt=28)


class Wait(BaseModel):
    """Advance game ticks, continuing any ongoing combat. The game is paused while you think."""

    ticks: int = Field(ge=1, le=10)


TOOL_MODELS: dict[str, type[BaseModel]] = {
    "harvest": Harvest,
    "walkTo": WalkTo,
    "equipItem": EquipItem,
    "attackNpc": AttackNpc,
    "setCombatStyle": SetCombatStyle,
    "eatFood": EatFood,
    "wait": Wait,
}


def to_chat_schemas() -> list[dict[str, Any]]:
    schemas = []
    for name, model in TOOL_MODELS.items():
        parameters = model.model_json_schema()
        parameters.pop("title", None)
        schemas.append(
            {
                "type": "function",
                "function": {"name": name, "description": model.__doc__ or "", "parameters": parameters},
            }
        )
    return schemas


TOOL_SCHEMAS = to_chat_schemas()
