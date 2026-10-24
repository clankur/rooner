"""Tool schemas for the live log-gathering task."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChopTree(BaseModel):
    """Chop a nearby normal tree and wait for logs."""

    tree_type: Literal["Tree"] | None = Field(None, description="Name of the tree to chop")


class WalkTo(BaseModel):
    """Walk to coordinates using pathfinding, auto-opening doors."""

    x: int = Field(..., ge=0, lt=16384, description="World X coordinate")
    z: int = Field(..., ge=0, lt=16384, description="World Z coordinate")


class EquipItem(BaseModel):
    """Equip the bronze axe from inventory."""

    item: Literal["Bronze axe"] = Field(..., description="Item name to equip")


TOOL_MODELS: dict[str, type[BaseModel]] = {
    "chopTree": ChopTree,
    "walkTo": WalkTo,
    "equipItem": EquipItem,
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
