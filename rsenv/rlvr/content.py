"""The supported content of the pinned engine, shared with the bridge."""

import json
from pathlib import Path
from typing import TypedDict


class HarvestRule(TypedDict):
    name: str
    loc_ids: list[int]
    item_id: int
    skill_index: int
    xp_tenths: int
    incidental_item_ids: list[int]


CONTENT = json.loads(Path(__file__).with_name("content.json").read_text())
ITEM_NAMES: dict[int, str] = {int(key): value for key, value in CONTENT["items"].items()}
HARVESTS: dict[int, HarvestRule] = {loc: row for row in CONTENT["harvests"] for loc in row["loc_ids"]}
SKILLS = {"attack": 0, "defence": 1, "strength": 2, "hitpoints": 3, "woodcutting": 8, "mining": 14}


def xp_for_level(level: int) -> int:
    return (sum(int(n + 300 * 2 ** (n / 10)) for n in range(1, level)) // 4) * 10


def level_for_xp(xp_tenths: int) -> int:
    # Match the pinned engine Player.ts curve (level / 10), not the standard RuneScape curve.
    points = 0
    for level in range(1, 99):
        points += int(level + 300 * 2 ** (level / 10))
        if xp_tenths < (points // 4) * 10:
            return level
    return 99
