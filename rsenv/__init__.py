"""RuneScape environment; heavy legacy training exports load only on demand."""
from importlib import import_module
from typing import Any

_EXPORTS = {
    **dict.fromkeys(("BridgeClient", "BridgeClientPool", "RSClient", "SimClient"), "rsenv.client"),
    **dict.fromkeys(("compute_reward", "format_state", "load_prompt_bank", "random_starting_state", "rollout_trajectory"), "rsenv.env"),
    **dict.fromkeys(("GameState", "Trajectory"), "rsenv.state"),
    **dict.fromkeys(("TOOL_SCHEMAS", "parse_tool_call", "parse_tool_calls"), "rsenv.tools"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
