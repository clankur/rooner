"""Parse tool calls from raw model output.

Adapted from vLLM's qwen3_coder tool parser:
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/blob/main/qwen3coder_tool_parser.py

Supports both JSON and Qwen3.5 XML formats inside <tool_call> tags.
Tool arguments are validated by the rollout before execution.
"""

import json
import re
from dataclasses import dataclass

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: object
    arguments: object


def _parse_xml_function_call(function_call_str: str) -> ToolCall | None:
    end_index = function_call_str.find(">")
    if end_index == -1:
        return None
    function_name = function_call_str[:end_index]

    parameters = function_call_str[end_index + 1 :]
    param_dict: dict[str, str] = {}
    for match_text in _PARAM_RE.findall(parameters):
        idx = match_text.find(">")
        if idx == -1:
            continue
        param_name = match_text[:idx]
        param_value = match_text[idx + 1 :]
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]
        param_dict[param_name] = param_value

    return ToolCall(name=function_name, arguments=param_dict)


def _parse_one(raw: str) -> ToolCall | None:
    """Parse a single <tool_call> body as JSON or XML."""
    raw = raw.strip()

    # JSON format (Qwen3)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "name" in obj:
            return ToolCall(name=obj["name"], arguments=obj.get("arguments", {}))
    except (json.JSONDecodeError, TypeError):
        pass

    # XML format (Qwen3.5 / Qwen3-Coder)
    fm = _FUNC_RE.search(raw)
    if fm:
        return _parse_xml_function_call(fm.group(1))

    return None


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls, leaving argument validation to the rollout."""
    results: list[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        parsed = _parse_one(m.group(1))
        if parsed is not None:
            results.append(parsed)
    return results
