"""Normalize platform-canonical chat datasets (OpenAI-style messages / Alpaca).

ShareGPT/DADMAtools-style inputs are restructured by the platform data
ingestion module before download, so only the canonical ``messages`` format
(with structured tool calls) plus the Alpaca fallback is accepted here.
"""

from __future__ import annotations

import copy
import json
from typing import Any


class DatasetFormatError(ValueError):
    """Raised when a row cannot be converted to a chat conversation."""


ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    # Defensive: some ingestion outputs keep tool-call turns under this role
    # with the call JSON in content. Converted below to structured tool_calls.
    "function_call": "assistant",
}


def _json_value(value: Any, field_name: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"{field_name} contains invalid JSON: {exc}") from exc


def normalize_tools(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return OpenAI-style tools, accepting either ``tools`` or ``functions``."""
    raw = row.get("tools")
    came_from_functions = False
    if raw is None and row.get("functions") is not None:
        raw = row["functions"]
        came_from_functions = True
    if raw is None:
        return None

    raw = _json_value(raw, "tools/functions")
    if isinstance(raw, dict) and "functions" in raw:
        raw = raw["functions"]
        came_from_functions = True
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise DatasetFormatError("tools/functions must be a list, dict, or JSON string")

    tools: list[dict[str, Any]] = []
    for tool in raw:
        tool = _json_value(tool, "tool")
        if not isinstance(tool, dict):
            raise DatasetFormatError("every tool definition must be an object")
        if came_from_functions or ("name" in tool and "function" not in tool):
            tools.append({"type": "function", "function": copy.deepcopy(tool)})
        else:
            tools.append(copy.deepcopy(tool))
    return tools or None


def _normalize_tool_call(call: Any) -> dict[str, Any]:
    if isinstance(call, str):
        call = _json_value(call, "tool_call")
    if not isinstance(call, dict):
        raise DatasetFormatError("every tool_call must be an object")

    if "function" in call:
        normalized = copy.deepcopy(call)
        normalized.setdefault("type", "function")
    elif "name" in call:
        function = {
            "name": call["name"],
            "arguments": call.get("arguments", call.get("parameters", {})),
        }
        normalized = {"type": "function", "function": function}
        if call.get("id") is not None:
            normalized["id"] = call["id"]
    else:
        raise DatasetFormatError("tool_call must contain either function or name")

    function = normalized.get("function")
    if not isinstance(function, dict) or not function.get("name"):
        raise DatasetFormatError("tool_call.function.name is required")
    if "arguments" not in function:
        function["arguments"] = {}
    return normalized


def normalize_message(message: Any) -> dict[str, Any]:
    message = _json_value(message, "message")
    if not isinstance(message, dict):
        raise DatasetFormatError("every conversation message must be an object")

    raw_role = message.get("role")
    raw_role_name = str(raw_role).lower() if raw_role is not None else None
    role = ROLE_MAP.get(raw_role_name) if raw_role_name is not None else None
    if role is None:
        raise DatasetFormatError(
            f"unsupported or missing message role: {raw_role!r} (expected one of "
            f"{sorted(set(ROLE_MAP))}; ShareGPT-style inputs must be restructured "
            "by the platform data ingestion module first)"
        )

    content = message.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise DatasetFormatError(
            "only text conversations are supported; message content must be a string"
        )

    if raw_role_name == "function_call":
        calls = _json_value(content, "function_call content")
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list) or not calls:
            raise DatasetFormatError(
                "a function_call message must contain a JSON object or non-empty JSON list"
            )
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [_normalize_tool_call(call) for call in calls],
        }

    normalized: dict[str, Any] = {"role": role, "content": content}
    for key in ("name", "tool_call_id"):
        if message.get(key) is not None:
            normalized[key] = message[key]

    reasoning = message.get("reasoning_content", message.get("thinking"))
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise DatasetFormatError("reasoning_content/thinking must be a string")
        normalized["reasoning_content"] = reasoning

    calls = message.get("tool_calls")
    if calls is None and message.get("function_call") is not None:
        calls = [message["function_call"]]
    if calls is not None:
        calls = _json_value(calls, "tool_calls")
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list):
            raise DatasetFormatError("tool_calls must be a list, object, or JSON string")
        flattened: list[Any] = []
        for call in calls:
            if isinstance(call, str):
                parsed = _json_value(call, "tool_call")
                if isinstance(parsed, list):
                    flattened.extend(parsed)
                else:
                    flattened.append(parsed)
            else:
                flattened.append(call)
        normalized["tool_calls"] = [_normalize_tool_call(call) for call in flattened]

    return normalized


def normalize_example(
    row: dict[str, Any], system_prompt: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Normalize one platform-canonical (OpenAI-chat) or Alpaca record."""
    if not isinstance(row, dict):
        raise DatasetFormatError("dataset row must be an object")

    raw_messages = row.get("messages")

    if raw_messages is not None:
        raw_messages = _json_value(raw_messages, "messages")
        if not isinstance(raw_messages, list):
            raise DatasetFormatError("messages must be a list or JSON string")
        messages = [normalize_message(message) for message in raw_messages]
    elif "instruction" in row and ("output" in row or "response" in row):
        instruction = str(row["instruction"])
        input_text = row.get("input")
        if input_text:
            instruction = f"{instruction}\n\n{input_text}"
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": str(row.get("response") or row.get("output") or "")},
        ]
    else:
        raise DatasetFormatError(
            "unsupported row schema; expected messages or instruction+output "
            "(ShareGPT-style conversations must be restructured by the platform "
            "data ingestion module first)"
        )

    if not messages:
        raise DatasetFormatError("conversation is empty")
    if system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages, normalize_tools(row)


def without_reasoning(message: dict[str, Any]) -> dict[str, Any]:
    """Return a non-thinking assistant target while preserving any tool calls."""
    result = copy.deepcopy(message)
    result.pop("reasoning_content", None)
    content = result.get("content", "")
    if "</think>" in content:
        content = content.split("</think>", 1)[1].lstrip("\n")
    elif content.lstrip().startswith("<think>"):
        # An unterminated reasoning trace is unsafe as a non-thinking target.
        content = ""
    result["content"] = content
    return result


def final_inference_case(
    row: dict[str, Any], system_prompt: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Split one test row into its prompt history and final expected assistant turn."""
    messages, tools = normalize_example(row, system_prompt)
    assistant_positions = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if assistant_positions:
        target_index = assistant_positions[-1]
        return messages[:target_index], messages[target_index], tools
    return messages, None, tools
