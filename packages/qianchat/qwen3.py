"""Qwen3 ChatML 模板、Responses 历史转换与工具调用解析。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .errors import BackendError, InvalidRequestError

JsonObject = dict[str, Any]
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: JsonObject


@dataclass(frozen=True)
class ParsedCompletion:
    text: str
    reasoning: str
    tool_calls: tuple[ParsedToolCall, ...]


def responses_messages(instructions: str, input_value: str | list[JsonObject]) -> list[JsonObject]:
    """把 OpenAI Responses input 转换为 Qwen3 消息。"""
    messages: list[JsonObject] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    for item in input_value:
        item_type = item.get("type")
        if item_type == "function_call":
            _append_function_call(messages, item)
            continue
        if item_type == "function_call_output":
            messages.append({"role": "tool", "content": _tool_output_text(item)})
            continue
        if item_type == "reasoning":
            continue
        role = item.get("role")
        if role not in {"system", "user", "assistant", "developer"}:
            raise InvalidRequestError("input 中存在不支持的消息类型", "input")
        normalized_role = "system" if role == "developer" else role
        messages.append({"role": normalized_role, "content": _content_text(item.get("content"))})
    if not messages or messages[-1].get("role") not in {"user", "tool"}:
        raise InvalidRequestError("input 最后一项必须是用户消息或工具结果", "input")
    return messages


def render_prompt(
    messages: list[JsonObject],
    tools: list[JsonObject],
    tool_choice: str | JsonObject,
    parallel_tool_calls: bool,
) -> tuple[str, set[str], str | None]:
    """渲染 Qwen3 官方 ChatML 工具模板，并返回允许的工具集合。"""
    normalized_tools = _normalize_tools(tools)
    selected_name = _selected_tool_name(tool_choice)
    if tool_choice == "none":
        normalized_tools = []
    elif selected_name is not None:
        normalized_tools = [
            tool
            for tool in normalized_tools
            if tool["function"]["name"] == selected_name
        ]
        if not normalized_tools:
            raise InvalidRequestError(
                f"tool_choice 指定了未定义工具：{selected_name}", "tool_choice"
            )
    if tool_choice == "required" and not normalized_tools:
        raise InvalidRequestError("tool_choice=required 时 tools 不能为空", "tool_choice")

    policy: list[str] = []
    if tool_choice == "required" or selected_name is not None:
        policy.append("本轮必须调用提供的函数，不要直接回答。")
    if selected_name is not None:
        policy.append(
            "arguments 必须严格满足函数 JSON Schema。若参数描述要求相对路径，"
            "项目根目录只能使用 .，禁止使用 / 或绝对路径；其他路径只能复制最近工具结果中出现的值。"
        )
    if normalized_tools and not parallel_tool_calls:
        policy.append("本轮最多调用一个函数。")

    first_system = ""
    start_index = 0
    if messages and messages[0].get("role") == "system":
        first_system = str(messages[0].get("content") or "")
        start_index = 1
    if policy:
        first_system = "\n\n".join(part for part in [first_system, *policy] if part)

    chunks: list[str] = []
    if normalized_tools:
        chunks.append("<|im_start|>system\n")
        if first_system:
            chunks.append(first_system + "\n\n")
        chunks.append(
            "# Tools\n\nYou may call one or more functions to assist with the user query."
            "\n\nYou are provided with function signatures within <tools></tools> XML tags:"
            "\n<tools>"
        )
        for tool in normalized_tools:
            chunks.append("\n" + json.dumps(tool, ensure_ascii=False, separators=(",", ":")))
        chunks.append(
            "\n</tools>\n\nFor each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
    elif first_system:
        chunks.append(f"<|im_start|>system\n{first_system}<|im_end|>\n")

    index = start_index
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            chunks.append("<|im_start|>user")
            while index < len(messages) and messages[index].get("role") == "tool":
                chunks.append(
                    "\n<tool_response>\n"
                    + str(messages[index].get("content") or "")
                    + "\n</tool_response>"
                )
                index += 1
            chunks.append("<|im_end|>\n")
            continue
        if role in {"user", "system"}:
            chunks.append(
                f"<|im_start|>{role}\n{str(message.get('content') or '')}<|im_end|>\n"
            )
        elif role == "assistant":
            chunks.append("<|im_start|>assistant\n")
            content = str(message.get("content") or "")
            if content:
                chunks.append(content)
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function", call)
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name")
                    arguments = function.get("arguments", {})
                    if isinstance(name, str) and name:
                        if content:
                            chunks.append("\n")
                            content = ""
                        chunks.append(
                            "<tool_call>\n"
                            + json.dumps(
                                {"name": name, "arguments": _arguments_object(arguments)},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n</tool_call>"
                        )
            chunks.append("<|im_end|>\n")
        else:
            raise InvalidRequestError(f"不支持的消息角色：{role}", "input")
        index += 1

    chunks.append("<|im_start|>assistant\n")
    allowed_names = {tool["function"]["name"] for tool in normalized_tools}
    return "".join(chunks), allowed_names, selected_name


def parse_completion(raw: str) -> ParsedCompletion:
    reasoning_match = _THINK_PATTERN.search(raw)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    calls: list[ParsedToolCall] = []
    for match in _TOOL_CALL_PATTERN.finditer(raw):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise BackendError("Qwen3 输出了无法解析的工具调用 JSON") from error
        if not isinstance(parsed, dict):
            raise BackendError("Qwen3 工具调用必须是 JSON 对象")
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise BackendError("Qwen3 工具调用缺少 name")
        try:
            argument_object = _arguments_object(arguments)
        except InvalidRequestError as error:
            raise BackendError("Qwen3 工具调用的 arguments 不是 JSON 对象") from error
        calls.append(ParsedToolCall(name=name, arguments=argument_object))
    text = _THINK_PATTERN.sub("", raw)
    text = _TOOL_CALL_PATTERN.sub("", text).strip()
    return ParsedCompletion(text=text, reasoning=reasoning, tool_calls=tuple(calls))

def parse_prefilled_tool_call(name: str, raw: str) -> ParsedToolCall:
    """解析指定 tool_choice 预填充后仅由模型续写的 arguments 对象。"""
    candidate = raw.lstrip()
    try:
        arguments, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as error:
        raise BackendError("Qwen3 未生成合法的工具 arguments JSON") from error
    if not isinstance(arguments, dict):
        raise BackendError("Qwen3 工具 arguments 必须是 JSON 对象")
    remainder = candidate[end:].strip()
    if remainder not in {"", "}"}:
        raise BackendError("Qwen3 工具 arguments 后存在非法内容")
    return ParsedToolCall(name=name, arguments=arguments)


def _normalize_tools(tools: list[JsonObject]) -> list[JsonObject]:
    normalized: list[JsonObject] = []
    names: set[str] = set()
    for tool in tools:
        if tool.get("type") != "function":
            raise InvalidRequestError("当前 Serve 仅支持 function 工具", "tools")
        function = tool.get("function")
        source = function if isinstance(function, dict) else tool
        name = source.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidRequestError("function 工具缺少 name", "tools")
        if name in names:
            raise InvalidRequestError(f"工具名称重复：{name}", "tools")
        names.add(name)
        description = source.get("description", "")
        parameters = source.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(description, str) or not isinstance(parameters, dict):
            raise InvalidRequestError(f"工具 {name} 的定义无效", "tools")
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return normalized


def _selected_tool_name(tool_choice: str | JsonObject) -> str | None:
    if isinstance(tool_choice, str):
        return None
    if tool_choice.get("type") != "function":
        raise InvalidRequestError("指定工具的 tool_choice.type 必须是 function", "tool_choice")
    function = tool_choice.get("function")
    source = function if isinstance(function, dict) else tool_choice
    name = source.get("name")
    if not isinstance(name, str) or not name:
        raise InvalidRequestError("指定工具的 tool_choice 缺少 name", "tool_choice")
    return name


def _append_function_call(messages: list[JsonObject], item: JsonObject) -> None:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise InvalidRequestError("function_call 缺少 name", "input")
    call = {"function": {"name": name, "arguments": item.get("arguments", {})}}
    if messages and messages[-1].get("role") == "assistant":
        calls = messages[-1].setdefault("tool_calls", [])
        if isinstance(calls, list):
            calls.append(call)
            return
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})


def _tool_output_text(item: JsonObject) -> str:
    output = item.get("output", "")
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {"input_text", "output_text", "text"}:
            continue
        text = block.get("text")
        if isinstance(text, str):
            chunks.append(text)
        elif isinstance(text, dict) and isinstance(text.get("value"), str):
            chunks.append(text["value"])
    return "\n".join(chunks)


def _arguments_object(arguments: Any) -> JsonObject:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise InvalidRequestError("工具 arguments 不是合法 JSON", "input") from error
        if isinstance(parsed, dict):
            return parsed
    raise InvalidRequestError("工具 arguments 必须是 JSON 对象", "input")
