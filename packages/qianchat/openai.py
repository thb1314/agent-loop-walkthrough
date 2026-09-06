"""本地 Serve 使用的 OpenAI Responses 请求子集。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from .errors import InvalidRequestError

JsonObject = dict[str, Any]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

_REASONING_BUDGETS: dict[str, int] = {
    "none": 0,
    "minimal": 64,
    "low": 128,
    "medium": 384,
    "high": 768,
    "xhigh": 1536,
}


@dataclass(frozen=True)
class ResponsesRequest:
    """已验证的 OpenAI Responses 请求；仅保留当前 Serve 实际支持的参数。"""

    model: str
    instructions: str
    input: str | list[JsonObject]
    tools: list[JsonObject]
    tool_choice: str | JsonObject
    parallel_tool_calls: bool
    max_output_tokens: int
    temperature: float | None
    top_p: float | None
    reasoning_effort: ReasoningEffort
    metadata: JsonObject

    @classmethod
    def parse(cls, payload: JsonObject, default_model: str) -> ResponsesRequest:
        supported = {
            "model",
            "instructions",
            "input",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "max_output_tokens",
            "temperature",
            "top_p",
            "reasoning",
            "store",
            "metadata",
            "stream",
        }
        unsupported = set(payload) - supported
        if unsupported:
            name = sorted(unsupported)[0]
            raise InvalidRequestError(f"当前 Serve 尚不支持参数 {name}", name)
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise InvalidRequestError("stream 必须是布尔值", "stream")
        if stream:
            raise InvalidRequestError("当前 Serve 尚不支持 stream=true", "stream")
        model = payload.get("model", default_model)
        if not isinstance(model, str) or not model.strip():
            raise InvalidRequestError("model 必须是非空字符串", "model")
        instructions = payload.get("instructions", "")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            raise InvalidRequestError("instructions 必须是字符串", "instructions")
        input_value = payload.get("input")
        if not isinstance(input_value, (str, list)):
            raise InvalidRequestError("input 必须是字符串或消息数组", "input")
        if isinstance(input_value, list) and not all(
            isinstance(item, dict) for item in input_value
        ):
            raise InvalidRequestError("input 数组中的条目必须是对象", "input")

        tools = payload.get("tools", [])
        if tools is None:
            tools = []
        if not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools):
            raise InvalidRequestError("tools 必须是对象数组", "tools")
        tool_choice = payload.get("tool_choice", "auto")
        if not isinstance(tool_choice, (str, dict)):
            raise InvalidRequestError("tool_choice 必须是字符串或对象", "tool_choice")
        if isinstance(tool_choice, str) and tool_choice not in {
            "auto",
            "none",
            "required",
        }:
            raise InvalidRequestError(
                "tool_choice 仅支持 auto、none、required 或指定函数对象",
                "tool_choice",
            )
        parallel_tool_calls = payload.get("parallel_tool_calls", True)
        if not isinstance(parallel_tool_calls, bool):
            raise InvalidRequestError(
                "parallel_tool_calls 必须是布尔值", "parallel_tool_calls"
            )

        max_output_tokens = payload.get("max_output_tokens", 1024)
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool):
            raise InvalidRequestError("max_output_tokens 必须是整数", "max_output_tokens")
        if not 1 <= max_output_tokens <= 32768:
            raise InvalidRequestError(
                "max_output_tokens 必须在 1 到 32768 之间", "max_output_tokens"
            )
        temperature = _optional_number(payload, "temperature", 0.0, 2.0)
        top_p = _optional_number(payload, "top_p", 0.0, 1.0, lower_inclusive=False)

        reasoning = payload.get("reasoning", {})
        if reasoning is None:
            reasoning = {}
        if not isinstance(reasoning, dict):
            raise InvalidRequestError("reasoning 必须是对象", "reasoning")
        unsupported_reasoning = set(reasoning) - {"effort"}
        if unsupported_reasoning:
            name = sorted(unsupported_reasoning)[0]
            raise InvalidRequestError(
                f"当前 Serve 尚不支持 reasoning.{name}", f"reasoning.{name}"
            )
        effort = reasoning.get("effort", "low")
        if not isinstance(effort, str) or effort not in _REASONING_BUDGETS:
            raise InvalidRequestError(
                "reasoning.effort 仅支持 none、minimal、low、medium、high、xhigh",
                "reasoning.effort",
            )

        store = payload.get("store", False)
        if not isinstance(store, bool):
            raise InvalidRequestError("store 必须是布尔值", "store")
        if store:
            raise InvalidRequestError("本地无状态 Serve 仅支持 store=false", "store")
        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise InvalidRequestError("metadata 必须是对象", "metadata")

        return cls(
            model=model.strip(),
            instructions=instructions,
            input=input_value,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=cast(ReasoningEffort, effort),
            metadata=metadata,
        )

    def reasoning_budget(self) -> int:
        requested = _REASONING_BUDGETS[self.reasoning_effort]
        if requested == 0:
            return 0
        final_reserve = min(64, max(1, self.max_output_tokens // 4))
        return max(0, min(requested, self.max_output_tokens - final_reserve))

    def effective_temperature(self) -> float:
        if self.temperature is not None:
            return self.temperature
        return 0.6 if self.reasoning_budget() else 0.7

    def effective_top_p(self) -> float:
        if self.top_p is not None:
            return self.top_p
        return 0.95 if self.reasoning_budget() else 0.8


def _optional_number(
    payload: JsonObject,
    name: str,
    minimum: float,
    maximum: float,
    *,
    lower_inclusive: bool = True,
) -> float | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidRequestError(f"{name} 必须是数字", name)
    number = value + 0.0
    lower_ok = number >= minimum if lower_inclusive else number > minimum
    if not lower_ok or number > maximum:
        bracket = "[" if lower_inclusive else "("
        raise InvalidRequestError(
            f"{name} 必须位于 {bracket}{minimum:g}, {maximum:g}]", name
        )
    return number
