"""只负责模型调用、工具执行和上下文回放的无状态 Agent 循环。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from packages.errors import MiniPiError
from packages.pi_ai.responses import (
    FunctionCall,
    JsonObject,
    ModelClient,
    output_text,
    parse_function_calls,
    response_output,
)

from .tools import AgentTool, ToolResult


@dataclass(frozen=True)
class AgentLoopConfig:
    model: str
    instructions: str
    max_turns: int
    request_options: JsonObject = field(default_factory=dict)
    required_tool_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentEvent:
    """循环向产品层发出的最小事件。"""

    type: Literal["assistant_text", "tool_call", "tool_result"]
    turn: int
    text: str = ""
    call: FunctionCall | None = None
    result: ToolResult | None = None


EventSink = Callable[[AgentEvent], None]


def _emit(sink: EventSink | None, event: AgentEvent) -> None:
    if sink is not None:
        sink(event)


def _tool_map(tools: Sequence[AgentTool]) -> dict[str, AgentTool]:
    result: dict[str, AgentTool] = {}
    for tool in tools:
        if tool.name in result:
            raise MiniPiError(f"工具名称重复：{tool.name}")
        result[tool.name] = tool
    return result


def run_agent(
    client: ModelClient,
    tools: Sequence[AgentTool],
    config: AgentLoopConfig,
    question: str,
    emit: EventSink | None = None,
) -> str:
    """运行一次无状态 Agent 循环并返回最终文本。"""
    tools_by_name = _tool_map(tools)
    unknown_required = set(config.required_tool_names) - set(tools_by_name)
    if unknown_required:
        names = "、".join(sorted(unknown_required))
        raise MiniPiError(f"必选工具未注册：{names}")
    tool_schemas = [tool.schema() for tool in tools]
    history: list[JsonObject] = [
        {"role": "user", "content": [{"type": "input_text", "text": question}]}
    ]
    required_index = 0

    for turn in range(1, config.max_turns + 1):
        tool_choice: str | JsonObject = "auto"
        if required_index < len(config.required_tool_names):
            tool_choice = {
                "type": "function",
                "name": config.required_tool_names[required_index],
            }
        request_payload: JsonObject = {
            "model": config.model,
            "instructions": config.instructions,
            "input": history,
            "tools": tool_schemas,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
            "store": False,
        }
        reserved = set(request_payload) & set(config.request_options)
        if reserved:
            names = "、".join(sorted(reserved))
            raise MiniPiError(f"模型请求参数不能覆盖 Agent 核心字段：{names}")
        request_payload.update(config.request_options)
        response = client.create(request_payload)
        output = response_output(response)
        text = output_text(output, response)
        if text:
            _emit(emit, AgentEvent(type="assistant_text", turn=turn, text=text))

        calls = parse_function_calls(output)
        # 必须先回放模型的原始 output，再追加 call_id 对应的工具结果。
        history.extend(output)
        if not calls:
            if not text:
                raise MiniPiError("模型既没有返回文本，也没有调用工具")
            return text

        for call in calls:
            _emit(emit, AgentEvent(type="tool_call", turn=turn, call=call))
            tool = tools_by_name.get(call.name)
            result = (
                tool.execute(call.arguments)
                if tool is not None
                else ToolResult(f"未知工具：{call.name}", is_error=True)
            )
            _emit(emit, AgentEvent(type="tool_result", turn=turn, call=call, result=result))
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result.content,
                }
            )
            if (
                required_index < len(config.required_tool_names)
                and call.name == config.required_tool_names[required_index]
                and not result.is_error
            ):
                required_index += 1

    raise MiniPiError(f"达到最大轮数 {config.max_turns}，模型仍未给出最终回答")
