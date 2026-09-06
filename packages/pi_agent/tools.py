"""Agent core 使用的工具定义与错误归一化。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from packages.errors import MiniPiError

ToolHandler = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class ToolResult:
    """一次工具执行的模型可见结果。"""

    content: str
    is_error: bool = False


@dataclass(frozen=True)
class AgentTool:
    """与具体产品无关的 Agent 工具定义。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            return ToolResult(self.handler(arguments))
        except (MiniPiError, OSError, RuntimeError, ValueError) as error:
            return ToolResult(str(error), is_error=True)
