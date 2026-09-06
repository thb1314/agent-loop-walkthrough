"""把项目文件系统能力包装成 Agent 工具。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from packages.errors import MiniPiError
from packages.pi_agent.tools import AgentTool

from .filesystem import ProjectFiles


def _path_parameters(description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": description,
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def _validated_path(arguments: Mapping[str, Any]) -> str:
    """Schema 给模型看，真正执行前仍需检查参数。"""
    if set(arguments) != {"path"}:
        raise MiniPiError("工具参数必须且只能包含 path")
    path = arguments["path"]
    if not isinstance(path, str) or not path.strip():
        raise MiniPiError("工具参数 path 必须是非空字符串")
    return path


def create_project_tools(files: ProjectFiles) -> list[AgentTool]:
    """创建只绑定当前项目根目录的两个只读工具。"""

    def list_files(arguments: Mapping[str, Any]) -> str:
        return files.list_files(_validated_path(arguments))

    def read_file(arguments: Mapping[str, Any]) -> str:
        return files.read_file(_validated_path(arguments))

    return [
        AgentTool(
            name="list_files",
            description="列出目录的直接子项，返回相对项目根目录的路径；目录名以 / 结尾。",
            parameters=_path_parameters("相对于项目根目录的路径；根目录使用 ."),
            handler=list_files,
        ),
        AgentTool(
            name="read_file",
            description="读取项目中的 UTF-8 文本文件。",
            parameters=_path_parameters("相对于项目根目录的文件路径"),
            handler=read_file,
        ),
    ]
