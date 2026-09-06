"""组合 pi-ai、pi-agent-core 与项目工具的命令行产品入口。"""

from __future__ import annotations

import json
import sys

from packages.errors import MiniPiError
from packages.pi_agent import AgentEvent, AgentLoopConfig, run_agent
from packages.pi_ai import ResponsesClient, load_api_key

from .config import parse_args
from .prompt import SYSTEM_INSTRUCTIONS
from .tools import ProjectFiles, create_project_tools


class ConsoleReporter:
    """把 Agent core 事件渲染成原有命令行日志。"""

    def __call__(self, event: AgentEvent) -> None:
        if event.type == "assistant_text":
            print(f"\n[第 {event.turn} 轮回答]\n{event.text}")
            return
        if event.type == "tool_call" and event.call is not None:
            print(
                f"[第 {event.turn} 轮工具调用] {event.call.name}"
                f"({json.dumps(event.call.arguments, ensure_ascii=False)})"
            )
            return
        if event.type == "tool_result" and event.result is not None:
            status = "错误" if event.result.is_error else "完成"
            content = event.result.content
            print(f"[工具{status}] {content[:500]}{'…' if len(content) > 500 else ''}")


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        with ProjectFiles(config.project_dir, config.max_read_bytes) as project_files:
            api_key = load_api_key(config.base_url, config.allow_remote_api_key)
            client = ResponsesClient(config.base_url, api_key, config.timeout)
            run_agent(
                client=client,
                tools=create_project_tools(project_files),
                config=AgentLoopConfig(
                    model=config.model,
                    instructions=SYSTEM_INSTRUCTIONS,
                    max_turns=config.max_turns,
                    required_tool_names=("read_file",),
                    request_options={
                        "max_output_tokens": config.max_output_tokens,
                        "reasoning": {"effort": config.reasoning_effort},
                        **(
                            {"temperature": config.temperature}
                            if config.temperature is not None
                            else {}
                        ),
                        **({"top_p": config.top_p} if config.top_p is not None else {}),
                    },
                ),
                question=config.question,
                emit=ConsoleReporter(),
            )
    except MiniPiError as error:
        print(f"mini-pi：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nmini-pi：用户中断", file=sys.stderr)
        return 130
    return 0
