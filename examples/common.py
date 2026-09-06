"""各阶段共用连接配置和输出方式，不隐藏阶段本身的控制流程。"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlparse

from packages.errors import MiniPiError
from packages.pi_agent import AgentEvent, AgentLoopConfig, AgentTool
from packages.pi_ai.responses import JsonObject, ResponsesClient
from packages.pi_coding_agent.prompt import SYSTEM_INSTRUCTIONS

DEFAULT_QUESTION = "帮我看看 reports 目录里的日报，告诉我这份日报的日期、订单数和营收。"
SAMPLE_PROJECT = Path(__file__).resolve().parents[1] / "sample_project"


def options(default_question: str = DEFAULT_QUESTION) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从0到1设计一个简易版AgentLoop：分步示例")
    parser.add_argument("question", nargs="?", default=default_question)
    parser.add_argument("--project", type=Path, default=SAMPLE_PROJECT)
    parser.add_argument("--max-turns", type=int, default=os.environ.get("QWEN_MAX_TURNS", "8"))
    result = parser.parse_args()
    if result.max_turns < 1:
        parser.error("--max-turns 必须大于 0")
    return result


def connect(args: argparse.Namespace) -> tuple[ResponsesClient, AgentLoopConfig]:
    base_url = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8021")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise MiniPiError("教学示例只连接本地模型：请使用回环地址的 HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise MiniPiError("不要把凭证放进 URL")
    try:
        timeout = float(os.environ.get("QWEN_TIMEOUT", "120"))
        max_tokens = int(os.environ.get("QWEN_MAX_OUTPUT_TOKENS", "2048"))
    except ValueError as error:
        raise MiniPiError("QWEN_TIMEOUT 必须为数字，QWEN_MAX_OUTPUT_TOKENS 必须为整数") from error
    if not math.isfinite(timeout) or timeout <= 0 or max_tokens <= 0:
        raise MiniPiError("请求超时和输出 token 上限必须为正数")
    reasoning = os.environ.get("QWEN_REASONING", "none")
    if reasoning not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        raise MiniPiError("QWEN_REASONING 不是受支持的值")
    client = ResponsesClient(base_url, os.environ.get("QWEN_API_KEY") or None, timeout)
    config = AgentLoopConfig(
        model=os.environ.get("QWEN_MODEL", "qwen3-0.6b"),
        instructions=SYSTEM_INSTRUCTIONS,
        max_turns=args.max_turns,
        request_options={"reasoning": {"effort": reasoning}, "max_output_tokens": max_tokens, "temperature": 0},
    )
    return client, config


def initial_history(question: str) -> list[JsonObject]:
    return [{"role": "user", "content": [{"type": "input_text", "text": question}]}]


def make_payload(config: AgentLoopConfig, history: list[JsonObject], tools: Sequence[AgentTool]) -> JsonObject:
    return {
        "model": config.model,
        "instructions": config.instructions,
        "input": history,
        "tools": [tool.schema() for tool in tools],
        "tool_choice": "auto" if tools else "none",
        "parallel_tool_calls": False,
        "store": False,
        **config.request_options,
    }


def report(event: AgentEvent) -> None:
    if event.type == "assistant_text":
        print(f"[第 {event.turn} 轮模型文本]\n{event.text}")
    elif event.type == "tool_call" and event.call is not None:
        print(f"[第 {event.turn} 轮调用] {event.call.name} {json.dumps(event.call.arguments, ensure_ascii=False)}")
        print(f"  call_id={event.call.call_id}")
    elif event.type == "tool_result" and event.result is not None:
        status = "失败" if event.result.is_error else "成功"
        print(f"[工具{status}]\n{event.result.content}")


def cli(main: Callable[[], None]) -> None:
    try:
        main()
    except KeyboardInterrupt:
        print("运行已中断", file=sys.stderr)
        raise SystemExit(130)
    except (MiniPiError, ValueError, OSError) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        raise SystemExit(1)
