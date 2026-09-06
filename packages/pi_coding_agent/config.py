"""Coding Agent 产品层的命令行配置。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8021"
DEFAULT_MODEL = "qwen3-0.6b"
DEFAULT_MAX_READ_BYTES = 200 * 1024


@dataclass(frozen=True)
class AppConfig:
    project_dir: Path
    question: str
    base_url: str
    model: str
    allow_remote_api_key: bool
    max_turns: int
    max_read_bytes: int
    timeout: float
    max_output_tokens: int
    reasoning_effort: str
    temperature: float | None
    top_p: float | None


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="使用本地 Qwen3 Serve 的 Python mini-pi")
    parser.add_argument("project_dir", type=Path, help="允许 Agent 查看且仅允许查看的项目目录")
    parser.add_argument("question", help="需要 Agent 回答的项目问题")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QIANCHAT_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL,
        help=f"Responses API 根地址（默认：{DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--allow-remote-api-key",
        action="store_true",
        help="允许向非本地 --base-url 发送 OPENAI_API_KEY（默认拒绝）",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QIANCHAT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL,
        help=f"模型名称（默认：{DEFAULT_MODEL}）",
    )
    parser.add_argument("--max-turns", type=int, default=8, help="最大模型调用轮数（默认：8）")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="每轮最大输出 token 数，包含推理 token（默认：1024）",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default="low",
        help="OpenAI reasoning.effort（默认：low）",
    )
    parser.add_argument("--temperature", type=float, default=None, help="OpenAI temperature")
    parser.add_argument("--top-p", type=float, default=None, help="OpenAI top_p")
    parser.add_argument(
        "--max-read-bytes",
        type=int,
        default=DEFAULT_MAX_READ_BYTES,
        help=f"单个文件读取上限（默认：{DEFAULT_MAX_READ_BYTES}）",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单次 API 超时秒数")
    args = parser.parse_args(argv)
    if not 1 <= args.max_turns <= 30:
        parser.error("--max-turns 必须在 1 到 30 之间")
    if not 1 <= args.max_output_tokens <= 32768:
        parser.error("--max-output-tokens 必须在 1 到 32768 之间")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature 必须在 0 到 2 之间")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p 必须大于 0 且不超过 1")
    if args.max_read_bytes <= 0:
        parser.error("--max-read-bytes 必须大于 0")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    return AppConfig(
        project_dir=args.project_dir,
        question=args.question,
        base_url=args.base_url,
        model=args.model,
        allow_remote_api_key=args.allow_remote_api_key,
        max_turns=args.max_turns,
        max_read_bytes=args.max_read_bytes,
        timeout=args.timeout,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
    )
