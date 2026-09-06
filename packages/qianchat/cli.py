"""QianChat Serve 命令行入口。"""

from __future__ import annotations

import argparse
import os

from .engine import Qwen3ResponsesEngine
from .ntp import LlamaCppNtpBackend
from .server import QianChatServer

DEFAULT_BACKEND_URL = "http://127.0.0.1:8012"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8021
DEFAULT_MODEL = "qwen3-0.6b"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="在纯 NTP 后端之上提供 OpenAI Responses 兼容接口"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口")
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("QIANCHAT_NTP_BASE_URL", DEFAULT_BACKEND_URL),
        help=f"llama.cpp 原生 /completion 根地址（默认：{DEFAULT_BACKEND_URL}）",
    )
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("QIANCHAT_NTP_API_KEY"),
        help="NTP 后端 API key（默认读取 QIANCHAT_NTP_API_KEY）",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QIANCHAT_MODEL", DEFAULT_MODEL),
        help=f"对外模型名称（默认：{DEFAULT_MODEL}）",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="NTP 单次请求超时秒数")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")

    backend = LlamaCppNtpBackend(
        args.backend_url,
        timeout=args.timeout,
        api_key=args.backend_api_key,
    )
    server = QianChatServer(
        (args.host, args.port),
        Qwen3ResponsesEngine(backend, model=args.model),
    )
    print(
        f"qianchat: OpenAI Responses http://{args.host}:{args.port}/v1/responses "
        f"-> NTP {backend.url} ({args.model})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nqianchat: 用户中断")
    finally:
        server.server_close()
    return 0
