"""Provider 凭证解析与远端发送策略。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from packages.errors import MiniPiError


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def load_api_key(base_url: str, allow_remote: bool = False) -> str | None:
    """本地 provider 可复用 Codex 凭证；远端发送必须显式授权。"""
    environment_key = os.environ.get("OPENAI_API_KEY")
    if not _is_loopback(base_url):
        if environment_key and not allow_remote:
            raise MiniPiError(
                "拒绝向非本地 --base-url 发送 OPENAI_API_KEY；"
                "确认目标可信后使用 --allow-remote-api-key"
            )
        return environment_key if allow_remote else None

    if environment_key:
        return environment_key

    auth_path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("OPENAI_API_KEY")
    return value if isinstance(value, str) and value else None
