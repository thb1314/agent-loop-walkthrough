"""只依赖 next-token prediction `/completion` 的推理后端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import BackendError

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class NtpRequest:
    """一次纯文本续写请求；不包含消息、工具或推理等级语义。"""

    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    seed: int = -1
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repeat_penalty: float = 1.0


@dataclass(frozen=True)
class NtpResult:
    """NTP 后端归一化后的续写结果。"""

    text: str
    stop_type: str
    stopping_word: str
    prompt_tokens: int
    output_tokens: int
    raw: JsonObject


class NtpBackend(Protocol):
    """上层 Serve 唯一依赖的模型能力。"""

    def complete(self, request: NtpRequest) -> NtpResult:
        """根据完整 prompt 预测后续 token。"""
        raise NotImplementedError


def completion_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/completion") else f"{base}/completion"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class LlamaCppNtpBackend:
    """通过 llama.cpp 原生 `/completion` 使用纯 NTP 能力。"""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self.url = completion_url(base_url)
        self.timeout = timeout
        self.api_key = api_key
        self.opener = urllib.request.build_opener(_NoRedirectHandler())

    def complete(self, request: NtpRequest) -> NtpResult:
        payload: JsonObject = {
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "presence_penalty": request.presence_penalty,
            "repeat_penalty": request.repeat_penalty,
            "seed": request.seed,
            "stop": list(request.stop),
            "stream": False,
            "cache_prompt": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "qianchat-serve/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener.open(http_request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1200]
            raise BackendError(f"NTP 后端 HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise BackendError(f"无法连接 NTP 后端：{error.reason}") from error
        except TimeoutError as error:
            raise BackendError(f"NTP 后端请求超时（{self.timeout:g} 秒）") from error

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as error:
            preview = raw.decode("utf-8", errors="replace")[:500]
            raise BackendError(f"NTP 后端返回非 JSON 内容：{preview}") from error
        if not isinstance(result, dict):
            raise BackendError("NTP 后端响应不是 JSON 对象")
        if isinstance(result.get("error"), dict):
            raise BackendError(
                f"NTP 后端错误：{json.dumps(result['error'], ensure_ascii=False)}"
            )
        text = result.get("content")
        if not isinstance(text, str):
            raise BackendError("NTP 后端响应缺少 content 字符串")
        timings = result.get("timings")
        if not isinstance(timings, dict):
            timings = {}
        return NtpResult(
            text=text,
            stop_type=str(result.get("stop_type") or "unknown"),
            stopping_word=str(result.get("stopping_word") or ""),
            prompt_tokens=_non_negative_int(
                result.get("tokens_evaluated"), timings.get("prompt_n")
            ),
            output_tokens=_non_negative_int(
                result.get("tokens_predicted"), timings.get("predicted_n")
            ),
            raw=result,
        )


def _non_negative_int(primary: Any, fallback: Any) -> int:
    for value in (primary, fallback):
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, float) and value >= 0:
            return round(value)
    return 0

