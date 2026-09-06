"""OpenAI Responses API 的最小同步客户端与响应解析。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from packages.errors import MiniPiError

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class FunctionCall:
    """模型请求执行的一次函数调用。"""

    call_id: str
    name: str
    arguments: JsonObject


class ModelClient(Protocol):
    """Agent 循环依赖的最小模型调用接口。"""

    def create(self, payload: JsonObject) -> JsonObject:
        """提交一次模型请求并返回 Responses JSON 对象。"""
        raise NotImplementedError


def responses_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/responses") else f"{base}/responses"


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """不跟随重定向，避免 Authorization 被转交给其他主机。"""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class ResponsesClient:
    """使用标准库访问 OpenAI Responses API。"""

    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        self.url = responses_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.opener = urllib.request.build_opener(NoRedirectHandler())

    def create(self, payload: JsonObject) -> JsonObject:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "agent-loop-walkthrough/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1200]
            raise MiniPiError(f"Provider HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise MiniPiError(f"无法连接 provider：{error.reason}") from error
        except TimeoutError as error:
            raise MiniPiError(f"Provider 请求超时（{self.timeout:g} 秒）") from error

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as error:
            preview = raw.decode("utf-8", errors="replace")[:500]
            raise MiniPiError(f"Provider 返回了非 JSON 内容：{preview}") from error
        if not isinstance(result, dict):
            raise MiniPiError("Provider 响应不是 JSON 对象")
        if isinstance(result.get("error"), dict):
            raise MiniPiError(f"Provider 错误：{json.dumps(result['error'], ensure_ascii=False)}")
        return result


def response_output(response: JsonObject) -> list[JsonObject]:
    output = response.get("output")
    if not isinstance(output, list):
        raise MiniPiError("Responses API 响应缺少 output 数组")
    if not all(isinstance(item, dict) for item in output):
        raise MiniPiError("Responses API output 中存在非对象条目")
    return output


def output_text(output: list[JsonObject], response: JsonObject) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"output_text", "text"}:
                continue
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)
            elif isinstance(text, dict) and isinstance(text.get("value"), str):
                chunks.append(text["value"])
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def parse_function_calls(output: list[JsonObject]) -> list[FunctionCall]:
    calls: list[FunctionCall] = []
    for item in output:
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        call_id = item.get("call_id") or item.get("id")
        raw_arguments = item.get("arguments", "{}")
        if not isinstance(name, str) or not name:
            raise MiniPiError("function_call 缺少 name")
        if not isinstance(call_id, str) or not call_id:
            raise MiniPiError(f"工具 {name} 的调用缺少 call_id")
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise MiniPiError(f"工具 {name} 的 arguments 不是合法 JSON") from error
        else:
            raise MiniPiError(f"工具 {name} 的 arguments 类型无效")
        if not isinstance(arguments, dict):
            raise MiniPiError(f"工具 {name} 的 arguments 必须是对象")
        calls.append(FunctionCall(call_id=call_id, name=name, arguments=arguments))
    return calls
