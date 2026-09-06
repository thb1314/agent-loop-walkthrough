"""QianChat OpenAI Responses HTTP Serve。"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from .engine import Qwen3ResponsesEngine
from .errors import BackendError, InvalidRequestError

_MAX_REQUEST_BYTES = 4 * 1024 * 1024


class QianChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: Qwen3ResponsesEngine,
    ) -> None:
        self.engine = engine
        super().__init__(server_address, QianChatHandler)


class QianChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/health"}:
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": cast(QianChatServer, self.server).engine.model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self._openai_error(HTTPStatus.NOT_FOUND, "未找到接口", "not_found_error")

    def do_POST(self) -> None:
        if self.path not in {"/responses", "/v1/responses"}:
            self._openai_error(HTTPStatus.NOT_FOUND, "未找到接口", "not_found_error")
            return
        try:
            payload = self._read_json_object()
            result = cast(QianChatServer, self.server).engine.create(payload)
        except InvalidRequestError as error:
            self._openai_error(
                HTTPStatus.BAD_REQUEST,
                str(error),
                "invalid_request_error",
                error.param,
            )
            return
        except BackendError as error:
            self._openai_error(
                HTTPStatus.BAD_GATEWAY,
                str(error),
                "server_error",
                code="ntp_backend_error",
            )
            return
        except (BrokenPipeError, ConnectionResetError):
            return
        self._json(HTTPStatus.OK, result.response)

    def _read_json_object(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise InvalidRequestError("请求缺少 Content-Length")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise InvalidRequestError("Content-Length 无效") from error
        if not 0 < length <= _MAX_REQUEST_BYTES:
            raise InvalidRequestError("请求体大小无效")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise InvalidRequestError("请求体不是合法 JSON") from error
        if not isinstance(payload, dict):
            raise InvalidRequestError("请求体必须是 JSON 对象")
        return payload

    def _openai_error(
        self,
        status: HTTPStatus,
        message: str,
        error_type: str,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self._json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": param,
                    "code": code,
                }
            },
        )

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"qianchat: {self.address_string()} - {format % args}")
