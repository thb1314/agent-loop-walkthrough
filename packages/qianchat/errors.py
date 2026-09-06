"""QianChat Serve 的请求与后端错误。"""


class QianChatError(Exception):
    """QianChat Serve 可归一化返回的错误基类。"""


class InvalidRequestError(QianChatError):
    """OpenAI 请求参数不合法或当前实现不支持。"""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


class BackendError(QianChatError):
    """纯 NTP 推理后端不可用或返回非法响应。"""
