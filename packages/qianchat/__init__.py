"""QianChat：用纯 NTP 后端封装 OpenAI Responses 协议。"""

from .ntp import LlamaCppNtpBackend, NtpBackend, NtpRequest, NtpResult
from .openai import ResponsesRequest

__all__ = [
    "LlamaCppNtpBackend",
    "NtpBackend",
    "NtpRequest",
    "NtpResult",
    "ResponsesRequest",
]
