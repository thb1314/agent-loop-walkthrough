"""在纯 NTP 后端之上实现 OpenAI Responses 语义。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .errors import BackendError, InvalidRequestError
from .ntp import NtpBackend, NtpRequest, NtpResult
from .openai import JsonObject, ResponsesRequest
from .qwen3 import (
    ParsedCompletion,
    parse_completion,
    parse_prefilled_tool_call,
    render_prompt,
    responses_messages,
)


@dataclass(frozen=True)
class GenerationResult:
    response: JsonObject
    reasoning_text: str


class Qwen3ResponsesEngine:
    """负责 reasoning 状态机、工具解析和 Responses 输出装配。"""

    def __init__(self, backend: NtpBackend, model: str = "qwen3-0.6b") -> None:
        self.backend = backend
        self.model = model

    def create(self, payload: JsonObject) -> GenerationResult:
        request = ResponsesRequest.parse(payload, self.model)
        if request.model != self.model:
            raise InvalidRequestError(f"未加载模型：{request.model}", "model")
        messages = responses_messages(request.instructions, request.input)
        prompt, allowed_tools, selected_tool = render_prompt(
            messages,
            request.tools,
            request.tool_choice,
            request.parallel_tool_calls,
        )
        reasoning_budget = request.reasoning_budget()
        temperature = request.effective_temperature()
        top_p = request.effective_top_p()

        reasoning_text = ""
        reasoning_tokens = 0
        prompt_tokens = 0
        if reasoning_budget:
            thinking = self.backend.complete(
                NtpRequest(
                    prompt=prompt + "<think>\n",
                    max_tokens=reasoning_budget,
                    temperature=temperature,
                    top_p=top_p,
                    stop=("</think>", "<|im_end|>"),
                )
            )
            reasoning_text = thinking.text.strip()
            reasoning_tokens = thinking.output_tokens
            prompt_tokens = thinking.prompt_tokens
            final_prompt = prompt + "<think>\n" + thinking.text + "\n</think>\n\n"
        else:
            final_prompt = prompt + "<think>\n\n</think>\n\n"

        remaining_tokens = max(1, request.max_output_tokens - reasoning_tokens)
        if selected_tool is not None:
            tool_prefix = (
                "<tool_call>\n"
                + json.dumps(
                    {"name": selected_tool},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )[:-1]
                + ',"arguments":'
            )
            final = self.backend.complete(
                NtpRequest(
                    prompt=final_prompt + tool_prefix,
                    max_tokens=remaining_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=("</tool_call>", "<|im_end|>"),
                )
            )
            parsed = ParsedCompletion(
                text="",
                reasoning="",
                tool_calls=(parse_prefilled_tool_call(selected_tool, final.text),),
            )
        else:
            final = self.backend.complete(
                NtpRequest(
                    prompt=final_prompt,
                    max_tokens=remaining_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=("<|im_end|>",),
                )
            )
            parsed = parse_completion(final.text)
        if prompt_tokens == 0:
            prompt_tokens = final.prompt_tokens
        if parsed.reasoning and not reasoning_text:
            reasoning_text = parsed.reasoning
        parsed = self._enforce_tool_contract(
            parsed,
            allowed_tools,
            selected_tool,
            request.parallel_tool_calls,
            request.tool_choice,
        )
        if not parsed.text and not parsed.tool_calls:
            raise BackendError("Qwen3 既没有返回文本，也没有生成工具调用")
        response = _responses_object(
            request=request,
            parsed=parsed,
            reasoning_tokens=reasoning_tokens,
            prompt_tokens=prompt_tokens,
            final_result=final,
        )
        return GenerationResult(response=response, reasoning_text=reasoning_text)

    @staticmethod
    def _enforce_tool_contract(
        parsed: ParsedCompletion,
        allowed_tools: set[str],
        selected_tool: str | None,
        parallel_tool_calls: bool,
        tool_choice: str | JsonObject,
    ) -> ParsedCompletion:
        calls = parsed.tool_calls
        for call in calls:
            if call.name not in allowed_tools:
                raise BackendError(f"Qwen3 请求调用未提供的工具：{call.name}")
        if selected_tool is not None and any(call.name != selected_tool for call in calls):
            raise BackendError(f"Qwen3 未遵守指定工具选择：{selected_tool}")
        if (tool_choice == "required" or selected_tool is not None) and not calls:
            raise BackendError("Qwen3 未遵守 tool_choice，未生成工具调用")
        if not parallel_tool_calls and len(calls) > 1:
            calls = calls[:1]
        return ParsedCompletion(
            text=parsed.text,
            reasoning=parsed.reasoning,
            tool_calls=tuple(calls),
        )


def _responses_object(
    request: ResponsesRequest,
    parsed: ParsedCompletion,
    reasoning_tokens: int,
    prompt_tokens: int,
    final_result: NtpResult,
) -> JsonObject:
    response_id = "resp_" + uuid.uuid4().hex
    output: list[JsonObject] = []
    if reasoning_tokens:
        output.append(
            {
                "id": "rs_" + uuid.uuid4().hex,
                "type": "reasoning",
                "summary": [],
                "status": "completed",
            }
        )
    if parsed.text:
        output.append(
            {
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": parsed.text,
                        "annotations": [],
                    }
                ],
            }
        )
    for call in parsed.tool_calls:
        output.append(
            {
                "id": "fc_" + uuid.uuid4().hex,
                "type": "function_call",
                "call_id": "call_" + uuid.uuid4().hex,
                "name": call.name,
                "arguments": json.dumps(
                    call.arguments, ensure_ascii=False, separators=(",", ":")
                ),
                "status": "completed",
            }
        )

    output_tokens = reasoning_tokens + final_result.output_tokens
    status = "incomplete" if final_result.stop_type == "limit" else "completed"
    result: JsonObject = {
        "id": response_id,
        "object": "response",
        "created_at": round(time.time()),
        "status": status,
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
        "instructions": request.instructions or None,
        "model": request.model,
        "output": output,
        "output_text": parsed.text,
        "parallel_tool_calls": request.parallel_tool_calls,
        "tool_choice": request.tool_choice,
        "tools": request.tools,
        "temperature": request.effective_temperature(),
        "top_p": request.effective_top_p(),
        "max_output_tokens": request.max_output_tokens,
        "reasoning": {"effort": request.reasoning_effort},
        "store": False,
        "metadata": request.metadata,
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": prompt_tokens + output_tokens,
        },
    }
    return result
