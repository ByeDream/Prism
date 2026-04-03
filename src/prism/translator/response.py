"""Translate OpenAI Chat Completions responses to Anthropic Messages API format."""

from __future__ import annotations

import json
import uuid

from prism.translator.models import (
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicUsage,
    OpenAIResponse,
)

_FINISH_REASON_MAP: dict[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
    "tool_calls": "tool_use",
    None: "end_turn",
}


def translate_response(oai: OpenAIResponse, model: str) -> AnthropicResponse:
    """Convert an OpenAI Chat Completions response to an Anthropic Messages response."""
    choice = oai.choices[0] if oai.choices else None
    finish = choice.finish_reason if choice else None

    content: list[AnthropicResponseContentBlock] = []

    if choice and choice.message.content:
        content.append(AnthropicResponseContentBlock(type="text", text=choice.message.content))

    if choice and choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            try:
                parsed_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                parsed_input = {}
            content.append(
                AnthropicResponseContentBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=parsed_input,
                )
            )

    if not content:
        content.append(AnthropicResponseContentBlock(type="text", text=""))

    usage = AnthropicUsage(
        input_tokens=oai.usage.prompt_tokens if oai.usage else 0,
        output_tokens=oai.usage.completion_tokens if oai.usage else 0,
    )

    return AnthropicResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model,
        content=content,
        stop_reason=_FINISH_REASON_MAP.get(finish, "end_turn"),
        usage=usage,
    )
