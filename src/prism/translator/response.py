"""Translate OpenAI Chat Completions responses to Anthropic Messages API format."""

from __future__ import annotations

import uuid

from prism.translator.models import (
    AnthropicContentBlock,
    AnthropicResponse,
    AnthropicUsage,
    OpenAIResponse,
)

_FINISH_REASON_MAP: dict[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
    None: "end_turn",
}


def translate_response(oai: OpenAIResponse, model: str) -> AnthropicResponse:
    """Convert an OpenAI Chat Completions response to an Anthropic Messages response."""
    choice = oai.choices[0] if oai.choices else None
    text = (choice.message.content or "") if choice else ""
    finish = choice.finish_reason if choice else None

    usage = AnthropicUsage(
        input_tokens=oai.usage.prompt_tokens if oai.usage else 0,
        output_tokens=oai.usage.completion_tokens if oai.usage else 0,
    )

    return AnthropicResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model,
        content=[AnthropicContentBlock(text=text)],
        stop_reason=_FINISH_REASON_MAP.get(finish, "end_turn"),
        usage=usage,
    )
