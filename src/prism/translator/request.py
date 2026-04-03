"""Translate Anthropic Messages API requests to OpenAI Chat Completions format."""

from __future__ import annotations

from prism.config import settings
from prism.translator.models import (
    AnthropicRequest,
    AnthropicTextBlock,
    OpenAIMessage,
    OpenAIRequest,
)


def _flatten_content(content: str | list[AnthropicTextBlock]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content)


def translate_request(req: AnthropicRequest) -> OpenAIRequest:
    """Convert an Anthropic Messages request into an OpenAI Chat Completions request."""
    messages: list[OpenAIMessage] = []

    if req.system:
        system_text = _flatten_content(req.system)
        messages.append(OpenAIMessage(role="system", content=system_text))

    for msg in req.messages:
        messages.append(
            OpenAIMessage(role=msg.role, content=_flatten_content(msg.content))
        )

    model = req.model
    if settings.default_model and model in ("", "default"):
        model = settings.default_model

    return OpenAIRequest(
        model=model,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop_sequences,
        stream=req.stream,
    )
