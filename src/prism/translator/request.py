"""Translate Anthropic Messages API requests to OpenAI Chat Completions format."""

from __future__ import annotations

import json
from typing import Any

from prism.config import settings
from prism.translator.models import (
    AnthropicImageBlock,
    AnthropicRequest,
    AnthropicTextBlock,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    OpenAIFunctionCall,
    OpenAIFunctionPayload,
    OpenAIMessage,
    OpenAIRequest,
    OpenAIToolCall,
    OpenAIToolChoiceFunction,
    OpenAIToolChoiceObject,
    OpenAIToolDefinition,
)


def _flatten_content(content: str | list[AnthropicTextBlock]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content)


def _translate_tools(req: AnthropicRequest) -> list[OpenAIToolDefinition] | None:
    if not req.tools:
        return None
    return [
        OpenAIToolDefinition(
            function=OpenAIFunctionPayload(
                name=t.name,
                description=t.description,
                parameters=t.input_schema,
            )
        )
        for t in req.tools
    ]


def _translate_tool_choice(
    req: AnthropicRequest,
) -> str | OpenAIToolChoiceObject | None:
    if req.tool_choice is None:
        return None
    tc = req.tool_choice
    if tc.type == "auto":
        return "auto"
    if tc.type == "any":
        return "required"
    if tc.type == "tool" and tc.name:
        return OpenAIToolChoiceObject(function=OpenAIToolChoiceFunction(name=tc.name))
    return None


def _translate_image_block(block: AnthropicImageBlock) -> dict[str, Any]:
    """Convert an Anthropic image block to an OpenAI image_url content part."""
    src = block.source
    if src.type == "base64":
        url = f"data:{src.media_type};base64,{src.data}"
    else:
        url = src.url or ""
    return {"type": "image_url", "image_url": {"url": url}}


def _translate_messages(
    messages: list[Any],
) -> list[OpenAIMessage]:
    """Convert Anthropic message list to OpenAI message list.

    Handles text, image, tool_use (assistant), and tool_result (user) content blocks.
    """
    oai_messages: list[OpenAIMessage] = []

    for msg in messages:
        if isinstance(msg.content, str):
            oai_messages.append(OpenAIMessage(role=msg.role, content=msg.content))
            continue

        text_parts: list[str] = []
        image_parts: list[dict[str, Any]] = []
        tool_calls: list[OpenAIToolCall] = []
        tool_results: list[OpenAIMessage] = []

        for block in msg.content:
            if isinstance(block, AnthropicTextBlock):
                text_parts.append(block.text)
            elif isinstance(block, AnthropicImageBlock):
                image_parts.append(_translate_image_block(block))
            elif isinstance(block, AnthropicToolUseBlock):
                tool_calls.append(
                    OpenAIToolCall(
                        id=block.id,
                        function=OpenAIFunctionCall(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )
            elif isinstance(block, AnthropicToolResultBlock):
                result_text = block.content
                if isinstance(result_text, list):
                    result_text = "\n".join(b.text for b in result_text)
                tool_results.append(
                    OpenAIMessage(
                        role="tool",
                        content=result_text,
                        tool_call_id=block.tool_use_id,
                    )
                )

        if msg.role == "assistant":
            content = "\n".join(text_parts) if text_parts else None
            oai_messages.append(
                OpenAIMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls if tool_calls else None,
                )
            )
        else:
            if image_parts:
                parts: list[dict[str, Any]] = [
                    {"type": "text", "text": t} for t in text_parts
                ] + image_parts
                oai_messages.append(OpenAIMessage(role="user", content=parts))
            elif text_parts:
                oai_messages.append(
                    OpenAIMessage(role="user", content="\n".join(text_parts))
                )
            for tr in tool_results:
                oai_messages.append(tr)

    return oai_messages


def translate_request(req: AnthropicRequest) -> OpenAIRequest:
    """Convert an Anthropic Messages request into an OpenAI Chat Completions request."""
    messages: list[OpenAIMessage] = []

    if req.system:
        system_text = _flatten_content(req.system)
        messages.append(OpenAIMessage(role="system", content=system_text))

    messages.extend(_translate_messages(req.messages))

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
        tools=_translate_tools(req),
        tool_choice=_translate_tool_choice(req),
    )
