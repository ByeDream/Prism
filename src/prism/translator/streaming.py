"""Translate an OpenAI SSE stream into Anthropic SSE events."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from prism.translator.models import OpenAIStreamChunk

_FINISH_REASON_MAP: dict[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
    "tool_calls": "tool_use",
    None: "end_turn",
}


@dataclass
class StreamUsage:
    """Mutable container that ``translate_stream`` populates as it processes chunks."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _ToolCallAccumulator:
    """Accumulates fragments for a single streaming tool call."""
    id: str = ""
    name: str = ""
    arguments: str = ""
    started: bool = False
    block_index: int = 0


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def translate_stream(
    chunks: AsyncIterator[bytes],
    model: str,
    usage: StreamUsage | None = None,
) -> AsyncIterator[str]:
    """Yield Anthropic SSE event strings from an OpenAI SSE byte stream.

    Handles both text content and tool_calls (function calling) streams.
    If a *usage* container is provided it will be populated with the final
    token counts once the stream is fully consumed.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0

    # Track content blocks: index 0 reserved for text, tool calls get index 1+
    text_block_opened = False
    # content_block_index tracks the next Anthropic content_block index to emit
    content_block_index = 0
    # Accumulate tool call fragments keyed by OpenAI's tool call index
    tool_accumulators: dict[int, _ToolCallAccumulator] = {}

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    buffer = ""
    async for raw in chunks:
        buffer += raw.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break

            try:
                chunk = OpenAIStreamChunk.model_validate_json(payload)
            except Exception:
                continue

            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            for choice in chunk.choices:
                # --- text content ---
                if choice.delta.content:
                    if not text_block_opened:
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        text_block_opened = True

                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {
                            "type": "text_delta",
                            "text": choice.delta.content,
                        },
                    })

                # --- tool calls ---
                if choice.delta.tool_calls:
                    # Close text block before first tool call
                    if text_block_opened:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        })
                        text_block_opened = False
                        content_block_index += 1

                    for tc_delta in choice.delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_accumulators:
                            tool_accumulators[idx] = _ToolCallAccumulator()

                        acc = tool_accumulators[idx]

                        if tc_delta.id:
                            acc.id = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            acc.name = tc_delta.function.name

                        if not acc.started and acc.id and acc.name:
                            acc.started = True
                            acc.block_index = content_block_index + idx
                            yield _sse("content_block_start", {
                                "type": "content_block_start",
                                "index": acc.block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": acc.id,
                                    "name": acc.name,
                                    "input": {},
                                },
                            })

                        if tc_delta.function and tc_delta.function.arguments:
                            acc.arguments += tc_delta.function.arguments
                            if acc.started:
                                yield _sse("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": acc.block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": tc_delta.function.arguments,
                                    },
                                })

                # --- finish ---
                if choice.finish_reason:
                    if text_block_opened:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        })
                        text_block_opened = False

                    for acc in tool_accumulators.values():
                        if acc.started:
                            yield _sse("content_block_stop", {
                                "type": "content_block_stop",
                                "index": acc.block_index,
                            })

                    stop_reason = _FINISH_REASON_MAP.get(
                        choice.finish_reason, "end_turn"
                    )
                    yield _sse("message_delta", {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason},
                        "usage": {"output_tokens": output_tokens},
                    })

    yield _sse("message_stop", {"type": "message_stop"})

    if usage is not None:
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
