"""Translate an OpenAI SSE stream into Anthropic SSE events."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from prism.translator.models import OpenAIStreamChunk

_FINISH_REASON_MAP: dict[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
    None: "end_turn",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def translate_stream(
    chunks: AsyncIterator[bytes],
    model: str,
) -> AsyncIterator[str]:
    """Yield Anthropic SSE event strings from an OpenAI SSE byte stream.

    The caller is responsible for wrapping this in a StreamingResponse.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    block_opened = False
    input_tokens = 0
    output_tokens = 0

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
                if choice.delta.content:
                    if not block_opened:
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        })
                        block_opened = True

                    output_tokens += len(choice.delta.content)
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "text_delta",
                            "text": choice.delta.content,
                        },
                    })

                if choice.finish_reason:
                    if block_opened:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": 0,
                        })

                    stop_reason = _FINISH_REASON_MAP.get(
                        choice.finish_reason, "end_turn"
                    )
                    yield _sse("message_delta", {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason},
                        "usage": {"output_tokens": output_tokens},
                    })

    if block_opened:
        pass  # content_block_stop already emitted on finish_reason

    yield _sse("message_stop", {"type": "message_stop"})
