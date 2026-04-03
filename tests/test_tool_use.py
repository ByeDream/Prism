"""Unit tests for tool use / function calling translation."""

from __future__ import annotations

import json

import pytest

from prism.translator.models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicTextBlock,
    AnthropicToolChoice,
    AnthropicToolDefinition,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    OpenAIChoice,
    OpenAIFunctionCall,
    OpenAIMessage,
    OpenAIResponse,
    OpenAIStreamChunk,
    OpenAIStreamChoice,
    OpenAIToolCall,
    OpenAIUsage,
)
from prism.translator.request import translate_request
from prism.translator.response import translate_response
from prism.translator.streaming import StreamUsage, translate_stream


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sse_bytes(chunks: list[dict]) -> list[bytes]:
    """Build raw SSE byte-chunks the way an OpenAI-compatible endpoint would."""
    result: list[bytes] = []
    for c in chunks:
        result.append(f"data: {json.dumps(c)}\n\n".encode())
    result.append(b"data: [DONE]\n\n")
    return result


async def _collect_stream(raw_chunks: list[bytes], model: str = "m") -> tuple[list[str], StreamUsage]:
    usage = StreamUsage()

    async def _iter():
        for c in raw_chunks:
            yield c

    events: list[str] = []
    async for event in translate_stream(_iter(), model=model, usage=usage):
        events.append(event)
    return events, usage


# ---------------------------------------------------------------------------
# Request translation — tool definitions
# ---------------------------------------------------------------------------

class TestToolDefinitionTranslation:
    def test_single_tool(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            tools=[
                AnthropicToolDefinition(
                    name="get_weather",
                    description="Get weather for a city",
                    input_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            ],
        )
        oai = translate_request(req)
        assert oai.tools is not None
        assert len(oai.tools) == 1
        t = oai.tools[0]
        assert t.type == "function"
        assert t.function.name == "get_weather"
        assert t.function.description == "Get weather for a city"
        assert t.function.parameters["properties"]["city"]["type"] == "string"

    def test_multiple_tools(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            tools=[
                AnthropicToolDefinition(name="a", description="desc_a", input_schema={}),
                AnthropicToolDefinition(name="b", description="desc_b", input_schema={}),
            ],
        )
        oai = translate_request(req)
        assert len(oai.tools) == 2
        assert oai.tools[0].function.name == "a"
        assert oai.tools[1].function.name == "b"

    def test_no_tools_yields_none(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
        )
        oai = translate_request(req)
        assert oai.tools is None


# ---------------------------------------------------------------------------
# Request translation — tool_choice
# ---------------------------------------------------------------------------

class TestToolChoiceTranslation:
    def test_auto(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            tool_choice=AnthropicToolChoice(type="auto"),
        )
        oai = translate_request(req)
        assert oai.tool_choice == "auto"

    def test_any_becomes_required(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            tool_choice=AnthropicToolChoice(type="any"),
        )
        oai = translate_request(req)
        assert oai.tool_choice == "required"

    def test_specific_tool(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
            tool_choice=AnthropicToolChoice(type="tool", name="get_weather"),
        )
        oai = translate_request(req)
        assert oai.tool_choice.type == "function"
        assert oai.tool_choice.function.name == "get_weather"

    def test_none_tool_choice(self):
        req = AnthropicRequest(
            model="m",
            messages=[AnthropicMessage(role="user", content="Hi")],
        )
        oai = translate_request(req)
        assert oai.tool_choice is None


# ---------------------------------------------------------------------------
# Request translation — tool_result messages
# ---------------------------------------------------------------------------

class TestToolResultTranslation:
    def test_tool_result_becomes_tool_message(self):
        req = AnthropicRequest(
            model="m",
            messages=[
                AnthropicMessage(role="user", content="What's the weather?"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicToolUseBlock(
                            id="toolu_01", name="get_weather", input={"city": "London"},
                        ),
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicToolResultBlock(
                            tool_use_id="toolu_01", content="Sunny, 22°C",
                        ),
                    ],
                ),
            ],
        )
        oai = translate_request(req)
        msgs = oai.messages
        assert msgs[0].role == "user"
        assert msgs[0].content == "What's the weather?"
        # assistant with tool_calls
        assert msgs[1].role == "assistant"
        assert msgs[1].tool_calls is not None
        assert msgs[1].tool_calls[0].id == "toolu_01"
        assert msgs[1].tool_calls[0].function.name == "get_weather"
        assert json.loads(msgs[1].tool_calls[0].function.arguments) == {"city": "London"}
        # tool result
        assert msgs[2].role == "tool"
        assert msgs[2].tool_call_id == "toolu_01"
        assert msgs[2].content == "Sunny, 22°C"

    def test_tool_result_with_text_blocks(self):
        req = AnthropicRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicToolResultBlock(
                            tool_use_id="t1",
                            content=[
                                AnthropicTextBlock(text="line 1"),
                                AnthropicTextBlock(text="line 2"),
                            ],
                        ),
                    ],
                ),
            ],
        )
        oai = translate_request(req)
        assert oai.messages[0].role == "tool"
        assert oai.messages[0].content == "line 1\nline 2"

    def test_mixed_text_and_tool_result_in_user_message(self):
        req = AnthropicRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicTextBlock(text="Here is the result:"),
                        AnthropicToolResultBlock(tool_use_id="t1", content="42"),
                    ],
                ),
            ],
        )
        oai = translate_request(req)
        assert oai.messages[0].role == "user"
        assert oai.messages[0].content == "Here is the result:"
        assert oai.messages[1].role == "tool"
        assert oai.messages[1].content == "42"

    def test_assistant_tool_use_with_text(self):
        req = AnthropicRequest(
            model="m",
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicTextBlock(text="Let me check."),
                        AnthropicToolUseBlock(id="t1", name="search", input={"q": "test"}),
                    ],
                ),
            ],
        )
        oai = translate_request(req)
        assert oai.messages[0].role == "assistant"
        assert oai.messages[0].content == "Let me check."
        assert len(oai.messages[0].tool_calls) == 1


# ---------------------------------------------------------------------------
# Response translation — tool_calls
# ---------------------------------------------------------------------------

class TestToolCallResponseTranslation:
    def test_single_tool_call(self):
        oai = OpenAIResponse(
            id="chatcmpl-1",
            model="m",
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            OpenAIToolCall(
                                id="call_1",
                                function=OpenAIFunctionCall(
                                    name="get_weather",
                                    arguments='{"city": "London"}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        resp = translate_response(oai, model="m")
        assert resp.stop_reason == "tool_use"
        assert len(resp.content) == 1
        block = resp.content[0]
        assert block.type == "tool_use"
        assert block.id == "call_1"
        assert block.name == "get_weather"
        assert block.input == {"city": "London"}

    def test_multiple_tool_calls(self):
        oai = OpenAIResponse(
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(
                        role="assistant",
                        tool_calls=[
                            OpenAIToolCall(
                                id="c1",
                                function=OpenAIFunctionCall(name="a", arguments="{}"),
                            ),
                            OpenAIToolCall(
                                id="c2",
                                function=OpenAIFunctionCall(name="b", arguments='{"x": 1}'),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
        resp = translate_response(oai, model="m")
        assert len(resp.content) == 2
        assert resp.content[0].name == "a"
        assert resp.content[1].name == "b"
        assert resp.content[1].input == {"x": 1}

    def test_mixed_text_and_tool_calls(self):
        oai = OpenAIResponse(
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(
                        role="assistant",
                        content="Let me look that up.",
                        tool_calls=[
                            OpenAIToolCall(
                                id="c1",
                                function=OpenAIFunctionCall(name="search", arguments='{"q": "test"}'),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
        resp = translate_response(oai, model="m")
        assert len(resp.content) == 2
        assert resp.content[0].type == "text"
        assert resp.content[0].text == "Let me look that up."
        assert resp.content[1].type == "tool_use"

    def test_malformed_arguments_yields_empty_dict(self):
        oai = OpenAIResponse(
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(
                        role="assistant",
                        tool_calls=[
                            OpenAIToolCall(
                                id="c1",
                                function=OpenAIFunctionCall(name="x", arguments="not-json"),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
        resp = translate_response(oai, model="m")
        assert resp.content[0].input == {}


# ---------------------------------------------------------------------------
# Streaming — tool calls
# ---------------------------------------------------------------------------

class TestStreamingToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call_stream(self):
        raw = _make_sse_bytes([
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '{"city":'}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": ' "London"}'}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ])
        events, usage = await _collect_stream(raw)
        event_text = "".join(events)

        assert "event: content_block_start" in event_text
        assert '"type": "tool_use"' in event_text
        assert '"name": "get_weather"' in event_text
        assert "input_json_delta" in event_text
        assert "city" in event_text
        assert '"stop_reason": "tool_use"' in event_text
        assert "event: content_block_stop" in event_text

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_stream(self):
        raw = _make_sse_bytes([
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Let me "}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"content": "check."}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "search", "arguments": ""}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '{"q": "test"}'}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ])
        events, usage = await _collect_stream(raw)
        event_text = "".join(events)

        # Should have text block, then tool use block
        assert '"type": "text_delta"' in event_text
        assert "Let me " in event_text
        assert "check." in event_text
        assert '"type": "tool_use"' in event_text
        assert '"name": "search"' in event_text
        assert "input_json_delta" in event_text

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_stream(self):
        raw = _make_sse_bytes([
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "a", "arguments": ""}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "{}"}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 1, "id": "call_2", "type": "function", "function": {"name": "b", "arguments": ""}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 1, "function": {"arguments": '{"x": 1}'}}
                ]}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ])
        events, usage = await _collect_stream(raw)
        event_text = "".join(events)

        assert '"name": "a"' in event_text
        assert '"name": "b"' in event_text
        assert event_text.count("event: content_block_start") == 2
        assert event_text.count("event: content_block_stop") == 2

    @pytest.mark.asyncio
    async def test_stream_usage_populated(self):
        raw = _make_sse_bytes([
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
            },
            {
                "id": "chat-1", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
            },
        ])
        events, usage = await _collect_stream(raw)
        assert usage.input_tokens == 42
        assert usage.output_tokens == 7
