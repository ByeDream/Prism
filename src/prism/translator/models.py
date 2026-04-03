"""Pydantic models for Anthropic Messages API and OpenAI Chat Completions API."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Anthropic Messages API models (inbound)
# ---------------------------------------------------------------------------

class AnthropicTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[AnthropicTextBlock] = ""
    is_error: bool | None = None


AnthropicContentBlock = Union[AnthropicTextBlock, AnthropicToolUseBlock, AnthropicToolResultBlock]


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicToolDefinition(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolChoice(BaseModel):
    type: Literal["auto", "any", "tool"]
    name: str | None = None


class AnthropicRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    system: str | list[AnthropicTextBlock] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None
    tools: list[AnthropicToolDefinition] | None = None
    tool_choice: AnthropicToolChoice | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicResponseContentBlock(BaseModel):
    """Content block in an Anthropic response (text or tool_use)."""
    type: Literal["text", "tool_use"] = "text"
    text: str | None = None
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


class AnthropicResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicResponseContentBlock]
    stop_reason: str | None = None
    usage: AnthropicUsage


class AnthropicErrorBody(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicErrorBody


# ---------------------------------------------------------------------------
# OpenAI Chat Completions API models (outbound / upstream)
# ---------------------------------------------------------------------------

class OpenAIFunctionPayload(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class OpenAIToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionPayload


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class OpenAIToolChoiceFunction(BaseModel):
    name: str


class OpenAIToolChoiceObject(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIToolChoiceFunction


class OpenAIRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None
    stream: bool = False
    tools: list[OpenAIToolDefinition] | None = None
    tool_choice: str | OpenAIToolChoiceObject | None = None


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str | None = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    model: str = ""
    choices: list[OpenAIChoice] = Field(default_factory=list)
    usage: OpenAIUsage | None = None


# ---------------------------------------------------------------------------
# OpenAI streaming chunk models
# ---------------------------------------------------------------------------

class OpenAIStreamToolCall(BaseModel):
    index: int = 0
    id: str | None = None
    type: str | None = None
    function: OpenAIStreamFunctionCall | None = None


class OpenAIStreamFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


# Rebuild OpenAIStreamToolCall so the forward ref resolves
OpenAIStreamToolCall.model_rebuild()


class OpenAIDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[OpenAIStreamToolCall] | None = None


class OpenAIStreamChoice(BaseModel):
    index: int = 0
    delta: OpenAIDelta
    finish_reason: str | None = None


class OpenAIStreamChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    model: str = ""
    choices: list[OpenAIStreamChoice] = Field(default_factory=list)
    usage: OpenAIUsage | None = None
