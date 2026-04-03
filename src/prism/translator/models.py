"""Pydantic models for Anthropic Messages API and OpenAI Chat Completions API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Anthropic Messages API models (inbound)
# ---------------------------------------------------------------------------

class AnthropicTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicTextBlock]


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


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicContentBlock]
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

class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None


class OpenAIRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None
    stream: bool = False


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

class OpenAIDelta(BaseModel):
    role: str | None = None
    content: str | None = None


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
