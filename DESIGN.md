# Prism Design Document

## 1. Overview

Prism is a lightweight API translation gateway. It exposes an **Anthropic Messages API**-compatible endpoint so that client applications (e.g. the Python `anthropic` SDK, Claude Agent API) can connect to it as if it were the Anthropic API. Prism then translates each request into the **OpenAI Chat Completions** format and forwards it to an upstream OpenAI-compatible model proxy, translating the response back on return.

```
┌─────────────────────┐        ┌──────────────────────────┐        ┌─────────────────────┐
│   Client Application │        │     Prism Gateway         │        │  Upstream Model API  │
│  (anthropic SDK)     │───────>│  Anthropic -> OpenAI      │───────>│  (OpenAI-compatible) │
│                      │<───────│  OpenAI   -> Anthropic    │<───────│                      │
└─────────────────────┘        └──────────────────────────┘        └─────────────────────┘
   Anthropic Messages API         Protocol Translation            OpenAI Chat Completions
```

## 2. Tech Stack

4 direct dependencies. No extra SSE libraries — Starlette's built-in `StreamingResponse`
handles SSE output, and OpenAI's SSE format is trivially parsed manually.

- **Python 3.11+** — Project requirement
- **FastAPI + uvicorn** — Async HTTP server with Pydantic request validation
- **httpx (async)** — Async HTTP client for upstream calls with streaming support
- **pydantic-settings** — Type-safe `.env` loading (Pydantic is already a FastAPI dependency)

## 3. Project Structure

```
Prism/
  src/prism/
    __init__.py
    app.py              # FastAPI application entry point
    config.py           # Pydantic Settings — loads .env
    server.py           # Uvicorn runner / CLI
    auth.py             # Incoming request authentication
    translator/
      __init__.py
      request.py        # Anthropic request -> OpenAI request
      response.py       # OpenAI response -> Anthropic response
      streaming.py      # SSE stream translation (OpenAI SSE -> Anthropic SSE)
      models.py         # Pydantic models for both API schemas
    upstream/
      __init__.py
      client.py         # httpx async client for upstream calls
  tests/
  .env.example          # Template with all variables, no secrets (committed)
  .env                  # Local config (gitignored)
  pyproject.toml        # Project metadata only
  requirements.txt      # Pinned dependencies (committed)
```

## 4. Dependency Management

`requirements.txt` is the sole dependency manifest, committed to the repo with pinned
versions. `pyproject.toml` only holds project metadata (name, version, description) and
does NOT list dependencies.

## 5. Configuration

All secrets and endpoint configuration live in `.env` (gitignored).
A `.env.example` is committed as a template.

```ini
# --- Prism Server ---
PRISM_HOST=0.0.0.0
PRISM_PORT=9877
PRISM_API_KEY=              # Key that agent apps use to authenticate to Prism

# --- Upstream (Company Model Proxy) ---
UPSTREAM_BASE_URL=          # OpenAI-compatible endpoint URL
UPSTREAM_API_KEY=           # Proxy token

# --- Model Mapping ---
DEFAULT_MODEL=              # Default upstream model name
```

## 6. Deployment

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env with real values
python -m prism
```

## 7. API Translation Specification

### 7.1 Endpoint Mapping

- Prism exposes: `POST /v1/messages` (Anthropic Messages API)
- Prism calls upstream: `POST {UPSTREAM_BASE_URL}/chat/completions` (OpenAI Chat Completions)

### 7.2 Request Translation (Anthropic -> OpenAI)

- **system**: Anthropic top-level `system` param -> OpenAI `{"role": "system", ...}` message prepended to the messages array
- **messages**: Anthropic content blocks `[{"type": "text", "text": "..."}]` -> OpenAI string content
- **model**: Pass through directly (configurable alias mapping in Phase 2)
- **max_tokens**: Direct mapping
- **temperature**, **top_p**: Direct pass-through
- **stop_sequences** -> **stop**
- **stream**: Direct pass-through
- **top_k**: Dropped (not supported by OpenAI)
- **tools / tool_choice**: Phase 2

### 7.3 Response Translation (OpenAI -> Anthropic)

**Non-streaming:**

- `choices[0].message.content` (string) -> `content: [{"type": "text", "text": "..."}]`
- `choices[0].finish_reason` mapping: `"stop"` -> `"end_turn"`, `"length"` -> `"max_tokens"`
- `usage.prompt_tokens` -> `usage.input_tokens`
- `usage.completion_tokens` -> `usage.output_tokens`
- Wrapped in Anthropic response envelope with `id`, `type`, `role`, `model`, `content`, `stop_reason`, `usage`

**Streaming:**

Translate OpenAI SSE stream into Anthropic SSE event sequence:

1. `message_start` — message metadata
2. `content_block_start` — `{"type": "text", "text": ""}`
3. `content_block_delta` — `{"type": "text_delta", "text": "..."}` (one per OpenAI chunk)
4. `content_block_stop`
5. `message_delta` — `stop_reason` and final usage
6. `message_stop`

## 8. Authentication

### 8.1 Inbound (Agent App -> Prism)

Prism validates the `x-api-key` header against `PRISM_API_KEY`.
This mirrors how the Anthropic SDK sends its API key.

### 8.2 Outbound (Prism -> Upstream)

Send `UPSTREAM_API_KEY` as `Authorization: Bearer {token}`.

## 9. Error Handling

- Upstream HTTP errors (4xx/5xx) -> Anthropic-style `{"type": "error", "error": {"type": "api_error", "message": "..."}}`
- Request validation errors -> `invalid_request_error`
- Timeout and connection errors -> `overloaded_error`

## 10. Phased Implementation

**Phase 1 (MVP):**
- Project scaffolding, config, `.env`
- Non-streaming request/response translation
- Streaming translation
- Basic inbound auth
- Upstream client with token auth

**Phase 2 (Enhancements):**
- Tool use / function calling translation
- Model name alias mapping table
- Image/vision content block support
- Logging and observability
