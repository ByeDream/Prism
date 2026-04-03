# Prism

An API gateway that accepts Anthropic-style API calls, translates them to OpenAI-compatible requests, and forwards them to configurable OpenAI-compatible model endpoints.

## Features

- **Protocol Translation** — Converts Anthropic Messages API format to OpenAI Chat Completions format
- **Endpoint Routing** — Forwards translated requests to configurable upstream OpenAI-compatible endpoints
- **Streaming Support** — Handles both streaming and non-streaming responses

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — fill in UPSTREAM_BASE_URL, UPSTREAM_API_KEY, and other values

# 3. Start
python -m prism
```

The server starts on `http://0.0.0.0:9877` by default (configurable via `PRISM_HOST` / `PRISM_PORT` in `.env`).

## Usage

Point any Anthropic SDK client at Prism:

```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-prism-api-key",      # PRISM_API_KEY value
    base_url="http://localhost:9877",   # Prism address (no /v1 suffix)
)

response = client.messages.create(
    model="qwen-14b-chat",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)
```

## Configuration

All configuration is done via environment variables (or a `.env` file). See [`.env.example`](.env.example) for the full list:

| Variable | Description | Default |
|---|---|---|
| `PRISM_HOST` | Server bind address | `0.0.0.0` |
| `PRISM_PORT` | Server port | `9877` |
| `PRISM_API_KEY` | Key clients use to authenticate to Prism (empty = auth disabled) | |
| `UPSTREAM_BASE_URL` | Upstream OpenAI-compatible endpoint URL | |
| `UPSTREAM_API_KEY` | Token for upstream authentication | |
| `DEFAULT_MODEL` | Fallback model name when client sends `"default"` | |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages API (translated to OpenAI upstream) |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
