# Prism

An API gateway that accepts Anthropic-style API calls, translates them to OpenAI-compatible requests, and forwards them to configurable OpenAI-compatible model endpoints.

## Features

- **Protocol Translation** — Converts Anthropic Messages API format to OpenAI Chat Completions format
- **Endpoint Routing** — Forwards translated requests to configurable upstream OpenAI-compatible endpoints
- **Streaming Support** — Handles both streaming and non-streaming responses
- **Multi-Client Access Control** — Per-client API keys with model permissions
- **Rate Limiting** — Per-client sliding-window rate limiter (requests/minute)
- **Usage Tracking** — SQLite-backed request logging with token counts and latency
- **Admin API** — Query usage stats and manage clients via REST endpoints

## Prerequisites

- Python 3.11+

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — fill in UPSTREAM_BASE_URL, UPSTREAM_API_KEY, and other values

# 3. Set up client keys
cp clients.example.json clients.json
# Edit clients.json — add your agent clients (see Client Setup below)

# 4. Start
python -m prism
```

The server starts on `http://0.0.0.0:9877` by default (configurable via `PRISM_HOST` / `PRISM_PORT` in `.env`).

## Usage

Point any Anthropic SDK client at Prism using the client's own key:

```python
import anthropic

client = anthropic.Anthropic(
    api_key="pk-pip-xxxxxxxxxxxx",       # Client's Prism key from clients.json
    base_url="http://localhost:9877",    # Prism address (no /v1 suffix)
)

response = client.messages.create(
    model="qwen-14b-chat",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)
```

## Client Setup

Each agent application gets its own API key via `clients.json`:

```json
{
  "clients": [
    {
      "name": "pip",
      "key": "pk-pip-xxxxxxxxxxxx",
      "allowed_models": ["*"],
      "rate_limit": 0
    },
    {
      "name": "another-agent",
      "key": "pk-agent2-xxxxxxxxxx",
      "allowed_models": ["claude-sonnet-4-6"],
      "rate_limit": 30
    }
  ]
}
```

| Field | Description |
|---|---|
| `name` | Unique identifier for the client, used in logs and usage tracking |
| `key` | API key the client sends via `x-api-key` header (convention: `pk-<name>-<random>`) |
| `allowed_models` | List of permitted model names; `["*"]` allows all models |
| `rate_limit` | Max requests per minute; `0` means unlimited |

**Rate limiting:** When a client exceeds its `rate_limit`, the API returns HTTP 429 with an Anthropic-style `rate_limit_error`. The window is a sliding 60-second counter that resets automatically.

**Backward compatibility:** If `clients.json` does not exist, Prism falls back to the single `PRISM_API_KEY` from `.env`. If neither is set, authentication is disabled.

## Configuration

All configuration is done via environment variables (or a `.env` file). See [`.env.example`](.env.example) for the full list:

| Variable | Description | Default |
|---|---|---|
| `PRISM_HOST` | Server bind address | `0.0.0.0` |
| `PRISM_PORT` | Server port | `9877` |
| `PRISM_API_KEY` | Legacy single-key auth (ignored when `clients.json` exists) | |
| `UPSTREAM_BASE_URL` | Upstream OpenAI-compatible endpoint URL | |
| `UPSTREAM_API_KEY` | Token for upstream authentication | |
| `DEFAULT_MODEL` | Fallback model name when client sends `"default"` | |
| `CLIENTS_FILE` | Path to client registry JSON | `clients.json` |
| `USAGE_DB` | Path to SQLite usage database | `prism_usage.db` |
| `PRISM_ADMIN_KEY` | Admin API key (empty = admin endpoints disabled) | |

## Usage Tracking

Every request is recorded to a SQLite database (`prism_usage.db` by default) with:
client name, model, streaming flag, input/output token counts, latency, and HTTP status.

You can query the database directly for ad-hoc analysis:

```bash
sqlite3 prism_usage.db "SELECT client, model, SUM(input_tokens), SUM(output_tokens), COUNT(*) FROM requests GROUP BY client, model"
```

Or use the Admin API (see below) for structured queries.

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/messages` | Client key | Anthropic Messages API (translated to OpenAI upstream) |
| `GET` | `/health` | None | Health check |
| `GET` | `/docs` | None | Interactive API documentation (Swagger UI) |
| `GET` | `/admin/usage` | Admin key | Paginated request log with filters |
| `GET` | `/admin/usage/summary` | Admin key | Aggregated usage statistics |
| `GET` | `/admin/clients` | Admin key | List all registered clients |
| `POST` | `/admin/clients` | Admin key | Register a new client |
| `DELETE` | `/admin/clients/{name}` | Admin key | Revoke a client |
| `POST` | `/admin/clients/{name}/rotate-key` | Admin key | Rotate a client's API key |
| `POST` | `/admin/clients/reload` | Admin key | Hot-reload client registry from disk |

## Admin API

The admin API is enabled by setting `PRISM_ADMIN_KEY` in `.env`. All admin endpoints require `Authorization: Bearer <admin-key>`.

### Query usage logs

```bash
# Recent requests (paginated)
curl -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/usage?limit=20"

# Filter by client and date range
curl -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/usage?client=pip&start=2025-01-01T00:00:00Z"

# Aggregated summary grouped by client
curl -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/usage/summary?group_by=client"

# Summary grouped by day
curl -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/usage/summary?group_by=day"
```

### Manage clients

```bash
# List all clients (keys are masked)
curl -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/clients"

# Register a new client (key is auto-generated)
curl -X POST -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "new-agent", "allowed_models": ["*"], "rate_limit": 60}' \
  "http://localhost:9877/admin/clients"

# Rotate a client's key
curl -X POST -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/clients/pip/rotate-key"

# Remove a client
curl -X DELETE -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/clients/pip"

# Hot-reload after manual edits to clients.json
curl -X POST -H "Authorization: Bearer $ADMIN_KEY" \
  "http://localhost:9877/admin/clients/reload"
```

## Testing

Unit tests (no server needed):

```bash
PYTHONPATH=src python -m pytest tests/test_translator.py tests/test_clients.py tests/test_usage.py tests/test_admin.py -v
```

End-to-end and SDK tests (requires a running Prism server):

```bash
# Terminal 1 — start the server
PYTHONPATH=src python -m prism

# Terminal 2 — run tests
PYTHONPATH=src \
  PRISM_TEST_URL=http://127.0.0.1:9877 \
  PRISM_TEST_KEY=your-client-key \
  PRISM_TEST_MODEL=your-model-name \
  python -m pytest tests/ -v -s
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
