# Configuration

All configuration is done via environment variables (or a `.env` file). See [`.env.example`](../.env.example) for the full list.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PRISM_HOST` | Server bind address | `0.0.0.0` |
| `PRISM_PORT` | Server port | `9877` |
| `PRISM_API_KEY` | Legacy single-key auth (ignored when `clients.json` exists) | |
| `UPSTREAM_BASE_URL` | Upstream OpenAI-compatible endpoint URL | |
| `UPSTREAM_API_KEY` | Token for upstream authentication | |
| `UPSTREAM_TIMEOUT` | Request timeout in seconds | `300` |
| `DEFAULT_MODEL` | Fallback model name when client sends `"default"` | |
| `CLIENTS_FILE` | Path to client registry JSON | `clients.json` |
| `USAGE_DB` | Path to SQLite usage database | `prism_usage.db` |
| `CORS_ORIGINS` | Comma-separated allowed origins, or `*` for all (empty = CORS disabled) | |
| `PRISM_ADMIN_KEY` | Admin API key (empty = admin endpoints disabled) | |

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

## Usage Tracking

Every request is recorded to a SQLite database (`prism_usage.db` by default) with:
client name, model, streaming flag, input/output token counts, latency, and HTTP status.

You can query the database directly for ad-hoc analysis:

```bash
sqlite3 prism_usage.db "SELECT client, model, SUM(input_tokens), SUM(output_tokens), COUNT(*) FROM requests GROUP BY client, model"
```

Or use the [Admin API](admin-api.md) for structured queries.
