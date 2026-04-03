# Admin API

The admin API is enabled by setting `PRISM_ADMIN_KEY` in `.env`. All admin endpoints require `Authorization: Bearer <admin-key>`.

## Endpoints

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

## Query Usage Logs

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

## Manage Clients

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
