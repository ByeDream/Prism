# Prism

An API gateway that accepts Anthropic-style API calls, translates them to OpenAI-compatible requests, and forwards them to configurable OpenAI-compatible model endpoints.

## Features

- **Protocol Translation** — Converts Anthropic Messages API format to OpenAI Chat Completions format
- **Tool Use / Function Calling** — Full bidirectional translation of tool definitions, tool calls, and tool results
- **Endpoint Routing** — Forwards translated requests to configurable upstream OpenAI-compatible endpoints
- **Streaming Support** — Handles both streaming and non-streaming responses (including streamed tool calls)
- **Multi-Client Access Control** — Per-client API keys with model permissions
- **Rate Limiting** — Per-client sliding-window rate limiter (requests/minute)
- **Usage Tracking** — SQLite-backed request logging with token counts and latency
- **Admin API** — Query usage stats and manage clients via REST endpoints

## Prerequisites

- Python 3.11+

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Admin API](docs/admin-api.md)
- [Testing](docs/testing.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
