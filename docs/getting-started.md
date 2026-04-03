# Getting Started

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — fill in UPSTREAM_BASE_URL, UPSTREAM_API_KEY, and other values

# 3. Set up client keys
cp clients.example.json clients.json
# Edit clients.json — add your agent clients (see Configuration docs)

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
