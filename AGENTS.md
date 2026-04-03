# AGENTS.md

## Project

Prism is an API gateway that translates Anthropic-style API calls into OpenAI-compatible requests and forwards them to configurable model endpoints.

## Public Repository Policy

This is a public GitHub repository. All generated code, comments, commit messages, and documentation MUST NOT contain:

- Personal or developer-identifying information (names, emails, usernames)
- Internal hostnames, IP addresses, or private URLs
- API keys, tokens, passwords, or any credentials
- Subjective motivations or personal context about the project

## Tech Stack

- Python 3.11+
- Async HTTP (aiohttp / httpx)
- Deployed as a standalone service on Linux

## Conventions

- Use `src/prism/` layout for all package source code
- Tests go in `tests/` at the project root
- Follow PEP 8; use type hints throughout
- Keep dependencies in `pyproject.toml`; pin versions in `requirements.txt` for deployment
- All configuration via environment variables or config files — never hardcoded
- All code comments and documentation must be written in English
