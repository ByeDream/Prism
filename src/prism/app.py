"""FastAPI application — exposes the Anthropic Messages API endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from prism.auth import verify_api_key
from prism.clients import ClientInfo
from prism.config import settings
from prism.ratelimit import RateLimiter
from prism.translator.models import (
    AnthropicErrorBody,
    AnthropicErrorResponse,
    AnthropicRequest,
    OpenAIResponse,
)
from prism.translator.request import translate_request
from prism.translator.response import translate_response
from prism.translator.streaming import StreamUsage, translate_stream
from prism.upstream import client as upstream_client
from prism.upstream.client import send_request, send_stream
from prism.usage import UsageRecorder

logger = logging.getLogger("prism")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await upstream_client.startup()
    yield
    await upstream_client.shutdown()


app = FastAPI(title="Prism", version="0.2.0", lifespan=_lifespan)

if settings.cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

rate_limiter = RateLimiter()
usage_recorder = UsageRecorder(settings.usage_db)


def _error_json(status: int, error_type: str, message: str) -> JSONResponse:
    body = AnthropicErrorResponse(
        error=AnthropicErrorBody(type=error_type, message=message),
    )
    return JSONResponse(status_code=status, content=body.model_dump())


def _get_client(request: Request) -> ClientInfo:
    return getattr(request.state, "client", ClientInfo(name="anonymous", key=""))


def _log_and_record(
    client: ClientInfo,
    model: str,
    stream: bool,
    status: int,
    latency_ms: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    logger.info(
        "request | client=%-12s model=%-20s stream=%-5s status=%d latency=%dms in_tok=%d out_tok=%d",
        client.name,
        model,
        stream,
        status,
        latency_ms,
        input_tokens,
        output_tokens,
    )
    asyncio.create_task(
        usage_recorder.record(
            client=client.name,
            model=model,
            stream=stream,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            status=status,
        )
    )


@app.post("/v1/messages", dependencies=[Depends(verify_api_key)])
async def create_message(body: AnthropicRequest, request: Request):
    client = _get_client(request)
    t0 = time.monotonic()

    if not client.model_allowed(body.model):
        elapsed = int((time.monotonic() - t0) * 1000)
        _log_and_record(client, body.model, body.stream, 403, elapsed)
        return _error_json(
            403,
            "permission_error",
            f"Client '{client.name}' is not allowed to use model '{body.model}'",
        )

    if not rate_limiter.check(client.name, client.rate_limit):
        elapsed = int((time.monotonic() - t0) * 1000)
        _log_and_record(client, body.model, body.stream, 429, elapsed)
        return _error_json(429, "rate_limit_error", "Rate limit exceeded. Please retry later.")

    oai_req = translate_request(body)
    payload = oai_req.model_dump(exclude_none=True)

    if body.stream:
        try:
            raw_stream = send_stream(payload)
            stream_usage = StreamUsage()
            sse = translate_stream(raw_stream, model=body.model, usage=stream_usage)

            async def _tracked_sse():
                try:
                    async for event in sse:
                        yield event
                finally:
                    elapsed = int((time.monotonic() - t0) * 1000)
                    _log_and_record(
                        client, body.model, True, 200, elapsed,
                        stream_usage.input_tokens, stream_usage.output_tokens,
                    )

            return StreamingResponse(
                _tracked_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            _log_and_record(client, body.model, True, 502, elapsed)
            logger.exception("Streaming request failed")
            return _error_json(502, "api_error", str(exc))

    try:
        resp = await send_request(payload)
    except Exception as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        _log_and_record(client, body.model, False, 502, elapsed)
        logger.exception("Upstream request failed")
        return _error_json(502, "api_error", f"Upstream error ({type(exc).__name__}): {exc or 'timeout'}")

    if resp.status_code >= 400:
        elapsed = int((time.monotonic() - t0) * 1000)
        _log_and_record(client, body.model, False, resp.status_code, elapsed)
        return _error_json(
            resp.status_code,
            "api_error",
            f"Upstream returned {resp.status_code}: {resp.text[:500]}",
        )

    oai_resp = OpenAIResponse.model_validate_json(resp.content)
    anthro_resp = translate_response(oai_resp, model=body.model)

    input_tokens = anthro_resp.usage.input_tokens if anthro_resp.usage else 0
    output_tokens = anthro_resp.usage.output_tokens if anthro_resp.usage else 0
    elapsed = int((time.monotonic() - t0) * 1000)
    _log_and_record(client, body.model, False, 200, elapsed, input_tokens, output_tokens)

    return JSONResponse(content=anthro_resp.model_dump())


@app.get("/health")
async def health():
    return {"status": "ok"}


if settings.prism_admin_key:
    from prism.admin import admin_router

    app.include_router(admin_router)
