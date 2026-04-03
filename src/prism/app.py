"""FastAPI application — exposes the Anthropic Messages API endpoint."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from prism.auth import verify_api_key
from prism.translator.models import (
    AnthropicErrorBody,
    AnthropicErrorResponse,
    AnthropicRequest,
    OpenAIResponse,
)
from prism.translator.request import translate_request
from prism.translator.response import translate_response
from prism.translator.streaming import translate_stream
from prism.upstream.client import send_request, send_stream

logger = logging.getLogger("prism")

app = FastAPI(title="Prism", version="0.1.0")


def _error_json(status: int, error_type: str, message: str) -> JSONResponse:
    body = AnthropicErrorResponse(
        error=AnthropicErrorBody(type=error_type, message=message),
    )
    return JSONResponse(status_code=status, content=body.model_dump())


@app.post("/v1/messages", dependencies=[Depends(verify_api_key)])
async def create_message(body: AnthropicRequest, request: Request):
    oai_req = translate_request(body)
    payload = oai_req.model_dump(exclude_none=True)

    if body.stream:
        try:
            raw_stream = send_stream(payload)
            sse = translate_stream(raw_stream, model=body.model)
            return StreamingResponse(
                sse,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        except Exception as exc:
            logger.exception("Streaming request failed")
            return _error_json(502, "api_error", str(exc))

    try:
        resp = await send_request(payload)
    except Exception as exc:
        logger.exception("Upstream request failed")
        return _error_json(502, "api_error", f"Upstream error ({type(exc).__name__}): {exc or 'timeout'}")

    if resp.status_code >= 400:
        return _error_json(
            resp.status_code,
            "api_error",
            f"Upstream returned {resp.status_code}: {resp.text[:500]}",
        )

    oai_resp = OpenAIResponse.model_validate_json(resp.content)
    anthro_resp = translate_response(oai_resp, model=body.model)
    return JSONResponse(content=anthro_resp.model_dump())


@app.get("/health")
async def health():
    return {"status": "ok"}
