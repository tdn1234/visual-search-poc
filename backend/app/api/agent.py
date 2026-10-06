"""HTTP API layer for the local AI shopping agent.

Responsibility: HTTP concerns only -- parse the multipart form, validate
the optional image, load/save conversation history, and shape the
response. The agent loop itself is `AgentService`'s.

Requires `X-API-Key` and is rate-limited per client IP (LLM calls are
the slowest thing this service does).

`shopper_id` is trusted as sent by the (API-key-holding) storefront and
is injected into shopper-scoped tools server-side; the model can never
choose it.

No `from __future__ import annotations` in this module deliberately --
see the note in `api/search.py`.
"""

import logging
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.auth import require_api_key
from app.config import (
    AGENT_INCLUDE_TRACE,
    AGENT_MESSAGE_MAX_LENGTH,
    AGENT_RATE_LIMIT,
    SHOPPER_ID_PATTERN,
)
from app.rate_limit import limiter
from app.schemas.agent import AgentChatResponse, AgentProduct
from app.validation import read_validated_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"], dependencies=[Depends(require_api_key)])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


@router.post("/agent/chat", response_model=AgentChatResponse, response_model_exclude_none=True)
@limiter.limit(AGENT_RATE_LIMIT)
async def agent_chat(
    request: Request,
    message: str = Form(
        ..., min_length=1, max_length=AGENT_MESSAGE_MAX_LENGTH, description="What the shopper says.",
        examples=["Show me red shoes under 80 dollars"],
    ),
    file: UploadFile | None = File(None, description="Optional product photo (JPEG, PNG or WebP) to search by."),
    session_id: str | None = Form(
        None, pattern=SESSION_ID_PATTERN,
        description="Omit to start a new conversation; send back the returned `session_id` to continue it.",
    ),
    shopper_id: str | None = Form(
        None, pattern=SHOPPER_ID_PATTERN,
        description="Shopper the events were recorded under; needed for personalized recommendations.",
    ),
) -> AgentChatResponse:
    """Chat with the shopping agent, optionally with an uploaded image.

    The reply text comes from the LLM; `products` comes straight from the
    tool results, so a UI can render cards from real data. Pass the
    returned `session_id` back to keep the conversation going (history is
    kept briefly in Redis).

    Raises:
        HTTPException 401: Missing/invalid `X-API-Key`.
        HTTPException 400/413: Bad image upload (see `app.validation`).
        HTTPException 422: Invalid message / session_id / shopper_id.
        HTTPException 429: Rate limit exceeded.
        HTTPException 503: The LLM backend (Ollama) is unreachable or errored.
        HTTPException 500: Unexpected server-side failure.
    """
    agent_service = request.app.state.agent_service
    session_store = request.app.state.agent_session_store

    user_message = message
    if file is not None:
        image = await read_validated_image(file, ALLOWED_CONTENT_TYPES)
        image_ref = agent_service.add_image(image.data)
        user_message = f"[The user uploaded an image: image_ref={image_ref}]\n{message}"

    session_id = session_id or uuid.uuid4().hex
    try:
        history = await run_in_threadpool(session_store.load, session_id)
    except Exception:  # noqa: BLE001 - chat still works without memory
        logger.warning("Could not load agent session %s", session_id, exc_info=True)
        history = []

    try:
        result = await agent_service.chat(user_message, history=history, shopper_id=shopper_id)
    except httpx.HTTPError as exc:
        logger.error("LLM backend error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The AI assistant is currently unavailable."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the API layer
        logger.exception("Unexpected error in the agent")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error while running the assistant."
        ) from exc

    try:
        await run_in_threadpool(session_store.append_turn, session_id, history, user_message, result.reply)
    except Exception:  # noqa: BLE001
        logger.warning("Could not save agent session %s", session_id, exc_info=True)

    return AgentChatResponse(
        reply=result.reply,
        products=[AgentProduct(**p) for p in result.products],
        session_id=session_id,
        steps=result.steps if AGENT_INCLUDE_TRACE else None,
    )
