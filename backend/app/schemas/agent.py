"""Response schemas for POST /agent/chat."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentProduct(BaseModel):
    """A product the agent's tools returned (real catalog data, not model prose)."""

    sku: str
    name: str
    price: float
    category: str
    color: str | None = None
    score: float | None = Field(None, description="Similarity/recommendation score, when the tool provided one.")


class AgentChatResponse(BaseModel):
    reply: str
    products: list[AgentProduct]
    session_id: str = Field(..., description="Send this back to continue the conversation.")
    steps: list[dict[str, Any]] | None = Field(None, description="Tool trace; only when AGENT_INCLUDE_TRACE is on.")
