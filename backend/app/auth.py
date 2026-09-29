"""API-key authentication.

Responsibility: a single FastAPI dependency, `require_api_key`, that
checks the `X-API-Key` header against `API_KEY` (an env-configured
shared secret). This is the only auth this POC needs -- there are no
user accounts, just trusted clients (a storefront backend, an internal
tool) holding one shared key, so a full login/token flow would be
overhead with nothing behind it.

Using `fastapi.security.APIKeyHeader` (rather than checking the header
manually) also registers a proper security scheme in the OpenAPI
schema, so Swagger UI at `/docs` shows an "Authorize" button that lets
you set the key once and exercise protected endpoints interactively.
"""

from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import API_KEY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Reject the request unless it carries the correct `X-API-Key` header.

    Args:
        api_key: The `X-API-Key` header value, or `None` if absent
            (`auto_error=False` on the scheme lets us raise our own
            error message instead of FastAPI's generic one).

    Raises:
        HTTPException 401: If the header is missing or doesn't match
            `API_KEY`.
    """
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Send it as the 'X-API-Key' header.",
        )
