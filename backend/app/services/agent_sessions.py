"""Conversation history for the agent, kept in Redis.

Only plain user/assistant text is stored (never tool traffic or image
bytes), capped at the last `max_messages`, and expiring after `ttl_seconds`
of inactivity. The Redis client is synchronous, so callers on the event
loop should run these methods in a threadpool.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from app.config import AGENT_HISTORY_MAX_MESSAGES, AGENT_SESSION_TTL_SECONDS


class RedisLike(Protocol):
    def get(self, key: str) -> Any: ...
    def set(self, key: str, value: str, ex: int | None = None) -> Any: ...


class SessionStore:
    def __init__(
        self,
        redis_client: RedisLike,
        ttl_seconds: int = AGENT_SESSION_TTL_SECONDS,
        max_messages: int = AGENT_HISTORY_MAX_MESSAGES,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._max_messages = max_messages

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agent:session:{session_id}"

    def load(self, session_id: str) -> list[dict[str, str]]:
        raw = self._redis.get(self._key(session_id))
        if not raw:
            return []
        try:
            history = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]

    def append_turn(self, session_id: str, history: list[dict[str, str]], user_message: str, reply: str) -> None:
        """Save `history` + this turn, trimmed to the last `max_messages`; refreshes the TTL."""
        updated = [*history, {"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        self._redis.set(self._key(session_id), json.dumps(updated[-self._max_messages :]), ex=self._ttl)
