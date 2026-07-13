from threading import Lock

from fastapi import HTTPException

from api.auth import AuthenticatedUser
from core.config import settings


_chat_counts: dict[str, int] = {}
_lock = Lock()


def _key_for(user: AuthenticatedUser) -> str:
    return user.get("sub") or user.get("email") or "unknown"


def get_chat_quota(user: AuthenticatedUser) -> dict[str, int]:
    limit = max(settings.chat_rate_limit, 0)
    key = _key_for(user)
    with _lock:
        used = _chat_counts.get(key, 0)
    return {
        "limit": limit,
        "used": used,
        "remaining": max(limit - used, 0),
    }


def consume_chat_quota(user: AuthenticatedUser) -> dict[str, int]:
    limit = max(settings.chat_rate_limit, 0)
    key = _key_for(user)

    with _lock:
        used = _chat_counts.get(key, 0)
        if used >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Chat limit reached. You can use {limit} chats in this session.",
            )

        used += 1
        _chat_counts[key] = used

    return {
        "limit": limit,
        "used": used,
        "remaining": max(limit - used, 0),
    }
