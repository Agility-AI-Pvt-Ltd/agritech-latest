import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, TypedDict
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from core.config import settings


SESSION_COOKIE = "kisan_mitra_session"
STATE_COOKIE = "kisan_mitra_oauth_state"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


class AuthenticatedUser(TypedDict, total=False):
    sub: str
    email: str
    name: str
    picture: str


router = APIRouter(prefix="/auth", tags=["auth"])


def _cookie_kwargs() -> dict[str, Any]:
    return {
        "httponly": True,
        "secure": settings.auth_cookie_secure,
        "samesite": settings.auth_cookie_samesite,
        "max_age": SESSION_MAX_AGE_SECONDS,
        "path": "/",
    }


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _session_secret() -> bytes:
    return settings.auth_session_secret.encode("utf-8")


def _sign(payload: dict[str, Any]) -> str:
    body = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = hmac.new(_session_secret(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url_encode(signature)}"


def _unsign(value: str) -> dict[str, Any]:
    try:
        body, signature = value.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid session") from exc

    expected = hmac.new(_session_secret(), body.encode("ascii"), hashlib.sha256).digest()
    supplied = _b64url_decode(signature)
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="Invalid session")

    payload = json.loads(_b64url_decode(body))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="Session expired")
    return payload


def create_session_token(user: AuthenticatedUser) -> str:
    now = int(time.time())
    return _sign(
        {
            "sub": user["sub"],
            "email": user.get("email", ""),
            "name": user.get("name", ""),
            "picture": user.get("picture", ""),
            "iat": now,
            "exp": now + SESSION_MAX_AGE_SECONDS,
        }
    )


def get_current_user(request: Request) -> AuthenticatedUser:
    session = request.cookies.get(SESSION_COOKIE)
    if not session:
        raise HTTPException(status_code=401, detail="Login with Google to use chat")

    payload = _unsign(session)
    return {
        "sub": str(payload["sub"]),
        "email": str(payload.get("email") or ""),
        "name": str(payload.get("name") or ""),
        "picture": str(payload.get("picture") or ""),
    }


def optional_current_user(request: Request) -> AuthenticatedUser | None:
    try:
        return get_current_user(request)
    except HTTPException:
        return None


@router.get("/google/login")
def google_login() -> RedirectResponse:
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")

    state = secrets.token_urlsafe(32)
    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.google_oauth_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }

    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")
    response.set_cookie(STATE_COOKIE, state, max_age=600, httponly=True, secure=settings.auth_cookie_secure, samesite="lax", path="/")
    return response


@router.get("/google/callback")
def google_callback(request: Request) -> RedirectResponse:
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    saved_state = request.cookies.get(STATE_COOKIE)

    if not code:
        raise HTTPException(status_code=400, detail="Missing Google authorization code")
    if not state or not saved_state or not hmac.compare_digest(state, saved_state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    token_response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": settings.google_oauth_client_id,
            "client_secret": settings.google_oauth_client_secret,
            "redirect_uri": settings.google_oauth_redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=10,
    )
    if not token_response.ok:
        raise HTTPException(status_code=400, detail="Google token exchange failed")

    id_token = token_response.json().get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Google did not return an identity token")

    profile_response = requests.get(GOOGLE_TOKENINFO_URL, params={"id_token": id_token}, timeout=10)
    if not profile_response.ok:
        raise HTTPException(status_code=400, detail="Google identity verification failed")

    profile = profile_response.json()
    if profile.get("aud") != settings.google_oauth_client_id:
        raise HTTPException(status_code=400, detail="Google identity audience mismatch")
    if profile.get("email_verified") not in ("true", True):
        raise HTTPException(status_code=400, detail="Google email is not verified")

    user: AuthenticatedUser = {
        "sub": profile["sub"],
        "email": profile.get("email", ""),
        "name": profile.get("name", ""),
        "picture": profile.get("picture", ""),
    }

    response = RedirectResponse(settings.frontend_url)
    response.set_cookie(SESSION_COOKIE, create_session_token(user), **_cookie_kwargs())
    response.delete_cookie(STATE_COOKIE, path="/")
    return response


@router.get("/me")
def me(request: Request) -> dict[str, Any]:
    user = optional_current_user(request)
    if user is None:
        return {"authenticated": False, "user": None}

    from api.rate_limit import get_chat_quota

    return {
        "authenticated": True,
        "user": user,
        "chat_quota": get_chat_quota(user),
    }


@router.post("/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}
