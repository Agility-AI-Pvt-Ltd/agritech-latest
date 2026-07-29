from __future__ import annotations

import html
import logging
import re
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from api.dependencies import get_chat_llm, get_chat_qdrant_client, get_chat_safety_llm
from core.config import settings


router = APIRouter(prefix="/api/vobiz", tags=["vobiz"])
logger = logging.getLogger(__name__)


def _xml_response(body: str) -> Response:
    return Response(content=body, media_type="application/xml")


def _xml_text(value: str) -> str:
    return html.escape(value, quote=False)


def _xml_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _response(*elements: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        + "\n".join(elements)
        + "\n</Response>"
    )


def _speak(text: str) -> str:
    return (
        f'    <Speak voice="{_xml_attr(settings.vobiz_speak_voice)}" '
        f'language="{_xml_attr(settings.vobiz_speak_language)}">'
        f"{_xml_text(_clean_for_speak(text))}</Speak>"
    )


def _hangup() -> str:
    return "    <Hangup/>"


def _get_public_base(request: Request) -> str:
    if settings.vobiz_public_base_url:
        return settings.vobiz_public_base_url.rstrip("/")
    return str(request.base_url).rstrip("/")


def _action_url(request: Request, route_path: str, **query: Any) -> str:
    base = _get_public_base(request)
    params = {k: v for k, v in query.items() if v is not None}
    if settings.vobiz_webhook_token:
        params["token"] = settings.vobiz_webhook_token
    suffix = f"?{urlencode(params)}" if params else ""
    return f"{base}{route_path}{suffix}"


def _gather(
    request: Request,
    prompt: str,
    *,
    retry: int = 0,
) -> str:
    action = _action_url(request, "/api/vobiz/call/gather", retry=retry)
    return (
        f'    <Gather action="{_xml_attr(action)}" method="POST" '
        f'inputType="speech" language="{_xml_attr(settings.vobiz_gather_language)}" '
        'speechModel="phone_call" speechEndTimeout="auto" executionTimeout="20" '
        f'hints="{_xml_attr(settings.vobiz_gather_hints)}">\n'
        f"{_speak(prompt)}\n"
        "    </Gather>"
    )


def _clean_for_speak(text: str) -> str:
    cleaned = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"[*#>|_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > settings.vobiz_max_response_chars:
        cleaned = cleaned[: settings.vobiz_max_response_chars].rsplit(" ", 1)[0].strip()
        cleaned = f"{cleaned}. You can ask a follow-up question."
    return cleaned or "Sorry, I could not prepare a response."


def _call_user_id(form: dict[str, Any]) -> str:
    caller = str(form.get("From") or "unknown").strip() or "unknown"
    return f"phone:{caller}"


def _conversation_id(form: dict[str, Any]) -> str:
    call_uuid = str(form.get("CallUUID") or "").strip()
    if call_uuid:
        return f"vobiz:{call_uuid}"
    caller = str(form.get("From") or "unknown").strip() or "unknown"
    called = str(form.get("To") or "unknown").strip() or "unknown"
    return f"vobiz:{caller}:{called}"


def _check_token(request: Request) -> None:
    expected = settings.vobiz_webhook_token
    if expected and request.query_params.get("token") != expected:
        raise HTTPException(status_code=403, detail="Invalid Vobiz webhook token")


async def _read_form(request: Request) -> dict[str, Any]:
    try:
        form = await request.form()
        return dict(form)
    except Exception:
        return {}


@router.api_route("/call/answer", methods=["GET", "POST"])
async def answer_call(request: Request) -> Response:
    """Initial Vobiz Answer URL for inbound phone calls."""
    _check_token(request)
    form = await _read_form(request)
    logger.info(
        "Vobiz call answered: CallUUID=%s From=%s To=%s",
        form.get("CallUUID"),
        form.get("From"),
        form.get("To"),
    )

    xml = _response(
        _gather(
            request,
            "Welcome to Kisan Mitra. Please ask your farming question after the tone.",
            retry=0,
        ),
        _speak("Sorry, I did not hear anything. Please call again."),
        _hangup(),
    )
    return _xml_response(xml)


@router.api_route("/call/gather", methods=["GET", "POST"])
async def gather_call_input(
    request: Request,
    llm=Depends(get_chat_llm),
    safety_llm=Depends(get_chat_safety_llm),
    qdrant_client=Depends(get_chat_qdrant_client),
) -> Response:
    """Handle Vobiz Gather callbacks and speak the Kisan Mitra response."""
    _check_token(request)
    form = await _read_form(request)
    speech = str(form.get("Speech") or "").strip()
    retry = int(request.query_params.get("retry") or "0")

    logger.info(
        "Vobiz gather: CallUUID=%s InputType=%s Speech=%r Confidence=%s",
        form.get("CallUUID"),
        form.get("InputType"),
        speech[:120],
        form.get("SpeechConfidenceScore"),
    )

    if not speech:
        if retry < 1:
            xml = _response(
                _gather(
                    request,
                    "I did not catch that. Please ask your farming question again.",
                    retry=retry + 1,
                ),
                _speak("No input received. Goodbye."),
                _hangup(),
            )
        else:
            xml = _response(_speak("No input received. Goodbye."), _hangup())
        return _xml_response(xml)

    if qdrant_client is None:
        logger.error("Vobiz call failed: chat Qdrant client is not initialized")
        return _xml_response(
            _response(
                _speak("Kisan Mitra is not ready right now. Please try again later."),
                _hangup(),
            )
        )

    from pipeline.graph import arun

    call_query = f"{settings.vobiz_call_response_instruction}\n\nCaller question: {speech}"
    try:
        result = await arun(
            query=call_query,
            llm=llm,
            safety_llm=safety_llm,
            qdrant_client=qdrant_client,
            conversation_id=_conversation_id(form),
            user_id=_call_user_id(form),
        )
        answer = result.get("final_response", "")
    except Exception as exc:
        logger.exception("Vobiz call agent failed: %s", exc)
        answer = "Sorry, Kisan Mitra had trouble answering that. Please ask again."

    xml = _response(
        _speak(answer),
        _gather(request, "You can ask another farming question now.", retry=0),
        _speak("Thank you for calling Kisan Mitra. Goodbye."),
        _hangup(),
    )
    return _xml_response(xml)


@router.api_route("/call/hangup", methods=["GET", "POST"])
async def call_hangup(request: Request) -> dict[str, bool]:
    """Optional Vobiz Hangup URL for logging call end events."""
    _check_token(request)
    form = await _read_form(request)
    logger.info(
        "Vobiz call ended: CallUUID=%s From=%s Duration=%s Cause=%s",
        form.get("CallUUID"),
        form.get("From"),
        form.get("Duration"),
        form.get("HangupCause"),
    )
    return {"ok": True}
