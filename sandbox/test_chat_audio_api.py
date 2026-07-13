from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_sentence_transformers_stub() -> None:
    """Avoid importing heavy optional deps during local API route testing."""
    if "sentence_transformers" in sys.modules:
        return

    stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def encode(self, *args, **kwargs):
            raise RuntimeError("SentenceTransformer.encode should not be used in this mocked API test.")

    stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = stub


_install_sentence_transformers_stub()

from fastapi.testclient import TestClient

import app as fastapi_app_module
from api.dependencies import (
    get_chat_llm,
    get_chat_qdrant_client,
    get_chat_safety_llm,
    get_speech_service,
)
from api.auth import get_current_user
from services.speech import SpeechToTextResult, TextToSpeechResult


class FakeSpeechService:
    def is_configured(self) -> bool:
        return True

    def speech_to_text(self, *, audio_base64: str, audio_mime_type: str | None = None, file_name: str | None = None) -> SpeechToTextResult:
        del audio_mime_type, file_name
        decoded = base64.b64decode(audio_base64).decode("utf-8")
        return SpeechToTextResult(transcript=f"audio says: {decoded}", language_code="hi-IN", request_id="stt-test-1")

    def text_to_speech(self, *, text: str) -> TextToSpeechResult:
        encoded = base64.b64encode(f"AUDIO::{text}".encode("utf-8")).decode("utf-8")
        return TextToSpeechResult(audio_base64=encoded, mime_type="audio/wav", request_id="tts-test-1")


def _override_dependencies() -> None:
    app = fastapi_app_module.app
    app.dependency_overrides[get_speech_service] = lambda: FakeSpeechService()
    app.dependency_overrides[get_chat_llm] = lambda: object()
    app.dependency_overrides[get_chat_safety_llm] = lambda: object()
    app.dependency_overrides[get_chat_qdrant_client] = lambda: object()
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "test-google-user",
        "email": "test@example.com",
        "name": "Test User",
        "picture": "",
    }


def _install_graph_run_stub() -> None:
    import pipeline.graph

    def fake_run(query: str, **kwargs):
        del kwargs
        return {
            "final_response": f"mocked response for: {query}",
            "loop_count": 1,
            "tool_calls": [
                {"tool": "faq_search_by_crop_stage"},
                {"tool": "rag_search"},
            ],
        }

    pipeline.graph.run = fake_run


def _exercise_text_mode(client: TestClient) -> dict:
    payload = {
        "user_id": "test-user-1",
        "conversation_id": "conv-text-1",
        "query": "मुझे यूरिया कितनी मात्रा में डालनी चाहिए?",
    }
    response = client.post("/api/chat", json=payload)
    response.raise_for_status()
    return response.json()


def _exercise_voice_pipeline(client: TestClient) -> dict:
    fake_audio = base64.b64encode("hello from audio".encode("utf-8")).decode("utf-8")

    stt_response = client.post(
        "/api/stt",
        json={
            "audio_base64": fake_audio,
            "audio_mime_type": "audio/wav",
        },
    )
    stt_response.raise_for_status()
    stt_result = stt_response.json()

    chat_response = client.post(
        "/api/chat",
        json={
            "user_id": "test-user-2",
            "conversation_id": "conv-audio-1",
            "query": stt_result["text"],
        },
    )
    chat_response.raise_for_status()
    chat_result = chat_response.json()

    tts_response = client.post(
        "/api/tts",
        json={"text": chat_result["response"]},
    )
    tts_response.raise_for_status()
    tts_result = tts_response.json()

    return {
        "stt": stt_result,
        "chat": chat_result,
        "tts": tts_result,
    }


def _exercise_stt_error_mode(client: TestClient) -> dict:
    response = client.post(
        "/api/stt",
        json={
            "audio_base64": "",
            "audio_mime_type": "audio/wav",
        },
    )
    return {
        "status_code": response.status_code,
        "body": response.json(),
    }


def main() -> int:
    _override_dependencies()
    _install_graph_run_stub()

    client = TestClient(fastapi_app_module.app)

    text_result = _exercise_text_mode(client)
    voice_pipeline_result = _exercise_voice_pipeline(client)
    stt_error_result = _exercise_stt_error_mode(client)

    print("TEXT MODE RESULT")
    print(json.dumps(text_result, ensure_ascii=False, indent=2))
    print()
    print("VOICE PIPELINE RESULT (/api/stt -> /api/chat -> /api/tts)")
    print(json.dumps(voice_pipeline_result, ensure_ascii=False, indent=2))
    print()
    print("STT VALIDATION RESULT")
    print(json.dumps(stt_error_result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
