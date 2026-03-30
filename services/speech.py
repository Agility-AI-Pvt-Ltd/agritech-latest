from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from typing import Any

import requests

from core.config import settings


@dataclass
class SpeechToTextResult:
    transcript: str
    language_code: str | None = None
    request_id: str | None = None


@dataclass
class TextToSpeechResult:
    audio_base64: str
    mime_type: str
    request_id: str | None = None


class SarvamSpeechService:
    def __init__(self) -> None:
        self.api_key = settings.sarvam_api_key
        self.base_url = settings.sarvam_base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def speech_to_text(
        self,
        *,
        audio_base64: str,
        audio_mime_type: str | None = None,
        file_name: str | None = None,
    ) -> SpeechToTextResult:
        if not self.is_configured():
            raise RuntimeError("SARVAM_API_KEY is not configured.")

        normalized_mime_type = self._normalize_audio_mime_type(audio_mime_type)
        audio_bytes = base64.b64decode(audio_base64)
        resolved_name = file_name or self._guess_file_name(normalized_mime_type)
        content_type = normalized_mime_type or "application/octet-stream"

        response = requests.post(
            f"{self.base_url}/speech-to-text",
            headers={"api-subscription-key": self.api_key},
            files={
                "file": (resolved_name, audio_bytes, content_type),
            },
            data={
                "model": settings.sarvam_stt_model,
                "language_code": settings.sarvam_stt_language_code,
                "mode": settings.sarvam_stt_mode,
            },
            timeout=60,
        )
        self._raise_for_status_with_body(response, "Sarvam STT")
        payload = response.json()
        return SpeechToTextResult(
            transcript=(payload.get("transcript") or "").strip(),
            language_code=payload.get("language_code"),
            request_id=payload.get("request_id"),
        )

    def text_to_speech(self, *, text: str) -> TextToSpeechResult:
        if not self.is_configured():
            raise RuntimeError("SARVAM_API_KEY is not configured.")

        payload = {
            "text": text,
            "target_language_code": settings.sarvam_tts_language_code,
            "model": settings.sarvam_tts_model,
            "speaker": settings.sarvam_tts_speaker,
            "speech_sample_rate": settings.sarvam_tts_sample_rate,
            "pace": settings.sarvam_tts_pace,
            "temperature": settings.sarvam_tts_temperature,
        }
        if settings.sarvam_tts_audio_format:
            payload["output_audio_codec"] = settings.sarvam_tts_audio_format

        response = requests.post(
            f"{self.base_url}/text-to-speech",
            headers={
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        self._raise_for_status_with_body(response, "Sarvam TTS")
        payload = response.json()
        audios = payload.get("audios") or []
        if not audios:
            raise RuntimeError("Sarvam TTS response did not include audio.")

        return TextToSpeechResult(
            audio_base64=audios[0],
            mime_type=self._format_to_mime_type(settings.sarvam_tts_audio_format),
            request_id=payload.get("request_id"),
        )

    @staticmethod
    def _guess_file_name(audio_mime_type: str | None) -> str:
        extension = mimetypes.guess_extension(audio_mime_type or "") or ".wav"
        return f"input_audio{extension}"

    @staticmethod
    def _normalize_audio_mime_type(audio_mime_type: str | None) -> str | None:
        if not audio_mime_type:
            return audio_mime_type
        return audio_mime_type.split(";", 1)[0].strip().lower()

    @staticmethod
    def _format_to_mime_type(audio_format: str) -> str:
        format_map: dict[str, str] = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "aac": "audio/aac",
            "opus": "audio/opus",
            "flac": "audio/flac",
            "pcm": "audio/L16",
            "mulaw": "audio/basic",
            "alaw": "audio/G711-ALAW",
        }
        return format_map.get(audio_format.lower(), "application/octet-stream")

    @staticmethod
    def _raise_for_status_with_body(response: requests.Response, source: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text.strip()
            if body:
                raise RuntimeError(
                    f"{source} failed with status {response.status_code}: {body}"
                ) from exc
            raise RuntimeError(
                f"{source} failed with status {response.status_code}."
            ) from exc
