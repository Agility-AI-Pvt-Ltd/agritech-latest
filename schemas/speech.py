from typing import Optional

from pydantic import BaseModel


class SpeechToTextRequest(BaseModel):
    audio_base64: str
    audio_mime_type: Optional[str] = None


class SpeechToTextResponse(BaseModel):
    text: str
    language_code: Optional[str] = None
    request_id: Optional[str] = None


class TextToSpeechRequest(BaseModel):
    text: str


class TextToSpeechResponse(BaseModel):
    audio_base64: str
    audio_mime_type: str
    request_id: Optional[str] = None
