from typing import List

from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    query: str


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    user_id: str
    tools_used: List[str]
    loop_count: int
    rate_limit_remaining: int
    rate_limit_limit: int
