from unittest.mock import patch

from fastapi.testclient import TestClient

from api.dependencies import get_chat_llm, get_chat_qdrant_client, get_chat_safety_llm
from app import app


def _fake_llm():
    return object()


def _fake_qdrant():
    return object()


def setup_module():
    app.dependency_overrides[get_chat_llm] = _fake_llm
    app.dependency_overrides[get_chat_safety_llm] = lambda: None
    app.dependency_overrides[get_chat_qdrant_client] = _fake_qdrant


def teardown_module():
    app.dependency_overrides.clear()


def test_vobiz_answer_returns_gather_xml():
    client = TestClient(app)

    response = client.post(
        "/api/vobiz/call/answer",
        data={"CallUUID": "call-1", "From": "+911234567890", "To": "+919999999999"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "<Gather" in response.text
    assert 'inputType="speech"' in response.text
    assert "/api/vobiz/call/gather" in response.text


def test_vobiz_gather_runs_agent_and_returns_next_prompt():
    client = TestClient(app)

    with patch("pipeline.graph.run") as run:
        run.return_value = {"final_response": "Apply irrigation today and check for pest symptoms."}
        response = client.post(
            "/api/vobiz/call/gather",
            data={
                "CallUUID": "call-1",
                "From": "+911234567890",
                "To": "+919999999999",
                "InputType": "speech",
                "Speech": "My maize crop leaves are yellow",
                "SpeechConfidenceScore": "0.91",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "Apply irrigation today" in response.text
    assert "You can ask another farming question now" in response.text
    assert run.call_args.kwargs["conversation_id"] == "vobiz:call-1"
    assert run.call_args.kwargs["user_id"] == "phone:+911234567890"
