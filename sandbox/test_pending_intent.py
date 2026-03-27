import re
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from core.config import settings
from pipeline.agent import agent_node
from pipeline.agent_guards import MAIZE_SOWING_DATE_REQUIREMENT
from pipeline.graph import run


class FakeMainLLM:
    def __init__(self):
        self.invoke_calls = 0
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invoke_calls += 1
        final_human = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            "",
        )
        if "Extract maize sowing date if clearly implied." in final_human:
            return AIMessage(content='{"sowing_date": null, "reason": "not_needed"}')

        if "Acknowledgment to include briefly:" in final_human:
            ack_match = re.search(r"Acknowledgment to include briefly:\s*(.+)", final_human)
            ack = ack_match.group(1).strip() if ack_match else "ठीक है, तारीख सेव हो गई है।"
            return AIMessage(
                content=(
                    f"{ack} अब मक्का में कीट के सही लक्षण देखकर ही कीटनाशक चुनें। "
                    "तना छेदक या पत्ती खाने वाले कीट दिखें तो स्थानीय अनुशंसा के अनुसार उचित कीटनाशक का प्रयोग करें।"
                )
            )

        return AIMessage(content='{}')


def _disable_safety():
    settings.chat_safety_enabled = False
    settings.chat_security_blocking_enabled = False
    settings.chat_low_info_blocking_enabled = False
    settings.chat_safety_model_classification_enabled = False
    settings.chat_safety_fail_closed = False


def test_maize_query_stores_pending_requirement():
    _disable_safety()
    llm = FakeMainLLM()

    state = agent_node(
        {
            "raw_input": "मक्का में कौन सा कीटनाशक डालूँ?",
            "chat_history": [],
            "tool_calls": [],
            "loop_count": 0,
            "needs_more_info": False,
            "pending_context": {},
            "errors": [],
        },
        llm=llm,
        qdrant_client=None,
    )

    assert "बुवाई की तारीख" in state.get("final_response", "")
    assert state.get("pending_user_intent") == "मक्का में कौन सा कीटनाशक डालूँ?"
    assert state.get("pending_requirement") == MAIZE_SOWING_DATE_REQUIREMENT
    assert state.get("pending_context") == {"crop": "maize"}
    assert llm.invoke_calls == 0


def test_relative_sowing_date_resumes_pending_query_and_clears_pending_state():
    _disable_safety()
    llm = FakeMainLLM()
    store = {}

    def fake_load_state(conversation_id):
        return store.get(conversation_id)

    def fake_save_state(conversation_id, state, user_id=None):
        store[conversation_id] = dict(state)

    with patch("pipeline.graph.db.load_state", side_effect=fake_load_state), patch(
        "pipeline.graph.db.save_state", side_effect=fake_save_state
    ):
        first = run(
            query="मक्का में कौन सा कीटनाशक डालूँ?",
            llm=llm,
            qdrant_client=None,
            conversation_id="conv_pending_01",
        )
        second = run(
            query="20 din phle",
            llm=llm,
            qdrant_client=None,
            conversation_id="conv_pending_01",
        )

    assert "बुवाई की तारीख" in first.get("final_response", "")
    assert store["conv_pending_01"].get("user_sowing_date") == "2026-03-06"
    assert store["conv_pending_01"].get("pending_user_intent") is None
    assert store["conv_pending_01"].get("pending_requirement") is None
    assert "2026-03-06" in second.get("final_response", "")
    assert "कीटनाशक" in second.get("final_response", "")
    assert llm.invoke_calls == 1


def test_history_fallback_resumes_when_pending_field_is_missing():
    _disable_safety()
    llm = FakeMainLLM()

    state = {
        "raw_input": "20 din phle",
        "chat_history": [
            {"role": "user", "content": "मक्का में कौन सा कीटनाशक डालूँ?"},
            {
                "role": "assistant",
                "content": (
                    "मक्का की सही और स्टेज-आधारित सलाह देने के लिए पहले बुवाई की तारीख बताइए। "
                    "कृपया तारीख YYYY-MM-DD में बताएं, जैसे 2025-07-10."
                ),
            },
        ],
        "tool_calls": [],
        "loop_count": 0,
        "needs_more_info": False,
        "pending_context": {},
        "errors": [],
    }

    intermediate = agent_node(state, llm=llm, qdrant_client=None)
    assert intermediate.get("needs_more_info") is True

    resumed = agent_node(intermediate, llm=llm, qdrant_client=None)
    assert resumed.get("needs_more_info") is False
    assert "2026-03-06" in resumed.get("final_response", "")
    assert "कीटनाशक" in resumed.get("final_response", "")


def test_sowing_date_without_pending_query_only_acknowledges():
    _disable_safety()
    llm = FakeMainLLM()

    state = agent_node(
        {
            "raw_input": "2026-03-06",
            "chat_history": [],
            "tool_calls": [],
            "loop_count": 0,
            "needs_more_info": False,
            "pending_context": {},
            "errors": [],
        },
        llm=llm,
        qdrant_client=None,
    )

    assert state.get("final_response") == "ठीक है, मैंने आपकी मक्का की बुवाई की तारीख 2026-03-06 सेव कर ली है।"
    assert state.get("pending_user_intent") is None
    assert llm.invoke_calls == 0
