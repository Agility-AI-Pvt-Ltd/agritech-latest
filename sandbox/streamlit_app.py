"""
streamlit_app.py  –  Kisan Mitra | Agritech Advisory Chat UI
Connects to the FastAPI /api/chat endpoint.

Usage:
  streamlit run sandbox/streamlit_app.py
"""

import uuid
import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="किसान मित्र | Kisan Mitra",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* App background */
.stApp { background: linear-gradient(135deg, #0d1f0e 0%, #0a2416 50%, #081a1a 100%); }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2b0e 0%, #0a1f1f 100%);
    border-right: 1px solid #1e4d20;
}
section[data-testid="stSidebar"] * { color: #d4edda !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a5c1e 0%, #0e7c3a 50%, #0d6b5c 100%);
    border-radius: 16px;
    padding: 22px 30px;
    margin-bottom: 20px;
    border: 1px solid #2d8a32;
    box-shadow: 0 8px 32px rgba(0,200,80,0.15);
}
.hero-banner h1 { color: #fff; margin: 0; font-size: 1.9rem; font-weight: 700; }
.hero-banner p  { color: #b8f0c8; margin: 5px 0 0 0; font-size: 0.95rem; }

/* Native chat message styling */
[data-testid="stChatMessage"] {
    background: rgba(13, 42, 16, 0.6) !important;
    border: 1px solid #1e4d20 !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
    padding: 12px !important;
}
[data-testid="stChatMessage"] p { color: #d4edda !important; }
[data-testid="stChatMessageContent"] { color: #d4edda !important; }

/* Chat input */
[data-testid="stChatInput"] {
    background: #0d2010 !important;
    border: 1px solid #1e5c20 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    color: #d4edda !important;
    background: transparent !important;
}
[data-testid="stChatInput"] button {
    background: #1a7c2e !important;
    border-radius: 10px !important;
}

/* Tool badges */
.tool-pill {
    display: inline-block;
    background: rgba(14,124,58,0.25);
    border: 1px solid #1e6b30;
    color: #7fefb0;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px 3px 0 0;
    font-weight: 500;
}

/* Sidebar button */
.stButton > button {
    background: linear-gradient(135deg, #1a7c2e, #0e6b3a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #25a83e, #16894a) !important;
    transform: translateY(-1px) !important;
}

/* Status card */
.status-card {
    background: rgba(14,50,16,0.5);
    border: 1px solid #1e4d20;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #b8f0c8;
    line-height: 1.8;
}

hr { border-color: #1e4d20 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

TOOL_ICONS = {
    "rag_search":           "📚 RAG Search",
    "faq_search_by_crop_stage": "🌽 FAQ Search",
    "set_crop_stage":       "🌱 Crop Stage",
    "bighaat_search":       "🛒 BigHaat",
    "get_weather":          "🌤️ Weather",
    "geocode_location":     "📍 Geocode",
    "web_search":           "🔍 Web Search",
    "get_current_datetime": "🕒 DateTime",
}


def _format_tool_label(tool_name: str) -> str:
    """Return a human-friendly tool label for the chat UI."""
    if tool_name in TOOL_ICONS:
        return TOOL_ICONS[tool_name]
    pretty_name = tool_name.replace("_", " ").strip().title()
    return f"🔧 {pretty_name}"

# ── Session State ──────────────────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, tools?}]
if "total_turns" not in st.session_state:
    st.session_state.total_turns = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 किसान मित्र")
    st.markdown("**Agritech Advisory Assistant**")
    st.markdown("---")

    st.markdown("### 🔧 Settings")
    new_uid = st.text_input("User ID", value=st.session_state.user_id)
    if new_uid != st.session_state.user_id:
        st.session_state.user_id = new_uid

    new_cid = st.text_input("Conversation ID", value=st.session_state.conversation_id)
    if new_cid != st.session_state.conversation_id:
        st.session_state.conversation_id = new_cid

    st.markdown("---")
    if st.button("🔄 New Conversation"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.total_turns = 0
        st.rerun()

    st.markdown(f"""
    <div class="status-card">
        🟢 <b>API:</b> {API_BASE}<br>
        💬 <b>Turns:</b> {st.session_state.total_turns}<br>
        🔑 <b>Conv:</b> <code>{st.session_state.conversation_id[:16]}…</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    samples = [
        "नमस्ते! आज खेत में क्या करना चाहिए?",
        "मेरी मक्का फसल में कीट लग गए हैं।",
        "अगले 3 दिन का मौसम कैसा रहेगा?",
        "खाद कब और कितनी डालें?",
        "फसल को गर्मी से कैसे बचाएं?",
    ]
    for q in samples:
        if st.button(q, key=f"sq_{q[:15]}"):
            st.session_state["_prefill"] = q
            st.rerun()

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🌾 किसान मित्र</h1>
    <p>आपका AI कृषि सलाहकार — Spring Corn (Zaid Maize) विशेषज्ञ, Uttar Pradesh</p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; color:#5a8a62; padding:50px 20px;">
        <div style="font-size:3rem">🌱</div>
        <p style="font-size:1rem; margin-top:12px; color:#7fefb0;">नमस्ते! कोई भी कृषि सवाल पूछें।</p>
        <p style="font-size:0.85rem; color:#5a8a62;">Ask anything about your crops in Hindi or English.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        avatar = "👨‍🌾" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            # Show tool badges if present
            if msg.get("tools"):
                badges = " ".join(
                    f'<span class="tool-pill">{_format_tool_label(t)}</span>'
                    for t in msg["tools"]
                )
                st.markdown(badges, unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", "")
query = st.chat_input(
    placeholder="अपना सवाल यहाँ लिखें… (Type your question here)",
    key="chat_input",
) or prefill

# ── Send & get response ────────────────────────────────────────────────────────
if query and query.strip():
    query = query.strip()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👨‍🌾"):
        st.markdown(query)

    # Call API and stream response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("सोच रहा हूँ… 🌿"):
            try:
                payload = {
                    "user_id":         st.session_state.user_id,
                    "conversation_id": st.session_state.conversation_id,
                    "query":           query,
                }
                resp = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=90)
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("response", "")
                tools  = data.get("tools_used", [])

                st.markdown(answer)

                if tools:
                    badges = " ".join(
                        f'<span class="tool-pill">{_format_tool_label(t)}</span>'
                        for t in tools
                    )
                    st.markdown(badges, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "tools": tools
                })
                st.session_state.total_turns += 1

            except requests.exceptions.ConnectionError:
                err = "❌ **Server से कनेक्ट नहीं हो पाया।** कृपया सुनिश्चित करें कि `python main.py` चल रहा है।"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "tools": []})

            except requests.exceptions.Timeout:
                err = "⏳ **Request timeout हो गई।** LLM busy है, फिर से कोशिश करें।"
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "tools": []})

            except Exception as e:
                err = f"❌ **Error:** {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "tools": []})

    st.rerun()
