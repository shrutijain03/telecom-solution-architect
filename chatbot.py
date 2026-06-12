import streamlit as st
import requests
from bs4 import BeautifulSoup
from groq import Groq
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from config import URL_SOURCES, MAX_CHARS_PER_URL, WEB_TIMEOUT_SECS, LLM_MODEL, MAX_TOKENS

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ──────────────────────────────────────────────────────
#  SCRAPE URLS  (cached — only runs once per session)
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """Scrape all URLs defined in config.py and return a list of content dicts."""
    knowledge = []
    for source in URL_SOURCES:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (TelecomCoPilot/1.0)"}
            resp = requests.get(source["url"], timeout=WEB_TIMEOUT_SECS,
                                verify=False, headers=headers)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.extract()
            text = " ".join(soup.get_text(separator=" ", strip=True).split())
            text = text[:MAX_CHARS_PER_URL]

            if text:
                knowledge.append({
                    "label": source["label"],
                    "url":   source["url"],
                    "domain": source["domain"],
                    "content": text
                })
        except Exception:
            # silently skip unreachable URLs
            pass
    return knowledge


def get_context_for_question(question: str, knowledge: list) -> tuple[str, list]:
    """
    Pick the most relevant chunks from scraped pages.
    Returns (context_string, list_of_sources_used).
    """
    q = question.lower()

    # Simple keyword → domain mapping
    domain_hints = []
    if any(w in q for w in ["api", "tmf", "open api", "tmforum"]):
        domain_hints.append("TMF_APIs")
    if any(w in q for w in ["architecture", "design", "oss", "bss", "oda"]):
        domain_hints.append("Architecture")
    if any(w in q for w in ["servicenow", "service now", "tmt", "itom"]):
        domain_hints.append("ServiceNow")
    if any(w in q for w in ["etom", "process", "fulfillment", "assurance"]):
        domain_hints.append("ETOM")

    # If domain matched, prefer those; otherwise use all
    if domain_hints:
        relevant = [k for k in knowledge if k["domain"] in domain_hints]
        if not relevant:
            relevant = knowledge
    else:
        relevant = knowledge

    # Build context (cap total to ~12000 chars to fit Groq context window)
    context_parts = []
    sources_used  = []
    total = 0
    for item in relevant:
        chunk = item["content"][:3000]
        if total + len(chunk) > 12000:
            break
        context_parts.append(f"[Source: {item['label']} — {item['url']}]\n{chunk}")
        sources_used.append({"label": item["label"], "url": item["url"]})
        total += len(chunk)

    return "\n\n---\n\n".join(context_parts), sources_used


# ──────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = {"name": "New Chat", "messages": []}


# ──────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────
def detect_domain(question: str) -> str:
    q = question.lower()
    if "etom" in q or "process" in q:
        return "ETOM"
    elif "sid" in q:
        return "SID"
    elif "api" in q or "tmf" in q:
        return "TMF_APIs"
    elif "architecture" in q or "design" in q or "oss" in q or "bss" in q:
        return "Architecture"
    return "All"


def is_architecture_query(question: str) -> bool:
    keywords = ["design", "architecture", "build", "implement", "solution", "system", "oss", "bss"]
    return any(w in question.lower() for w in keywords)


# ──────────────────────────────────────────────────────
#  LLM
# ──────────────────────────────────────────────────────
def generate_answer(question: str, context: str) -> str:
    import re

    has_context = bool(context.strip())
    context_note = (
        "Use the web content below as your primary source. "
        "Cite sources where relevant."
        if has_context
        else "No web context available — answer from your telecom domain knowledge."
    )

    if is_architecture_query(question):
        prompt = f"""
You are a Telecom Solution Architect. {context_note}

Web Context:
{context}

Question: {question}

Answer in this format:
🏗️ Architecture Components:
- Key systems (OSS, BSS, APIs, DB, etc.)

🔄 Flow:
- Step-by-step flow

🔗 APIs:
- TMF APIs used

📊 Integration:
- How systems connect
"""
    else:
        prompt = f"""
You are a Telecom Solution Architect AI. {context_note}

Web Context:
{context}

Question: {question}

Answer STRICTLY in this format:
📘 Definition:
- 2–3 clear lines

🔧 Telecom Context:
- 2–3 lines with telecom relevance

🏗️ Architecture Relevance:
- Importance in system design

💡 Example:
- Real telecom use case

🔗 Related APIs:
- List relevant APIs

Rules: bullet points only, ≥ 2 bullets per section, no combined sections.
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a telecom solution architect AI. Provide detailed answers with clear sections and bullet points."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5,
            max_tokens=MAX_TOKENS,
        )
        content = response.choices[0].message.content if response.choices else ""
        if not content:
            return "⚠️ No response generated."

        content = content.strip()
        content = re.sub(r"(📘|🔧|🏗️|💡|🔗|📊)", r"\n\1", content)

        formatted = []
        for line in content.split("\n"):
            line = line.strip()
            if any(sym in line for sym in ["📘", "🔧", "🏗️", "💡", "🔗", "📊"]):
                formatted.append(f"\n{line}")
            elif line:
                formatted.append(line if line.startswith("-") else f"- {line}")
        return "\n".join(formatted)

    except Exception as e:
        return f"⚠️ Groq Error: {str(e)}"


# ──────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Chats")

    if st.button("＋ New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat = chat_id
        st.session_state.chats[chat_id] = {"name": "New Chat", "messages": []}
        st.rerun()

    st.write("")
    for chat_id, chat_data_item in st.session_state.chats.items():
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(chat_data_item["name"], key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.rerun()
        with col2:
            st.button("⋮", key=f"menu_{chat_id}")

    st.divider()
    st.markdown("### Appearance")
    dark_mode = st.toggle("Dark Mode", value=False)

    st.divider()
    st.markdown("### Knowledge Domain")
    domain_filter = st.radio("", ["Auto", "ETOM", "SID", "TMF_APIs", "Architecture"])

    st.divider()
    with st.expander("🌐 Active URL Sources"):
        for src in URL_SOURCES:
            st.markdown(f"**{src['label']}**")
            st.caption(f"[{src['url']}]({src['url']})")
        st.caption("_Edit `config.py` to add/remove sources_")


# ──────────────────────────────────────────────────────
#  THEME
# ──────────────────────────────────────────────────────
if dark_mode:
    PAGE_BG, CARD_BG    = "#0f172a", "#1e293b"
    TEXT_COLOR, SUBTEXT = "#f9fafb", "#94a3b8"
    BORDER              = "#334155"
    BTN_BG, BTN_HOVER   = "#2d3748", "#4b5563"
    SB_BG,  SB_HOVER    = "#2d3748", "#4b5563"
    SB_BDR              = "#4b5563"
    SG_BG               = "linear-gradient(135deg,#312e81,#4c1d95)"
    SG_TEXT, SG_BDR     = "#e0e7ff", "#4338ca"
else:
    PAGE_BG, CARD_BG    = "#f8fafc", "#f1f5f9"
    TEXT_COLOR, SUBTEXT = "#111827", "#6b7280"
    BORDER              = "#e5e7eb"
    BTN_BG, BTN_HOVER   = "#f1f5f9", "#e2e8f0"
    SB_BG,  SB_HOVER    = "#e5e7eb", "#d1d5db"
    SB_BDR              = "#d1d5db"
    SG_BG               = "linear-gradient(135deg,#ede9fe,#ddd6fe)"
    SG_TEXT, SG_BDR     = "#4c1d95", "#a78bfa"

BTN_TEXT = "#f9fafb" if dark_mode else "#111827"
BTN_BDR  = "#4b5563" if dark_mode else "#e5e7eb"

st.markdown(f"""
<style>
body, .stApp {{
    background-color: {PAGE_BG}; color: {TEXT_COLOR};
    font-family: "Segoe UI", sans-serif;
}}
.chat-container {{ max-width: 700px; margin: auto; }}
.user-msg {{ text-align: right; }}
.user-bubble {{
    display: inline-block;
    background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    color: white; padding: 10px 14px; border-radius: 18px; max-width: 70%;
}}
.user-bubble small {{ color: #e5e7eb; }}
.bot-bubble {{
    background: {CARD_BG}; color: {TEXT_COLOR};
    padding: 14px; border-radius: 14px;
    border: 1px solid {BORDER}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
small {{ color: {SUBTEXT}; }}
textarea, input {{
    background-color: {CARD_BG} !important; color: {TEXT_COLOR} !important;
    border: 1px solid {BORDER} !important; border-radius: 10px !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {CARD_BG}; border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_COLOR}; }}
.stButton > button {{
    border-radius: 6px !important; background-color: {BTN_BG} !important;
    color: {BTN_TEXT} !important; border: 1px solid {BTN_BDR} !important;
}}
.stButton > button p {{ color: {BTN_TEXT} !important; }}
.stButton > button:hover {{ background-color: {BTN_HOVER} !important; }}
section[data-testid="stSidebar"] .stButton > button {{
    background-color: {SB_BG} !important; color: {BTN_TEXT} !important;
    border: 1px solid {SB_BDR} !important;
    width: 100% !important; text-align: left !important;
}}
section[data-testid="stSidebar"] .stButton > button p {{ color: {BTN_TEXT} !important; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background-color: {SB_HOVER} !important; }}
div[data-testid="column"] .stButton > button {{
    background: {SG_BG} !important; color: {SG_TEXT} !important;
    border: 1px solid {SG_BDR} !important; font-size: 13px !important;
    white-space: normal !important; height: auto !important;
    min-height: 52px !important; line-height: 1.4 !important;
}}
div[data-testid="column"] .stButton > button p {{ color: {SG_TEXT} !important; }}
div[data-testid="column"] .stButton > button:hover {{ opacity: 0.85 !important; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TITLE + KNOWLEDGE BASE LOAD
# ──────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;'>📡 Telecom Solution Architect Co‑Pilot</h1>
<p style='text-align:center; color:gray;'>
AI Assistant for Telecom Architecture, TM Forum & OSS/BSS Design
</p>
""", unsafe_allow_html=True)

with st.spinner("🌐 Loading knowledge from web sources..."):
    knowledge_base = load_knowledge_base()

if knowledge_base:
    st.success(f"✅ Loaded {len(knowledge_base)} source(s): " +
               ", ".join(k["label"] for k in knowledge_base), icon="🌐")
else:
    st.warning("⚠️ No web sources loaded. Answers will use Groq's built-in knowledge only.")

chat_data = st.session_state.chats[st.session_state.current_chat]
chat      = chat_data["messages"]


# ──────────────────────────────────────────────────────
#  SUGGESTED QUERIES
# ──────────────────────────────────────────────────────
if not chat:
    st.markdown(f"""
    <p style='text-align:center;font-size:15px;font-weight:500;color:{TEXT_COLOR};margin-bottom:8px;'>
        💡 Try a suggested query
    </p>""", unsafe_allow_html=True)

    suggested = [
        "Compare eTOM vs ServiceNow data models",
        "Map TMF APIs to order-to-cash lifecycle",
        "Design telecom OSS for fault management",
        "Explain service activation using TMF APIs",
    ]
    col1, col2 = st.columns(2)
    for idx, query in enumerate(suggested):
        with (col1 if idx % 2 == 0 else col2):
            if st.button(query, key=f"suggest_{idx}", use_container_width=True):
                st.session_state.prefill = query
                st.rerun()


# ──────────────────────────────────────────────────────
#  INPUT
# ──────────────────────────────────────────────────────
default_q = st.session_state.get("prefill", "")
question  = st.chat_input("Ask telecom architecture...")

if default_q and not question:
    question = default_q
    st.session_state.prefill = ""


# ──────────────────────────────────────────────────────
#  HANDLE QUERY
# ──────────────────────────────────────────────────────
def answer_question(q: str) -> dict:
    context, sources = get_context_for_question(q, knowledge_base)
    answer = generate_answer(q, context)
    return {"text": answer, "sources": sources}


if question:
    ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")
    chat.append(("user", question, ts))

    if chat_data["name"] == "New Chat":
        chat_data["name"] = question[:30]

    domain = detect_domain(question) if domain_filter == "Auto" else domain_filter
    typing = st.empty()
    typing.markdown(f"🔍 Searching web sources...  🕵🏻 Domain: **{domain}**")

    result = answer_question(question)
    typing.empty()
    chat.append(("bot", result, ts))


# ──────────────────────────────────────────────────────
#  DISPLAY
# ──────────────────────────────────────────────────────
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for i, (role, msg, ts) in enumerate(chat):
    if role == "user":
        st.markdown(f"""
        <div class="user-msg">
          <div class="user-bubble">{msg}<br><small>{ts}</small></div>
        </div>""", unsafe_allow_html=True)

    else:
        if isinstance(msg, dict):
            text    = msg.get("text", "")
            sources = msg.get("sources", [])
        else:
            text, sources = msg, []

        st.markdown(f"""
        <div class="bot-bubble">
          {text}<br><small>{ts}</small>
        </div>""", unsafe_allow_html=True)

        if sources:
            st.markdown("**🔗 Sources:**")
            for s in sources:
                st.markdown(f"- [{s['label']}]({s['url']})")

        col1, _ = st.columns([1, 1])
        with col1:
            if st.button("🔄 Regenerate", key=f"regen_{i}"):
                prev_q = next(
                    (chat[j][1] for j in range(i - 1, -1, -1) if chat[j][0] == "user"),
                    None,
                )
                if prev_q:
                    result   = answer_question(prev_q.strip())
                    regen_ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")
                    chat.append(("bot", result, regen_ts))
                    st.rerun()

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)