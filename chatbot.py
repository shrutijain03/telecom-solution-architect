import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import uuid
from sentence_transformers import CrossEncoder
from datetime import datetime
from zoneinfo import ZoneInfo

from config import (
    DB_PATH, EMBEDDING_MODEL, RERANKER_MODEL,
    LLM_MODEL, RETRIEVER_K, RERANKER_TOP_N,
)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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


def calculate_confidence(docs: list) -> str:
    if not docs:
        return "Low"
    return "High" if len(docs) >= 2 else "Medium"


def is_architecture_query(question: str) -> bool:
    keywords = ["design", "architecture", "build", "implement", "solution", "system", "oss", "bss"]
    return any(w in question.lower() for w in keywords)


def format_source(metadata: dict) -> tuple:
    """Return (label, url_or_empty) for a doc's metadata."""
    if metadata.get("source") == "web":
        url = metadata.get("url", "")
        return url, url
    else:
        label  = metadata.get("file_name", "PDF document")
        domain = metadata.get("domain", "")
        return "📄 " + label + (f" ({domain})" if domain else ""), ""


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
            if st.button("⋮", key=f"menu_{chat_id}"):
                pass

    st.divider()

    st.markdown("### Appearance")
    dark_mode = st.toggle("Dark Mode", value=False)

    st.divider()

    st.markdown("### Knowledge Domain")
    domain_filter = st.radio("", ["Auto", "ETOM", "SID", "TMF_APIs", "Architecture"])

    st.divider()

    # Active source info panel
    with st.expander("📋 Active Sources"):
        try:
            from config import PDF_SOURCES, WEB_SOURCES, ENABLE_PDF, ENABLE_WEB
            if ENABLE_PDF:
                st.markdown("**PDF folders**")
                for domain, path in PDF_SOURCES.items():
                    exists = "✅" if os.path.isdir(path) else "❌"
                    st.caption(f"{exists} {domain} → `{path}`")
            if ENABLE_WEB:
                st.markdown("**URLs**")
                for url in WEB_SOURCES:
                    st.caption(f"🌐 {url}")
            st.caption("_Edit `config.py` to add/remove sources, then re-run `ingest.py`_")
        except Exception:
            st.caption("config.py not found")


# ──────────────────────────────────────────────────────
#  THEME COLOURS
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


# ──────────────────────────────────────────────────────
#  STYLES
# ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
body, .stApp {{
    background-color: {PAGE_BG};
    color: {TEXT_COLOR};
    font-family: "Segoe UI", sans-serif;
}}
.chat-container {{ max-width: 700px; margin: auto; }}
.user-msg {{ text-align: right; }}
.user-bubble {{
    display: inline-block;
    background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    color: white;
    padding: 10px 14px;
    border-radius: 18px;
    max-width: 70%;
}}
.user-bubble small {{ color: #e5e7eb; }}
.bot-bubble {{
    background: {CARD_BG};
    color: {TEXT_COLOR};
    padding: 14px;
    border-radius: 14px;
    border: 1px solid {BORDER};
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
small {{ color: {SUBTEXT}; }}
textarea, input {{
    background-color: {CARD_BG} !important;
    color: {TEXT_COLOR} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {CARD_BG};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_COLOR}; }}

.stButton > button {{
    border-radius: 6px !important;
    background-color: {BTN_BG} !important;
    color: {BTN_TEXT} !important;
    border: 1px solid {BTN_BDR} !important;
}}
.stButton > button p {{ color: {BTN_TEXT} !important; }}
.stButton > button:hover {{ background-color: {BTN_HOVER} !important; }}

section[data-testid="stSidebar"] .stButton > button {{
    background-color: {SB_BG} !important;
    color: {BTN_TEXT} !important;
    border: 1px solid {SB_BDR} !important;
    width: 100% !important;
    text-align: left !important;
}}
section[data-testid="stSidebar"] .stButton > button p {{ color: {BTN_TEXT} !important; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background-color: {SB_HOVER} !important; }}

div[data-testid="column"] .stButton > button {{
    background: {SG_BG} !important;
    color: {SG_TEXT} !important;
    border: 1px solid {SG_BDR} !important;
    font-size: 13px !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 52px !important;
    line-height: 1.4 !important;
}}
div[data-testid="column"] .stButton > button p {{ color: {SG_TEXT} !important; }}
div[data-testid="column"] .stButton > button:hover {{ opacity: 0.85 !important; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TITLE
# ──────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;'>📡 Telecom Solution Architect Co‑Pilot</h1>
<p style='text-align:center; color:gray;'>
AI Assistant for Telecom Architecture, TM Forum & OSS/BSS Design
</p>
""", unsafe_allow_html=True)

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
#  LOAD MODELS  (cached — only load once)
# ──────────────────────────────────────────────────────
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_MODEL)


vectordb = load_db()
reranker = load_reranker()


def get_retriever(domain_hint: str):
    """Domain-filtered retriever; falls back to unfiltered if needed."""
    k = RETRIEVER_K
    if domain_hint in ("Auto", "All"):
        return vectordb.as_retriever(search_kwargs={"k": k})
    return vectordb.as_retriever(
        search_kwargs={"k": k, "filter": {"domain": domain_hint}}
    )


def rerank_docs(question: str, docs: list) -> list:
    if not docs:
        return docs
    pairs  = [(question, doc.page_content) for doc in docs if doc.page_content]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:RERANKER_TOP_N]]


# ──────────────────────────────────────────────────────
#  LLM
# ──────────────────────────────────────────────────────
def generate_answer(question: str, context: str) -> str:
    import re

    if is_architecture_query(question):
        prompt = f"""
You are a Telecom Solution Architect. Respond using bullet points and clearly separated sections.
Use the context if available; otherwise use your telecom knowledge.

Context:
{context}

Question:
{question}

Format:
🏗️ Architecture Components:
- Key systems (OSS, BSS, APIs, DB)

🔄 Flow:
- Step-by-step

🔗 APIs:
- TMF APIs used

📊 Integration:
- How systems connect
"""
    else:
        prompt = f"""
You are a Telecom Solution Architect AI. Respond using bullet points and clearly separated sections.
Use the context if available; otherwise use telecom domain knowledge.

Context:
{context}

Question:
{question}

Format (strictly follow):
📘 Definition:
- 2–3 clear lines

🔧 Telecom Context:
- 2–3 lines with telecom relevance

🏗️ Architecture Relevance:
- Importance in system design

💡 Example:
- Real telecom use case

🔗 Related APIs:
- Relevant APIs

Rules: bullet points only, ≥ 2 bullets per section, no combined sections.
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a telecom solution architect AI. Detailed answers with clear sections and bullet points."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5,
            max_tokens=800,
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
#  RAG PIPELINE  (shared by query + regenerate)
# ──────────────────────────────────────────────────────
def retrieve_and_answer(question: str, domain_hint: str) -> dict:
    retriever = get_retriever(domain_hint)
    docs      = retriever.invoke(question)

    # Fallback: domain filter returned nothing → retry without filter
    if not docs and domain_hint not in ("Auto", "All"):
        docs = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K}).invoke(question)

    docs = rerank_docs(question, docs)

    print(f"DEBUG → docs found: {len(docs)}")
    for d in docs:
        print(f"  SOURCE: {d.metadata}")

    context    = "\n\n".join([doc.page_content[:300] for doc in docs])
    confidence = calculate_confidence(docs)
    answer     = generate_answer(question, context)

    sources = []
    for doc in docs:
        label, url = format_source(doc.metadata)
        sources.append({"label": label, "url": url})

    return {"text": answer, "sources": sources, "confidence": confidence}


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
if question:
    ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")
    chat.append(("user", question, ts))

    if chat_data["name"] == "New Chat":
        chat_data["name"] = question[:30]

    domain = detect_domain(question) if domain_filter == "Auto" else domain_filter

    typing = st.empty()
    typing.markdown(f"🔍 Searching telecom knowledge...  🕵🏻 Domain: **{domain}**")

    result = retrieve_and_answer(question, domain_filter)
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
        </div>
        """, unsafe_allow_html=True)

    else:
        if isinstance(msg, dict):
            text       = msg.get("text", "")
            sources    = msg.get("sources", [])
            confidence = msg.get("confidence", "")
            # back-compat: plain strings → dict
            sources = [
                s if isinstance(s, dict) else {"label": s, "url": ""}
                for s in sources
            ]
        else:
            text, sources, confidence = msg, [], ""

        st.markdown(f"""
        <div class="bot-bubble">
          {text}<br><small>{ts}</small>
        </div>
        """, unsafe_allow_html=True)

        if sources:
            st.markdown("**🔗 Sources:**")
            for s in sources:
                label = s.get("label", "")
                url   = s.get("url", "")
                if url:
                    st.markdown(f"- [🌐 {url}]({url})")
                else:
                    st.markdown(f"- {label}")

        if confidence:
            st.caption(f"Confidence: {confidence}")

        col1, _ = st.columns([1, 1])
        with col1:
            if st.button("🔄 Regenerate", key=f"regen_{i}"):
                prev_q = next(
                    (chat[j][1] for j in range(i - 1, -1, -1) if chat[j][0] == "user"),
                    None,
                )
                if prev_q:
                    regen_domain = detect_domain(prev_q) if domain_filter == "Auto" else domain_filter
                    result   = retrieve_and_answer(prev_q.strip(), regen_domain)
                    regen_ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")
                    chat.append(("bot", result, regen_ts))
                    st.rerun()

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)