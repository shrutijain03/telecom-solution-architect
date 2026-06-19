"""
chatbot.py
──────────
Telecom Solution Architect Co-Pilot
• Knowledge source : Pinecone (built from web URLs + PDFs via ingest.py)
• Groq / Llama3    : structures and polishes the retrieved content ONLY
• RAG              : retrieval from Pinecone, top-3 docs
"""
import streamlit as st
import os

# ✅ Load from Streamlit secrets FIRST
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "telecom-copilot"   # same as ingest.py

# ✅ Then set environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
GROQ_MODEL = "llama-3.1-8b-instant"

# ── URL reference map (for citing sources in replies) ────────────────────────
DOMAIN_URLS = {
    "etom":         "https://www.tmforum.org/business-process-framework/",
    "sid":          "https://www.tmforum.org/information-framework-sid/",
    "api":          "https://www.tmforum.org/oda/open-apis/",
    "tmf":          "https://www.tmforum.org/oda/open-apis/",
    "oda":          "https://www.tmforum.org/oda/",
    "architecture": "https://www.tmforum.org/oda/oda-component-framework/",
    "servicenow":   "https://www.servicenow.com/docs",
    "network inventory": "https://www.servicenow.com/docs/r/telecom-network-inventory/telecommunications-network-inventory/telecom-network-inventory.html",
    "oss":          "https://www.tmforum.org/oda/",
    "bss":          "https://www.tmforum.org/oda/",
    "canvas":       "https://www.tmforum.org/oda/oda-canvas/",
}

# ── Session state ─────────────────────────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.current_chat = cid
    st.session_state.chats[cid] = {"name": "New Chat", "messages": []}


# ── Domain detection ──────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "ETOM":         ["etom", "business process", "fulfillment", "assurance", "billing process",
                     "operations support", "level 1", "level 2", "level 3"],
    "SID":          ["sid", "information framework", "data model", "entity", "shared information"],
    "TMF_APIs":     ["api", "tmf api", "open api", "tmf6", "rest api", "swagger",
                     "order management api", "product catalog api"],
    "ODA":          ["oda", "open digital architecture", "canvas", "component", "oda component"],
    "Architecture": ["architecture", "design", "solution", "oss", "bss", "system design",
                     "integration", "microservice", "cloud native"],
    "ServiceNow":   ["servicenow", "snow", "tmt", "itom", "csm", "network inventory",
                     "telecom service management"],
}

def detect_domain(question: str) -> str:
    q = question.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def is_architecture_query(question: str) -> bool:
    keywords = ["design", "architecture", "build", "implement", "solution",
                "system", "oss", "bss", "how to", "integrate", "deploy"]
    return any(w in question.lower() for w in keywords)


# ── Load DB & reranker (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vector_store

vectordb  = load_db()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

@st.cache_data(ttl=300)
def cached_context(question: str):
    return get_context(question)

def get_context(question: str) -> tuple[str, list[str]]:
    docs = retriever.invoke(question)

    if not docs:
        return "", []

    top_docs = docs[:3]
    context = "\n\n".join(d.page_content for d in top_docs)

    sources = []
    for d in top_docs:
        if d.metadata.get("url"):
            sources.append(d.metadata["url"])
        elif d.metadata.get("file_name"):
            sources.append(f"PDF: {d.metadata['file_name']}")

    return context, list(dict.fromkeys(sources))

# ── Derive extra reference URLs from question keywords ────────────────────────
def get_reference_urls(question: str) -> list[str]:
    q = question.lower()
    urls = []
    for keyword, url in DOMAIN_URLS.items():
        if keyword in q:
            urls.append(url)
    return list(dict.fromkeys(urls))

# ── LLM: structure & polish retrieved context ─────────────────────────────────
def generate_answer(question: str, context: str, history: list) -> str:

    import re

    # Clean + limit context (3000 chars ≈ 750 tokens, leaves room for a full answer)
    context = re.sub(r"\s+", " ", context)
    context = re.sub(r"[^\x00-\x7F]+", " ", context)
    context = context[:3000]

    architecture_mode = is_architecture_query(question)

    if architecture_mode:
        task_instructions = """This is an ARCHITECTURE / DESIGN question.

Design it using telecom standards and frameworks such as:
- TM Forum (eTOM, SID)
- ODA (Open Digital Architecture)

Do NOT use generic software layers (Presentation/Application/Data Layer).
Instead use telecom-specific structure:
- ODA Functional Domains (Core Commerce, Production, Engagement, etc.)
- OSS/BSS system components
- Service Orchestration layers
- Resource / Network layers

Cover, in order:
1. Architecture Components
2. End-to-End Flow (telecom example)
3. Component Interaction
4. Real telecom use-case (e.g. service activation, network issue)"""
    else:
        task_instructions = """This is a DEFINITION / EXPLANATION question.
Give a clear, structured explanation: what it is, why it matters in telecom,
and a short real-world example."""

    prompt = f"""You are a Telecom Solution Architect AI assistant.

Your ONLY knowledge source is the CONTEXT below, retrieved from TM Forum,
ServiceNow, and internal PDF documents. Do NOT use facts from your own
training data — only restructure and polish what is in the context. If the
context lacks detail on something, say so plainly rather than inventing it.

CONTEXT:
{context if context else "⚠️ No relevant context was retrieved for this query."}

QUESTION:
{question}

TASK:
{task_instructions}

Write a complete, well-structured answer using markdown headers and bullet
points. Keep it focused — around 200-300 words — but make sure every section
you start is finished. Do not cut off mid-sentence.

IMPORTANT FOR ALL ANSWERS:
- Where possible, refer to TM Forum concepts or frameworks from the context
- When using frameworks (TM Forum, SID, eTOM, ODA), refer explicitly to them.
- Only mention components that are standard or clearly inferred from telecom systems.
- Avoid inventing component names.
- Prefer phrases like:
  "According to TM Forum..."
  "The framework defines..."
- Avoid generic statements like:
"improves efficiency" unless explained with telecom context
- Be specific to telecom systems
- Only state relationships or roles if you are confident from context.
- Do not infer incorrect relationships (e.g., SID being a process model).
FORMAT:

- Use ONE main heading at the top using (#)
- Use smaller headings (###) for sections
- Do NOT use multiple large headings
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=900   # ✅ was 250 — this was truncating every answer
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Unable to generate response: {e}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Chats")

    if st.button("＋ New Chat", use_container_width=True):
        cid = str(uuid.uuid4())
        st.session_state.current_chat = cid
        st.session_state.chats[cid]   = {"name": "New Chat", "messages": []}
        st.rerun()

    st.write("")
    for cid, cdata in st.session_state.chats.items():
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(cdata["name"], key=cid, use_container_width=True):
                st.session_state.current_chat = cid
        with col2:
            st.button("⋮", key=f"menu_{cid}")

    st.divider()
    st.markdown("### Appearance")
    dark_mode = st.toggle("Dark Mode", value=False)
    st.divider()
    st.markdown("### Knowledge Domain")
    domain_filter = st.radio("", ["Auto", "ETOM", "SID", "TMF_APIs", "ODA", "Architecture", "ServiceNow"])
    st.divider()
    st.caption("💡 Answers are grounded in TM Forum & ServiceNow web content + PDFs. Groq structures the output only.")


# ── Theme ─────────────────────────────────────────────────────────────────────
if dark_mode:
    PAGE_BG, CARD_BG, TEXT_COLOR = "#0f172a", "#1e293b", "#f9fafb"
    SUBTEXT,  BORDER             = "#94a3b8",  "#334155"
else:
    PAGE_BG, CARD_BG, TEXT_COLOR = "#f8fafc", "#f1f5f9", "#111827"
    SUBTEXT,  BORDER             = "#6b7280",  "#e5e7eb"

BTN_BG    = "#2d3748" if dark_mode else "#f1f5f9"
BTN_COLOR = "#f9fafb" if dark_mode else "#111827"
BTN_BDR   = "#4b5563" if dark_mode else "#e5e7eb"
BTN_HOV   = "#4b5563" if dark_mode else "#e2e8f0"
SB_BTN_BG = "#2d3748" if dark_mode else "#e5e7eb"
SB_BTN_BDR= "#4b5563" if dark_mode else "#d1d5db"
SB_HOV    = "#4b5563" if dark_mode else "#d1d5db"
SG_BG     = "linear-gradient(135deg,#312e81,#4c1d95)" if dark_mode else "linear-gradient(135deg,#ede9fe,#ddd6fe)"
SG_COLOR  = "#e0e7ff" if dark_mode else "#4c1d95"
SG_BDR    = "#4338ca" if dark_mode else "#a78bfa"

st.markdown(f"""
<style>
body, .stApp {{background-color:{PAGE_BG};color:{TEXT_COLOR};font-family:"Segoe UI",sans-serif;}}
.chat-container {{max-width:720px;margin:auto;}}
.user-msg {{text-align:right;}}
.user-bubble {{display:inline-block;background:linear-gradient(135deg,#8B5CF6,#6D28D9);color:white;
  padding:10px 14px;border-radius:18px;max-width:75%;}}
.user-bubble small {{color:#e5e7eb;}}
.bot-bubble {{background:{CARD_BG};color:{TEXT_COLOR};padding:14px;border-radius:14px;
  border:1px solid {BORDER};box-shadow:0 4px 12px rgba(0,0,0,0.05);}}
small {{color:{SUBTEXT};}}
section[data-testid="stSidebar"] {{background-color:{CARD_BG};border-right:1px solid {BORDER};}}
section[data-testid="stSidebar"] * {{color:{TEXT_COLOR};}}
.stButton>button {{border-radius:6px!important;background-color:{BTN_BG}!important;
  color:{BTN_COLOR}!important;border:1px solid {BTN_BDR}!important;}}
.stButton>button p {{color:{BTN_COLOR}!important;}}
.stButton>button:hover {{background-color:{BTN_HOV}!important;}}
section[data-testid="stSidebar"] .stButton>button {{background-color:{SB_BTN_BG}!important;
  color:{BTN_COLOR}!important;border:1px solid {SB_BTN_BDR}!important;width:100%!important;text-align:left!important;}}
section[data-testid="stSidebar"] .stButton>button p {{color:{BTN_COLOR}!important;}}
section[data-testid="stSidebar"] .stButton>button:hover {{background-color:{SB_HOV}!important;}}
div[data-testid="column"] .stButton>button {{background:{SG_BG}!important;color:{SG_COLOR}!important;
  border:1px solid {SG_BDR}!important;font-size:13px!important;white-space:normal!important;
  height:auto!important;min-height:52px!important;line-height:1.4!important;}}
div[data-testid="column"] .stButton>button p {{color:{SG_COLOR}!important;}}
div[data-testid="column"] .stButton>button:hover {{opacity:0.85!important;}}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* MAIN TITLE (make this clearly bigger) */
h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    margin-top: 14px !important;
    margin-bottom: 10px !important;
}

/* SECTION HEADINGS */
h2 {
    font-size: 20px !important;
    font-weight: 600 !important;
    margin-top: 12px !important;
}

/* SUB HEADINGS */
h3 {
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* NORMAL TEXT */
p, li {
    font-size: 14px !important;
}

</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;'>📡 Telecom Solution Architect Co‑Pilot</h1>
<p style='text-align:center;color:gray;'>
AI Assistant for Telecom Architecture · TM Forum · OSS/BSS · ServiceNow<br>
""", unsafe_allow_html=True)

# ── Chat pointer ──────────────────────────────────────────────────────────────
chat_data = st.session_state.chats[st.session_state.current_chat]
chat      = chat_data["messages"]

# ── Suggested queries (only on empty chat) ────────────────────────────────────
if not chat:
    st.markdown(f"<p style='text-align:center;font-size:15px;font-weight:500;color:{TEXT_COLOR};margin-bottom:8px;'>💡 Try a suggested query</p>", unsafe_allow_html=True)
    suggested = [
    
        "Which TMF Open APIs are used in order-to-cash lifecycle?",
        "Design an OSS architecture using ODA components",
        "Compare SID data model with ServiceNow CMDB",
        "How does ServiceNow TMT integrate with TM Forum APIs?"
        
    ]
    col1, col2 = st.columns(2)
    for idx, q in enumerate(suggested):
        with (col1 if idx % 2 == 0 else col2):
            if st.button(q, key=f"suggest_{idx}", use_container_width=True):
                st.session_state.prefill = q
                st.rerun()


# ── Input ─────────────────────────────────────────────────────────────────────
prefill  = st.session_state.pop("prefill", "")
question = st.chat_input("Ask telecom architecture...") or (prefill if prefill else None)

# ── Handle query ──────────────────────────────────────────────────────────────
if question:
    domain = domain_filter if domain_filter != "Auto" else detect_domain(question)
    ts     = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")

    chat.append(("user", question, ts))
    if chat_data["name"] == "New Chat":
        chat_data["name"] = question[:35]

    typing = st.empty()
    typing.markdown(f"🔍 Searching knowledge base · Domain: **{domain}**")

    context, rag_sources = cached_context(question)

    typing.markdown("🧠 Structuring answer with Groq...")

    answer = generate_answer(question, context, chat[:-1])

    typing.empty()

    kw_sources = get_reference_urls(question)
    all_sources = list(dict.fromkeys(rag_sources + kw_sources))

    if not all_sources:
        all_sources = ["https://www.tmforum.org/oda/"]

    chat.append(("bot", {"text": answer, "sources": all_sources}, ts))


# ── Render chat ───────────────────────────────────────────────────────────────
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for i, (role, msg, ts) in enumerate(chat):

    if role == "user":
        st.markdown(f"""
        <div class="user-msg">
          <div class="user-bubble">{msg}<br><small>{ts}</small></div>
        </div>""", unsafe_allow_html=True)

    else:
        text    = msg.get("text", "")    if isinstance(msg, dict) else msg
        sources = msg.get("sources", []) if isinstance(msg, dict) else []

        st.markdown(f"""
        <div class="bot-msg">
          <div class="bot-bubble">{text}<br><small>{ts}</small>
        """, unsafe_allow_html=True)

        if sources:
            st.markdown("**🔗 Sources (retrieved from):**")
            for s in sources:
                st.markdown(f"- {s}")

        st.markdown("</div></div>", unsafe_allow_html=True)

        col1, _ = st.columns([1, 4])
        with col1:
            if st.button("🔄 Regenerate", key=f"regen_{i}"):
                prev_q = next(
                    (chat[j][1] for j in range(i - 1, -1, -1) if chat[j][0] == "user"),
                    None
                )
                if prev_q:
                    regen_context, regen_sources = cached_context(prev_q)
                    kw_sources   = get_reference_urls(prev_q)
                    all_regen_sources = list(dict.fromkeys(regen_sources + kw_sources))
                    new_answer   = generate_answer(prev_q, regen_context, chat[:i])
                    regen_ts     = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")
                    chat.append(("bot", {"text": new_answer, "sources": all_regen_sources}, regen_ts))
                    st.rerun()

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)