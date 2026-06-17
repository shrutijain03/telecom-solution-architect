"""
chatbot.py
──────────
Telecom Solution Architect Co-Pilot
• Knowledge source : Pinecone (built from web URLs via ingest.py)
• Groq / Llama3    : structures and polishes the retrieved content ONLY
• RAG              : fully enabled with CrossEncoder reranking
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
from sentence_transformers import CrossEncoder
from groq import Groq
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
    "TMF_APIs":     ["api", "tmf api", "open api", "tmf6", "tmf6", "rest api", "swagger",
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

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vectordb  = load_db()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
reranker  = load_reranker()

# ── Retrieve + rerank ─────────────────────────────────────────────────────────
def get_context(question: str) -> tuple[str, list[str]]:
    """Returns (context_text, list_of_source_urls)."""

    # ✅ Step 1: Refine query
    refined_query = f"""
    Telecom TM Forum OSS BSS context:
    {question}
    Focus on:
    - TMF APIs
    - eTOM processes
    - ServiceNow integration
    - telecom architecture
    """

    # ✅ Step 2: Retrieve documents
    docs = retriever.invoke(refined_query)

    if not docs:
        return "", []

    # ✅ Step 3: Rerank
    pairs = [(refined_query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    # ✅ ✅ YOU WERE MISSING THIS LINE
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    top_docs = []

    # ✅ Step 4: Filtering loop (correct indentation)
    for doc, score in ranked:

        # ✅ Boost PDF priority
        if any(term in question.lower() for term in ["gb921", "tmf070", "blueprint"]):
            if doc.metadata.get("file_name"):
                score += 0.3

        # ✅ Boost keyword match
        if any(word in doc.page_content.lower() for word in question.lower().split()) or \
     any(term in doc.metadata.get("file_name","").lower() for term in question.lower().split()):
          score += 0.2

        # ✅ Filter top chunks
        top_docs = [doc for doc, _ in ranked[:6]]

    # ✅ Step 5: Build context
    context = "\n\n".join(d.page_content for d in top_docs)
    context = context.replace(" .", ".")

    # ✅ DEBUG
    print("\n--- SELECTED SOURCES ---")
    for d in top_docs:
        print(d.metadata)

    # ✅ Step 6: Extract sources
    sources = []

    for d in top_docs:
        if d.metadata.get("url"):
            sources.append(d.metadata["url"])
        elif d.metadata.get("file_name"):
            sources.append(f"PDF: {d.metadata['file_name']}")

    print("\n--- Retrieved context preview ---")
    print(context[:300])

    return context, sources

# ── Derive extra reference URLs from question keywords ────────────────────────
def get_reference_urls(question: str) -> list[str]:
    q = question.lower()
    urls = []
    for keyword, url in DOMAIN_URLS.items():
        if keyword in q:
            urls.append(url)
    return list(dict.fromkeys(urls))   # deduplicate, preserve order

# ── LLM: structure & polish retrieved context ─────────────────────────────────
def generate_answer(question: str, context: str, history: list) -> str:

    import re

    # ✅ CLEAN + LIMIT CONTEXT
    context = re.sub(r"\s+", " ", context)
    context = re.sub(r"[^\x00-\x7F]+", " ", context)
    context = context[:4000]

    prompt = f"""
You are a Telecom Solution Architect AI.

Your job is to answer based on the QUESTION TYPE.

QUESTION:
{question}

CONTEXT:
{context}

---

INSTRUCTIONS:

First, understand the type of question:

1. If it is:
   - "What is"
   - "Explain"
   → Give clear explanation (simple + structured)

2. If the question is about architecture or design:

You MUST provide:

1. Architecture Layers (ALL layers)
2. Component Details (what each layer contains)
3. End-to-End Flow (MANDATORY)
   - Explain how a request moves through the system step-by-step
4. Component Interaction
   - How layers communicate with each other
5. Real Telecom Example (MANDATORY)

IMPORTANT:
- Do NOT stop at listing layers
- Always include flow and interaction
- Always complete the design

3. If it is:
   - "Compare"
   - "Difference"
   - "Differentiate"
   → Provide comparison table or structured differences

4. If it is:
   - "Use cases"
   - "Application"
   → Provide bullet list of use cases

---
RULES:
✅ ONLY compare when the question asks for comparison
✅ Always start with 2-3 lines INTRODUCTION
✅ Always end with a 1-2 line CONCLUSION
✅ DO NOT introduce unrelated concepts  
✅ DO NOT force tables unnecessarily  
✅ Keep answer relevant to the question  
✅ Use context as main reference  

---
STYLE GUIDELINES:

✅ Keep answers clear and structured  
✅ Avoid generic textbook language  
✅ Think like a solution architect  

---
If any section (flow, interaction, or example) is not clearly found in context,
you MUST infer it based on standard telecom architecture principles.
Now generate the best possible answer.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception:
        return "⚠️ Unable to generate response. Please try a simpler query."

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Chats")

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
    st.markdown("### 🎨 Appearance")
    dark_mode = st.toggle("Dark Mode", value=False)
    st.divider()
    st.markdown("### 🗂️ Knowledge Domain")
    domain_filter = st.radio("", ["Auto", "ETOM", "SID", "TMF_APIs", "ODA", "Architecture", "ServiceNow"])
    st.divider()
    st.caption("💡 Answers are grounded in TM Forum & ServiceNow web content. Groq structures the output only.")


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
        "Explain eTOM Level 2 process for Service Problem Management",
        "Which TMF Open APIs are used in order-to-cash lifecycle?",
        "Design an OSS architecture using ODA components",
        "How does ServiceNow TMT integrate with TM Forum APIs?",
        "Compare SID data model with ServiceNow CMDB",
        "What is the ODA Canvas and how does it enable microservices?",
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

    # ── RAG retrieval ──────────────────────────────────────────────────────
    typing = st.empty()
    typing.markdown(f"🔍 Searching knowledge base · Domain: **{domain}**")

    context, rag_sources = get_context(question)

    typing.markdown("🧠 Structuring answer with Groq...")

    # ── Generate (Groq polishes retrieved content) ─────────────────────────
    answer = generate_answer(question, context, chat[:-1])   # exclude current user msg

    typing.empty()

    # ── Collect all source URLs ────────────────────────────────────────────
    kw_sources = get_reference_urls(question)
    all_sources = list(dict.fromkeys(rag_sources + kw_sources))   # RAG sources first

    # Fallback if nothing matched
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

        # Regenerate button
        col1, _ = st.columns([1, 4])
        with col1:
            if st.button("🔄 Regenerate", key=f"regen_{i}"):
                prev_q = next(
                    (chat[j][1] for j in range(i - 1, -1, -1) if chat[j][0] == "user"),
                    None
                )
                if prev_q:
                    regen_context, regen_sources = get_context(prev_q)
                    kw_sources   = get_reference_urls(prev_q)
                    all_regen_sources = list(dict.fromkeys(regen_sources + kw_sources))
                    new_answer   = generate_answer(prev_q, regen_context, chat[:i])
                    regen_ts     = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")
                    chat.append(("bot", {"text": new_answer, "sources": all_regen_sources}, regen_ts))
                    st.rerun()

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)