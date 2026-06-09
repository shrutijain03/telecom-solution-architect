import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import os 
from groq import Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
import uuid
from sentence_transformers import CrossEncoder
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------- SESSION INIT ----------------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = {
        "name": "New Chat",
        "messages": []
    }

# ---------------- DOMAIN DETECTION ----------------
def detect_domain(question):
    q = question.lower()
    if "etom" in q or "process" in q:
        return "ETOM"
    elif "sid" in q:
        return "SID"
    elif "api" in q or "tmf" in q:
        return "TMF_APIs"
    elif "architecture" in q:
        return "Architecture"
    return "All"

 #---------------- CONFIDENCE ----------------
def calculate_confidence(docs):
    if not docs:
        return "Low"
    elif len(docs) >= 2:
        return "High"
    else:
        return "Medium"
# ---------------- ARCHITECTURE DETECTION ----------------
def is_architecture_query(question):
    keywords = [
        "design", "architecture", "build", "implement",
        "solution", "system", "oss", "bss"
    ]
    return any(word in question.lower() for word in keywords)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    # ---------- CHATS ----------
    st.markdown("### Chats")

    if st.button(" New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat = chat_id
        st.session_state.chats[chat_id] = {
            "name": "New Chat",
            "messages": []
        }
        st.rerun()

    st.write("")

    # Chat list
    for chat_id, chat_data in st.session_state.chats.items():
        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(chat_data["name"], key=f"{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id

        with col2:
            if st.button("⋮", key=f"menu_{chat_id}"):
                pass  # optional dropdown later

    st.divider()

    # ---------- APPEARANCE -------
    st.markdown("### Appearance")

    dark_mode = st.toggle("Dark Mode", value=False)

    st.divider()

    # ---------- DOMAIN ----------
    st.markdown("### Knowledge Domain")

    domain_filter = st.radio(
        "",
        ["Auto", "ETOM", "SID", "TMF_APIs", "Architecture"]
    )

# ---------------- CHAT POINTER ----------------
chat_data = st.session_state.chats[st.session_state.current_chat]
chat = chat_data["messages"]

# ---------------- COLORS ----------------
if dark_mode:
    PAGE_BG = "#0f172a"
    CARD_BG = "#1e293b"
    TEXT_COLOR = "#f9fafb"
    SUBTEXT = "#94a3b8"
    BORDER = "#334155"
else:
    PAGE_BG = "#f8fafc"
    CARD_BG = "#f1f5f9"
    TEXT_COLOR = "#111827"
    SUBTEXT = "#6b7280"
    BORDER = "#e5e7eb"

# ---------------- STYLE ----------------
st.markdown(f"""
<style>
body, .stApp {{
    background-color: {PAGE_BG};
    color: {TEXT_COLOR};
    font-family: "Segoe UI", sans-serif;
}}

.chat-container {{
    max-width: 700px;
    margin: auto;
}}

.user-msg {{
    text-align: right;
}}

.user-bubble {{
    display: inline-block;
    background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    color: white;
    padding: 10px 14px;
    border-radius: 18px;
    max-width: 70%;
}}

/* ✅ FIX TIMESTAMP INSIDE USER MESSAGE */
.user-bubble small {{
    color: #e5e7eb;
}}

.bot-bubble {{
    background: {CARD_BG};
    color: {TEXT_COLOR};
    padding: 14px;
    border-radius: 14px;
    border: 1px solid {BORDER};
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}

small {{
    color: {SUBTEXT};
}}

textarea, input {{
    background-color: {CARD_BG};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

section[data-testid="stSidebar"] {{
    background-color: {CARD_BG};
    border-right: 1px solid {BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT_COLOR};
}} 
/* ===== ALL BUTTONS BASE ===== */
.stButton > button {{
    border-radius: 6px !important;
    background-color: """ + ( "#2d3748" if dark_mode else "#f1f5f9" ) + """ !important;
    color: """ + ( "#f9fafb" if dark_mode else "#111827" ) + """ !important;
    border: 1px solid """ + ( "#4b5563" if dark_mode else "#e5e7eb" ) + """ !important;
}}

/* Force text color inside ALL buttons (Streamlit wraps label in <p>) */
.stButton > button p {{
    color: """ + ( "#f9fafb" if dark_mode else "#111827" ) + """ !important;
}}

.stButton > button:hover {{
    background-color: """ + ( "#4b5563" if dark_mode else "#e2e8f0" ) + """ !important;
}}

/* ===== SIDEBAR BUTTONS ===== */
section[data-testid="stSidebar"] .stButton > button {{
    background-color: """ + ( "#2d3748" if dark_mode else "#e5e7eb" ) + """ !important;
    color: """ + ( "#f9fafb" if dark_mode else "#111827" ) + """ !important;
    border: 1px solid """ + ( "#4b5563" if dark_mode else "#d1d5db" ) + """ !important;
    width: 100% !important;
    text-align: left !important;
}}

section[data-testid="stSidebar"] .stButton > button p {{
    color: """ + ( "#f9fafb" if dark_mode else "#111827" ) + """ !important;
}}

section[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: """ + ( "#4b5563" if dark_mode else "#d1d5db" ) + """ !important;
}}

/* ===== SUGGESTED QUERY BUTTONS ===== */
div[data-testid="column"] .stButton > button {{
    background: """ + ( "linear-gradient(135deg, #312e81, #4c1d95)" if dark_mode else "linear-gradient(135deg, #ede9fe, #ddd6fe)" ) + """ !important;
    color: """ + ( "#e0e7ff" if dark_mode else "#4c1d95" ) + """ !important;
    border: 1px solid """ + ( "#4338ca" if dark_mode else "#a78bfa" ) + """ !important;
    font-size: 13px !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 52px !important;
    line-height: 1.4 !important;
}}

div[data-testid="column"] .stButton > button p {{
    color: """ + ( "#e0e7ff" if dark_mode else "#4c1d95" ) + """ !important;
}}

div[data-testid="column"] .stButton > button:hover {{
    opacity: 0.85 !important;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------


st.markdown("""
<h1 style='text-align: center;'>📡 Telecom Solution Architect Co‑Pilot</h1>
<p style='text-align: center; color: gray;'>
AI Assistant for Telecom Architecture, TM Forum & OSS/BSS Design
</p>
""", unsafe_allow_html=True)



# ---------------- SUGGESTED QUERIES ----------------
if not chat:
    st.markdown(f"""
    <p style='text-align:center; font-size:15px; font-weight:500; color:{TEXT_COLOR}; margin-bottom:8px;'>
        💡 Try a suggested query
    </p>
    """, unsafe_allow_html=True)

    suggested = [
        "Compare eTOM vs ServiceNow data models",
        "Map TMF APIs to order-to-cash lifecycle",
        "Design telecom OSS for fault management",
        "Explain service activation using TMF APIs",
    ]

    col1, col2 = st.columns(2)
    for idx, query in enumerate(suggested):
        col = col1 if idx % 2 == 0 else col2
        with col:
            if st.button(query, key=f"suggest_{idx}", use_container_width=True):
                st.session_state.prefill = query
                st.rerun()

# ---------------- LOAD DB ----------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    return Chroma(persist_directory="./tmforum_db", embedding_function=embeddings)

vectordb = None

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

def rerank_docs(question, docs):
    if not docs:
        return docs
    
    pairs = [(question, doc.page_content) for doc in docs if doc.page_content]

    scores = reranker.predict(pairs)

    # attach scores to docs
    scored_docs = list(zip(docs, scores))

    # sort by score (highest first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # return top 2
    return [doc for doc, score in scored_docs[:1]]

# ---------------- LLM ----------------
def generate_answer(question, context):
    if is_architecture_query(question):

        prompt = f"""
You are a Telecom Solution Architect. Always respond using bullet points and clearly separated sections.

Use the context below if available.
If the context is empty, answer based on your telecom knowledge.

Context:
{context}

Question:
{question}

Give output in this format:

🏗️ Architecture Components:
- List key systems (OSS, BSS, APIs, DB, etc.)

🔄 Flow:
- Step-by-step flow of how system works

🔗 APIs:
- Mention TMF APIs used

📊 Integration:
- How systems connect (ServiceNow, CRM, Network)

Keep it practical and concise.
"""

    else:

        prompt = f"""
You are a Telecom Solution Architect AI. Always respond using bullet points and clearly separated sections.

Use the context below if available.
If the context is empty, answer based on telecom domain knowledge.

Context:
{context}

Question:
{question}

Answer STRICTLY in the following format.

📘 Definition:
- 2–3 clear lines

🔧 Telecom Context:
- 2–3 lines with telecom relevance

🏗️ Architecture Relevance:
- Explain importance in system design (2–3 lines)

💡 Example:
- Real telecom use case

🔗 Related APIs:
- List relevant APIs

IMPORTANT:
- Write each section in separate lines
- Do NOT combine sections
- Each section MUST be written as bullet points starting with "-"
- Each section MUST contain at least 3 detailed bullet points.
- Provide detailed explanations, not short summaries.
You MUST always break sections into new lines using newline characters.
"""
    try:
        response = client.chat.completions.create(
            model = "openai/gpt-oss-20b",  
            messages=[
            {"role": "system", "content": "You are a telecom solution architect AI."},
            {"role": "user", "content": prompt}
        ],
            temperature=0.2,
            max_tokens=500
    )
        content = response.choices[0].message.content
        import re

        content = content.strip()

        # ✅ Step 1: break sections into new lines
        content = re.sub(r"(📘|🔧|🏗️|💡|🔗)", r"\n\1", content)

        # ✅ Step 2: split into lines
        lines = content.split("\n")

        formatted_lines = []

        for line in lines:
            line = line.strip()

            if any(symbol in line for symbol in ["📘", "🔧", "🏗️", "💡", "🔗"]):
                formatted_lines.append(f"\n{line}")  # section header
            elif line:
                formatted_lines.append(f"- {line}")  # force bullet

        content = "\n".join(formatted_lines)
        
    except Exception as e:
     return f"⚠️ Groq Error: {str(e)}"


# ---------------- INPUT ----------------
default_q = st.session_state.get("prefill", "")
question = st.chat_input("Ask telecom architecture...")

if default_q and not question:
    question = default_q
    st.session_state.prefill = ""

# ---------------- HANDLE QUERY ----------------
if question:
    domain = detect_domain(question)
    
    ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")


    # ✅ Add user message
    chat.append(("user", question, ts))

    if chat_data["name"] == "New Chat":
        chat_data["name"] = question[:30]

    domain = detect_domain(question)
    selected = domain if domain_filter == "Auto" else domain_filter

    # ✅ No DB (Gemini-only mode)
    docs = []
    context = ""
    confidence = "Low"

    # ✅ Loading message
    typing = st.empty()
    typing.markdown("🔍 Searching telecom knowledge...")
    typing.markdown(f"🕵🏻 Detected Domain: {domain}")
    typing.markdown("🧠 Generating architecture-aware answer...")

    # ✅ Generate answer
    answer = generate_answer(question, context)

    typing.empty()

    # ✅ ✅ GENERATE SOURCES (FIXED POSITION)
    sources = []

    if "etom" in question.lower():
        sources = ["TM Forum eTOM Framework"]
    elif "sid" in question.lower():
        sources = ["TM Forum SID Model"]
    elif "api" in question.lower() or "tmf" in question.lower():
        sources = ["TMF Open APIs"]
    elif "servicenow" in question.lower():
        sources = ["ServiceNow Documentation"]
    else:
        sources = ["Telecom Domain Knowledge (AI Generated)"]

    # ✅ Add bot response
    chat.append((
        "bot",
        {
            "text": answer,
            "sources": sources,
            "domain": domain
        },
        ts
    ))
# ---------------- DISPLAY ----------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for i, (role, msg, ts) in enumerate(chat):

    if role == "user":
        st.markdown(f"""
        <div class="user-msg">
            <div class="user-bubble">
                {msg}<br><small>{ts}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if isinstance(msg, dict):
            text = msg.get("text", "")
            sources = msg.get("sources", [])
            confidence = msg.get("confidence", "")
        else:
            text = msg
            sources = []
            confidence = ""

        # ✅ BOT MESSAGE
        st.markdown(f"""
    <div class="bot-msg">
    <div class="bot-bubble">
        {text}<br>
        <small>{ts}</small>
     """, unsafe_allow_html=True)

        # ✅ SOURCES
        st.markdown("<br><b>🔗 Sources</b>", unsafe_allow_html=True)

        for s in sources:
              st.markdown(
        f"<div style='background:{CARD_BG}; color:{TEXT_COLOR}; padding:6px 10px; border-radius:8px; margin:4px 0; font-size:13px; border:1px solid {BORDER};'>✅ {s}</div>",
        unsafe_allow_html=True
    )
        st.markdown("</div></div>", unsafe_allow_html=True)

        # ✅ BUTTONS INSIDE LOOP ✅
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄Regenerate", key=f"regen_{i}"):

                prev_user_msg = None

                for j in range(i - 1, -1, -1):
                    if chat[j][0] == "user":
                        prev_user_msg = chat[j][1]
                        break

                prev_user_msg = prev_user_msg.strip() if isinstance(prev_user_msg, str) else ""

                if prev_user_msg:

                    docs = []
                    context = ""
                    sources = []
                    confidence = "Low"

                    if docs:
                        docs = rerank_docs(prev_user_msg, docs)
                        confidence = calculate_confidence(docs)
                    else:
                        docs = []

                context = "\n\n".join([d.page_content[:150] for d in docs]) if docs else ""

                new_answer = generate_answer(prev_user_msg, context)

                ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")

                chat.append((
                "bot",
                {"text": new_answer, "sources": [], "confidence": confidence},
                ts
                ))

                st.rerun()

    

        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)