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

#--------------------------URL SOURCES-------------------------
URL_SOURCES = {
    "network inventory": [
        "https://www.servicenow.com/docs/r/telecom-network-inventory/telecommunications-network-inventory/telecom-network-inventory.html"
    ],
    "tmf api": [
        "https://www.tmforum.org/oda/open-apis/"
    ],
    "etom": [
        "https://www.tmforum.org/business-process-framework-etom/"
    ],
    "sid": [
        "https://www.tmforum.org/information-framework-sid/"
    ],
    "architecture": [
        "https://www.tmforum.org/oda/"
    ]
}
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


vectordb = load_db()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

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
- Each section MUST contain at least 2 detailed bullet points.
- Provide detailed explanations, not short summaries.
You MUST always break sections into new lines using newline characters.
"""
    try:
        response = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",  
            messages=[
            {"role": "system", "content": "You are a telecom solution architect AI. Provide detailed answers with clear sections and bullet points."},
            {"role": "user", "content": prompt}
        ],
            temperature=0.5,
            max_tokens=800
    )
        content = response.choices[0].message.content if response.choices else ""

        if not content:
           return "⚠️ No response generated."

        import re

        content = content.strip()
        content = re.sub(r"(📘|🔧|🏗️|💡|🔗)", r"\n\1", content)
        
       
        lines = content.split("\n")

        formatted = []

        for line in lines:
            line = line.strip()

            if any(sym in line for sym in ["📘", "🔧", "🏗️", "💡", "🔗"]):
                formatted.append(f"\n{line}")  # section header
            elif line: 
                 if line.startswith("-"):
                  formatted.append(line)
                 else:
                  formatted.append(f"- {line}")

        content = "\n".join(formatted)
        
    except Exception as e:
     return f"⚠️ Groq Error: {str(e)}"
    return content

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

    # ✅ RETRIEVE DOCUMENTS (RAG ✅)
    #docs = retriever.invoke(question)
    #docs = rerank_docs(question, docs)

    #for d in docs:
        #print("SOURCE:", d.metadata)

    # ✅ BUILD CONTEXT ✅
    context = "Answer using telecom standards from TM Forum and ServiceNow best practices."
    #context = "\n\n".join([doc.page_content[:300] for doc in docs])
    #confidence = calculate_confidence(docs)

    # ✅ LOADING UI
    typing = st.empty()
    typing.markdown("🔍 Searching telecom knowledge...")
    typing.markdown(f"🕵🏻 Detected Domain: {domain}")
    typing.markdown("🧠 Generating architecture-aware answer...")

    # ✅ PASS CONTEXT TO LLM ✅ (THIS IS THE KEY LINE)
    answer = generate_answer(question, context)

    typing.empty()

    # ✅ EXTRACT REAL SOURCES ✅
    sources = []

q = question.lower()

for key, links in URL_SOURCES.items():
    if key in q:
        sources.extend(links)

        if not sources:
         sources = [
        "https://www.tmforum.org/oda/"
    ]
        if sources:
          st.markdown("**🔗 Sources:**")
    for s in sources:
        st.markdown(f"- {s}")

    # ✅ ADD BOT RESPONSE
    chat.append((
        "bot",
        {
            "text": answer,
            "sources": sources,
            #"confidence": confidence
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

        # ✅ SOURCES — correctly inside the else (bot) block
        if sources:
            st.markdown("**🔗 Sources:**")
            for s in sources:
                st.markdown(f"- {s}")

        if confidence:
            st.caption(f"Confidence: {confidence}")

        st.markdown("</div></div>", unsafe_allow_html=True)

        # ✅ REGENERATE BUTTON — only for bot messages
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Regenerate", key=f"regen_{i}"):
                prev_user_msg = None

                for j in range(i - 1, -1, -1):
                    if chat[j][0] == "user":
                        prev_user_msg = chat[j][1]
                        break

                prev_user_msg = prev_user_msg.strip() if isinstance(prev_user_msg, str) else ""

                if prev_user_msg:
                    # ✅ FIX: actually retrieve docs for regeneration
                    regen_docs = retriever.invoke(prev_user_msg)
                    regen_docs = rerank_docs(prev_user_msg, regen_docs)
                    regen_context = "\n\n".join([d.page_content[:300] for d in regen_docs])
                    regen_confidence = calculate_confidence(regen_docs)
                    regen_sources = []
                    for doc in regen_docs:
                        if "url" in doc.metadata:
                            regen_sources.append(doc.metadata["url"])
                        else:
                            regen_sources.append(doc.metadata.get("file_name", "PDF"))

                    new_answer = generate_answer(prev_user_msg, regen_context)
                    regen_ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M %p")

                    chat.append((
                        "bot",
                        {"text": new_answer, "sources": regen_sources, "confidence": regen_confidence},
                        regen_ts
                    ))
                    st.rerun()

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)