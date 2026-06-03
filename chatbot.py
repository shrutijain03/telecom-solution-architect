import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import os 
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
from datetime import datetime
import uuid
from sentence_transformers import CrossEncoder

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
    TEXT_COLOR = "white"
    SUBTEXT = "#9ca3af"
else:
    PAGE_BG = "#f8fafc"
    TEXT_COLOR = "#111827"
    SUBTEXT = "#6b7280"

# ---------------- STYLE ----------------
st.markdown(f"""
<style>
body, .stApp {{
    background-color: {PAGE_BG};
    color: {TEXT_COLOR};
    font-family: "Segoe UI", sans-serif;
}}

.chat-container {{
    max-width: 600px;
    margin: auto
}}
.user-msg {{
    text-align: right;
    margin: 8px 0;
}}
.user-bubble {{
    display: inline-block;
    background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    color: white;
    padding: 10px 14px;
    border-radius: 18px;
    font-size: 14px;
    max-width: 70%;
}}
.bot-msg {{
    text-align: left;
    margin: 8px 0;
}}
.bot-msg, .user-msg {{
    margin-bottom: 16px;
}}

.bot-bubble {{
    background: """ + ("#1e293b" if dark_mode else "#f9fafb") + """;
    color: { "#e5e7eb" if dark_mode else "#111827" };
    padding: 16px;
    border-radius: 16px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid { "#334155" if dark_mode else "#e5e7eb" };
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    margin-top: 6px;
    white-space: pre-line;
}}
.bot-bubble:hover {{
    transform: translateY(-2px);
}}

section[data-testid="stSidebar"] {{
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
}}

section[data-testid="stSidebar"] h3 {{
    font-size: 14px;
    font-weight: 600;
    color: #374151;
}}

section[data-testid="stSidebar"] button {{
    border-radius: 8px;
}}
.main .block-container {{
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 1rem;
}}

section[data-testid="stSidebar"] {{
    background-color: { "#1f2937" if dark_mode else "#f1f5f9" };
    border-right: 1px solid { "#374151" if dark_mode else "#e5e7eb" };
}}

section[data-testid="stSidebar"] * {{
    color: { "#f9fafb" if dark_mode else "#111827" };
}}
button {{
    border-radius: 6px;
    height: 32px;
    font-size: 12px;
}}

* ✅ ADD HERE */
h1 {{
    margin-bottom: 5px;
}}

p {{
    margin-top: 0px;
}}
small {{
    color: {SUBTEXT};
    font-size: 11px;
}}
.bot-msg b {{
    color: """ + ("#e5e7eb" if dark_mode else "#111827") + """;
}}
textarea, input {{
    background-color: """ + ("#1e293b" if dark_mode else "white") + """;
    color: """ + ("#e5e7eb" if dark_mode else "#111827") + """;
}}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------
st.title("📡 Telecom Co‑Pilot")
st.markdown("""
<div style="text-align:center; margin-bottom:20px">
<h4 style="color:gray">AI Assistant for Telecom Architecture & TM Forum Standards</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Suggested Queries
- Compare eTOM vs ServiceNow data models  
- Map TMF APIs to order-to-cash lifecycle  
- Design telecom OSS for fault management  
- Explain service activation using TMF APIs  
""")

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
You are a Telecom Solution Architect.

Use the context to design a telecom system.

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
You are a Telecom Solution Architect AI.

Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Answer in this format:

📘 Definition:
(1–2 lines)

🔧 Telecom Context:
(1–2 lines)

🏗️ Architecture Relevance:
(1–2 lines)

💡 Example:
(1–2 lines)

🔗 Related APIs:
Keep answer under 100 words.
"""

    model = genai.GenerativeModel("gemini-1.0-pro")

    response = model.generate_content(prompt)

    content = response.text

    if not content:
        return "⚠️ No response generated."

    return content.strip()

# ---------------- INPUT ----------------
default_q = st.session_state.get("prefill", "")
question = st.chat_input("Ask telecom architecture...")

if default_q and not question:
    question = default_q
    st.session_state.prefill = ""

# ---------------- HANDLE QUERY ----------------
if question:
    domain = detect_domain(question)
    ts = datetime.now().strftime("%I:%M %p")

    chat.append(("user", question, ts))

    if chat_data["name"] == "New Chat":
        chat_data["name"] = question[:30]

    domain = detect_domain(question)
    selected = domain if domain_filter == "Auto" else domain_filter

    docs = []
    context = ""
    sources = []
    confidence = "Low"



# ✅ ADD HERE (step 4 integration)
    dconfidence = "Low"
    context = ""

    sources = []
    if docs:
        for d in docs:
            file_name = d.metadata.get("file_name", "Unknown")
            sources.append(f"{d.metadata.get('source')} → {file_name}")

    typing = st.empty()
    typing.markdown("🔍 Searching telecom knowledge...")
    
    typing.markdown(f"🕵🏻 Detected Domain: {domain}")

    typing.markdown("🧠 Generating architecture-aware answer...")

    answer = generate_answer(question, context)

    typing.empty()
    chat.append(("bot", {"text": answer, "sources": sources, "domain": domain,"confidence": confidence}, ts))

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
                {text}
        """, unsafe_allow_html=True)

        # ✅ SOURCES
        if sources:
            st.markdown("<br><b>Sources:</b>", unsafe_allow_html=True)
            for s in sources:
                clean_name = s.replace(".pdf", "").replace("_", " ")
                st.markdown(f"- {clean_name}")

        st.markdown("</div></div>", unsafe_allow_html=True)

        # ✅ BUTTONS INSIDE LOOP ✅
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Regenerate", key=f"regen_{i}"):

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

                chat.append((
                    "bot",
                    {"text": new_answer, "sources": [], "confidence": confidence},
                    datetime.now().strftime("%I:%M %p")
                ))

                st.rerun()

        # ✅ CONFIDENCE DISPLAY
        if confidence:
            st.markdown(f"**Confidence:** {confidence}")

        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)