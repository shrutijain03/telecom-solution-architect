# 📡 Telecom Solution Architect Co‑Pilot

🚀 **Live App:**  
👉 https://telecom-solution-architect-4cquhhnqfdujgxakdwjqbd.streamlit.app/

---

## 🧠 Overview

An AI-powered telecom assistant designed to help Solution Architects answer **architecture-level and TM Forum-related questions** quickly and efficiently.

The system is built using a **dual architecture approach**:

- 🖥️ **Local Version (RAG + Ollama)** → Document-grounded answers  
- 🌐 **Deployed Version (Groq API)** → Fast, scalable AI responses  


## 🎯 Project Overview

This project transforms telecom knowledge into an intelligent assistant that can:

- Answer architecture-level questions  
- Provide **structured and domain-aware responses**  
- Support multiple telecom frameworks  
- Simulate a **Solution Architect knowledge assistant**

---

## 🧠 Supported Knowledge Domains

- 📘 **ETOM** – Business process framework  
- 🧩 **SID** – Information/data model  
- 🔗 **TM Forum APIs** – Open API ecosystem  
- 🏗️ **Architecture** – System design & solution patterns  
- 🧾 **ServiceNow (TMT)** – ITOM, CSM, service workflows  

---

## ⚙️ How It Works

### 🖥️ Local Version (RAG-Based)

1. User submits a question  
2. System detects domain  
3. Retrieves relevant document chunks from vector DB  
4. Passes context to local LLM (Ollama)  
5. Generates **grounded response**

---

### 🌐 Deployed Version (Groq API)

1. User submits a question  
2. System detects domain  
3. Uses **prompt engineering instead of RAG**  
4. Sends request to Groq API  
5. Generates **structured telecom response**

---

## 🧠 Key Features

- 🤖 **Auto Domain Detection**  
- 🧾 **Structured Responses**  
- ⚡ **Fast AI Responses (Groq)**  
- 🧠 **Multi-Domain Telecom Knowledge**  
- 🌗 **Dark Mode UI**  
- 🔄 **Regenerate Responses**  
- 📚 **Source References**  
- ⏱️ **Timestamps & Chat History**

---

## 🧰 Tech Stack

### 🖥️ Local RAG System
- Python  
- LangChain  
- ChromaDB  
- HuggingFace Embeddings  
- Ollama (phi3-mini)  

---

### 🌐 Deployed System
- Streamlit  
- Groq API (Llama3)  
- Prompt Engineering  

---

## 📂 Project Structure
project/
│
├── chatbot.py           # Main chatbot app (Groq deployed version)
├── chatbot_gemini_backup.py   # Backup of Gemini version
├── ingest.py
├── rag_query.py
├── README.md
│
├── ETOM/
├── SID/
├── TMF_APIs/
├── Architecture/
├── ServiceNow/
│
└── tmforum_db/

---

## ⚙️ Setup & Run (Local RAG Version)

```bash
venv\Scripts\activate
pip install -r requirements.txt
python ingest.py
streamlit run chatbot.py


🌐 Deployed Version
👉 Available here:
https://telecom-solution-architect-4cquhhnqfdujgxakdwjqbd.streamlit.app/

Uses Groq API
No local setup required
Accessible via browser


🧠 Design Approach

Uses RAG to reduce hallucination
Combines domain detection + structured prompts
Uses prompt engineering for architecture answers
Balances accuracy (RAG) and scalability (Groq)


🚀 Future Enhancements

Integrate real-time telecom data sources
Enable cloud-based RAG using vector databases (e.g., Pinecone)
Improve UI with architecture diagrams and analytics

✅ Dynamic Knowledge Updates:

Include TM Forum documentation links along with PDFs
Keep knowledge base automatically updated as TM Forum evolves


👩‍💻 Author
Shruti Jain
Intern