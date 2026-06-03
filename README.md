# 🧠 Telecom Solution Architect Co‑Pilot

An AI-powered Retrieval-Augmented Generation (RAG) application designed to assist telecom solution architects by providing structured, domain-aware answers across TM Forum frameworks and ServiceNow modules.

---

## 🎯 Project Overview

This project transforms telecom documentation into an intelligent assistant that can:

- Answer architecture-level questions
- Provide structured and concise explanations
- Retrieve accurate information from multiple telecom domains
- Simulate a knowledge assistant for Solution Architects

---

## 🧠 Supported Knowledge Domains

The system integrates multiple telecom knowledge sources:

- 📘 **ETOM** – Business process framework  
- 🧩 **SID** – Information/data model  
- 🔗 **TM Forum APIs** – Open API ecosystem  
- 🏗️ **Architecture** – System design + deployment blueprints  
- 🧾 **ServiceNow (TMT)** – ITOM, CSM, and service management  

---

## ⚙️ How It Works

1. User submits a question  
2. System automatically detects the relevant domain  
3. Retrieves top relevant document chunks from vector database  
4. Passes context to LLM (local model via Ollama)  
5. Generates structured and concise response  

---

## 🧠 Key Features

- 🤖 **Auto Domain Detection**
- 🔍 **Context-Based Retrieval (RAG)**
- 🧾 **Structured Responses (Overview, Key Points, Explanation)**
- ⚡ **Fast Context Optimization**
- 🧠 **Multi-Domain Knowledge Integration**
- 🌓 **Dark Mode UI**
- 📊 **Confidence Indicator**
- 📚 **Source References**

---

## 🧰 Tech Stack

- **Python**
- **Streamlit** → Frontend UI  
- **LangChain** → RAG pipeline  
- **ChromaDB** → Vector database  
- **HuggingFace Embeddings** → Text embeddings  
- **Ollama (phi3-mini)** → Local LLM  

---

## 📂 Project Structure
project/
│
├── chatbot.py           # Main chatbot application
├── ingest.py            # Data ingestion script
├── rag_query.py         # Testing retrieval
├── README.md            # Project documentation
│
├── ETOM/                # ETOM documents
├── SID/                 # SID documents
├── TMF_APIs/            # TM Forum API docs
├── Architecture/        # Architecture documents
├── ServiceNow/          # ServiceNow TMT documents
│
└── tmforum_db/          # Vector database

---

## ⚙️ Setup & Run Instructions

### 1️⃣ Activate virtual environment

```bash
venv\Scripts\activate


2️⃣ Install dependencies
Shellpip install -r requirements.txtShow more lines

3️⃣ Run ingestion (build knowledge base)
Shellpython ingest.pyShow more lines

4️⃣ Start chatbot
Shellstreamlit run chatbot.pyShow more lines

⚠️ Notes

The system runs locally using Ollama (LLM)
Requires initial embedding time depending on dataset size
Large document sets may increase response time without optimization


🧠 Design Approach

Uses RAG architecture to avoid hallucination
Combines domain filtering + embedding search
Applies prompt engineering for structured output
Maintains balance between accuracy and response speed


🚀 Future Improvements

✅ Optimize chunk size for performance
✅ Add conversational memory (multi-turn context)
✅ Support document upload (dynamic ingestion)
✅ Deploy using cloud LLM for public access
✅ Add cross-domain comparison (ETOM vs ServiceNow)


🎤 Demo Highlights

Automatic domain identification
Structured architecture-level answers
Multi-domain telecom knowledge retrieval
Clean, modern UI with chat interface


👩‍💻 Author
Shruti Jain
Intern
