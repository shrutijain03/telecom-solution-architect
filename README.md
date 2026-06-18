# Telecom Solution Architect Co‑Pilot

**Live App:**  
 https://telecom-solution-architect-4cquhhnqfdujgxakdwjqbd.streamlit.app/

---

## Overview

An AI-powered telecom assistant designed to help Solution Architects answer **architecture-level and TM Forum-related questions** quickly.

The system uses a **RAG-based architecture**, combining:

- Groq LLM (fast response generation)  
- Telecom PDFs (domain knowledge)  
- Web URLs (TM Forum + ServiceNow)  

This ensures responses are **accurate, structured, and grounded**.

---

## Project Overview

The assistant can:

- Answer telecom architecture questions  
- Provide structured, domain-aware responses  
- Support multiple TM Forum frameworks  
- Act as a Solution Architect knowledge assistant  

---

## Supported Domains

- ETOM – Business processes  
- SID – Data models  
- TMF APIs – Open APIs  
- Architecture – OSS/BSS design  
- ServiceNow (TMT)  

---

## How It Works (RAG Pipeline)

1. User asks a question  
2. System retrieves relevant data from **Pinecone**  
3. Combines PDF + web context  
4. Sends context to **Groq LLM**  
5. Generates a structured answer  

---

## Key Features

- Auto domain detection  
- Unified PDF + Web knowledge base  
- Fast responses (Groq)  
- Grounded answers (RAG)  
- Source references  
- Regenerate responses  
- Chat history  

---

##  Tech Stack

- Streamlit  
- Groq (Llama 3.1 8B)  
- Pinecone  
- HuggingFace Embeddings  
- Python  

---

## Project Structure
Project/
│
├── chatbot.py
├── ingest.py
├── config.py
├── README.md
│
├── ETOM/
├── SID/
├── TMF_APIs/
├── Architecture/
├── ServiceNow/

---

## Setup & Run

Local Version
```bash
venv\Scripts\activate
pip install -r requirements.txt
streamlit run chatbot.py
  
Deployed Version
https://telecom-solution-architect-4cquhhnqfdujgxakdwjqbd.streamlit.app/

Uses Groq API
No local setup required


Future Enhancements

Add more telecom sources
Improve architecture depth
Generate diagrams automatically


Author
Shruti Jain
Intern
