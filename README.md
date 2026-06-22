# Telecom Solution Architect Co‑Pilot

**Live App:**  
https://telecom-solution-architect-iyg3coig7frdzs34eflfxa.streamlit.app/

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

## Tech Stack

- Vector DB: Pinecone (cloud)
- Embeddings: HuggingFace MiniLM-L6-v2
- LLM: Groq (llama-3.3-70b-instant)
- Sources: PDFs + Web URLs
- UI: Streamlit
- Language: Python 3.11
 
---

## Project Structure
telecom-solution-architect-copilot/
│
├── chatbot.py           # Main Streamlit application
├── ingest.py            # PDF + web ingestion pipeline
├── rag_query.py         # Retrieval and query logic
├── config.py            # Configuration settings
├── scheduler.py         # Auto-ingestion scheduler
│
├── ETOM/                # Business process documents
├── SID/                 # Data model documents
├── SID extra/           # Additional SID references
├── TMF_APIs/            # TM Forum API docs
├── Architecture/        # System architecture docs
│
├── .streamlit/
│   └── secrets.toml     # API keys (not tracked in git)
│
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
├── .gitignore
├── venv/                # Local virtual environment (ignored)
├── pycache/         # Python cache (ignored)


---

## Setup & Run
### Deployed Version
https://telecom-solution-architect-iyg3coig7frdzs34eflfxa.streamlit.app/

- Uses Groq API
- No local setup required

### Local Version(Optional)

```bash
venv\Scripts\activate.bat
pip install -r requirements.txt
python ingest.py
streamlit run chatbot.py
```

## Future Enhancements

- Architecture Automation – Enable auto-generation of architecture diagrams
- Enterprise Integration – Support enterprise knowledge sources (SharePoint, internal docs)
- Expanded Data Sources – Add more standards (3GPP, ETSI) and web sources


# Author
Shruti Jain
(Intern)
