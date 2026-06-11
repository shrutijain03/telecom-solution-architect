# ==============================================================================
#  Telecom Co-Pilot — Central Configuration
#  Edit ONLY this file to add / remove sources. No other file needs changing.
# ==============================================================================

# ──────────────────────────────────────────────
#  PDF SOURCES
#  Key   → domain label shown in the UI
#  Value → folder path (relative or absolute)
#
#  To ADD a domain:   add a new key/value pair below
#  To REMOVE a domain: delete or comment out the line
#  To DISABLE all PDFs: set ENABLE_PDF = False
# ──────────────────────────────────────────────
PDF_SOURCES = {
    "ETOM":         "./ETOM",
    "SID":          "./SID",
    "TMF_APIs":     "./TMF_APIs",
    "Architecture": "./Architecture",
    "ServiceNow":   "./ServiceNow",
    # "MyNewDomain": "./MyNewDomain",   ← example: just uncomment to add
}

# ──────────────────────────────────────────────
#  WEB / URL SOURCES
#  Add any public URL here — HTML pages, docs, GitHub, etc.
#  Each URL is scraped, chunked, and stored in the vector DB.
#
#  To ADD a URL:    append it to the list
#  To REMOVE a URL: delete or comment out the line
#  To DISABLE all URLs: set ENABLE_WEB = False
# ──────────────────────────────────────────────
WEB_SOURCES = [
    "https://www.tmforum.org/oda/open-apis/",
    "https://www.tmforum.org/open-digital-architecture/open-apis/",
    "https://github.com/tmforum-apis",
    "https://www.servicenow.com/docs/",
    # "https://your-internal-wiki.example.com/telecom",  ← example
]

# ──────────────────────────────────────────────
#  FEATURE FLAGS
# ──────────────────────────────────────────────
ENABLE_PDF = True       # Set False to skip all PDF ingestion
ENABLE_WEB = True       # Set False to skip all URL ingestion

# ──────────────────────────────────────────────
#  PERFORMANCE TUNING
# ──────────────────────────────────────────────
MAX_PDF_PAGES    = 2000   # Hard cap on total PDF pages ingested
MAX_WEB_CHARS    = 15000  # Max characters scraped per URL
WEB_TIMEOUT_SECS = 10     # HTTP request timeout per URL
WEB_WORKERS      = 4      # Parallel threads for URL fetching

CHUNK_SIZE       = 400    # Characters per chunk (smaller = faster retrieval)
CHUNK_OVERLAP    = 50     # Overlap between chunks

RETRIEVER_K      = 4      # Docs fetched from vector DB per query
RERANKER_TOP_N   = 2      # Docs kept after reranking

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
DB_PATH          = "./tmforum_db"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL        = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────
#  SCHEDULER
# ──────────────────────────────────────────────
REFRESH_INTERVAL_HOURS = 24   # How often scheduler re-ingests sources