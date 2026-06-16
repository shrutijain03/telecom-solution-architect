# -------------------- WEB-ONLY KNOWLEDGE SOURCES --------------------
# PDFs removed - knowledge comes entirely from URLs below.
# Groq is used ONLY for structuring/polishing answers, NOT as knowledge source.

PDF_SOURCES = {}   # Empty - no PDFs
ENABLE_PDF  = True
ENABLE_WEB  = True

WEB_SOURCES = [

    # ── eTOM / Business Process Framework ──────────────────────────────
    "https://www.tmforum.org/business-process-framework/",

    # ── SID / Information Framework ────────────────────────────────────
    "https://www.tmforum.org/information-framework-sid/",

    # ── TM Forum Open APIs ──────────────────────────────────────────────
    "https://www.tmforum.org/oda/open-apis/",

    # ── TM Forum ODA (Open Digital Architecture) ───────────────────────
    "https://www.tmforum.org/oda/",
    "https://www.tmforum.org/oda/oda-canvas/",

    # ── OSS/BSS Architecture ────────────────────────────────────────────

    # ── ServiceNow Telecom (TMT) ────────────────────────────────────────
    "https://www.servicenow.com/docs/r/telecom-network-inventory/telecommunications-network-inventory/telecom-network-inventory.html",
    "https://www.servicenow.com/docs",
]

PINECONE_API_KEY = "pcsk_3keAtS_Rq93c29chSeB1yL8ja1RWjAKvNPWYsVBsJ5QVqJeY5ehHBYAPLSJ5YedWUfXAL2"
PINECONE_INDEX_NAME = "telecom-copilot"