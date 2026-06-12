# ==============================================================================
#  Telecom Co-Pilot — Configuration
#  Edit ONLY this file to add/remove URL sources. No other file needs changing.
# ==============================================================================

# ── URL SOURCES ────────────────────────────────────────────────────────────────
# Add any public URL here. The app will scrape these at startup and use the
# content as context for answering questions.
#
# To ADD a source:    append a new dict to the list
# To REMOVE a source: delete or comment out the entry
# ──────────────────────────────────────────────────────────────────────────────
URL_SOURCES = [
    {
        "label": "TM Forum Open APIs",
        "url": "https://www.tmforum.org/oda/open-apis/",
        "domain": "TMF_APIs"
    },
    {
        "label": "TM Forum Open Digital Architecture",
        "url": "https://www.tmforum.org/open-digital-architecture/",
        "domain": "Architecture"
    },
    {
        "label": "TM Forum GitHub APIs",
        "url": "https://github.com/tmforum-apis",
        "domain": "TMF_APIs"
    },
    {
        "label": "ServiceNow Telecom Docs",
        "url": "https://www.servicenow.com/docs/",
        "domain": "ServiceNow"
    }
]
# ── SCRAPING SETTINGS ─────────────────────────────────────────────────────────
MAX_CHARS_PER_URL = 8000    # Max characters to keep per scraped page
WEB_TIMEOUT_SECS  = 10      # HTTP timeout per URL

# ── LLM SETTINGS ─────────────────────────────────────────────────────────────
LLM_MODEL  = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000