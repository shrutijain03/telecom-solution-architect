"""
ingest.py
─────────
Fetches all WEB_SOURCES, chunks them, and stores in ChromaDB.
No PDFs — knowledge lives entirely in URLs.
Run locally before deploying:  python ingest.py
The resulting ./tmforum_db folder must be committed to GitHub.
"""

import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")   # suppress SSL/urllib3 noise

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import WEB_SOURCES, ENABLE_WEB

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH        = "./tmforum_db"
MAX_CHARS_URL  = 20000   # chars per URL (covers ~5 pages of dense text)
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 60

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Fetch one URL ────────────────────────────────────────────────────────────
def fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, verify=False, headers=HEADERS)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "noscript"]):
            tag.decompose()

        # Prefer main content blocks
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", {"id": "content"}) or
            soup.find("div", {"class": "content"}) or
            soup
        )

        text = main.get_text(separator="\n", strip=True)

        # Collapse blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text  = "\n".join(lines)

        return text[:MAX_CHARS_URL]

    except Exception as e:
        print(f"   ❌ Error fetching {url}: {e}")
        return ""


# ── Load all web documents ───────────────────────────────────────────────────
def load_web_documents() -> list[Document]:
    documents = []
    print(f"\n🌐 Loading {len(WEB_SOURCES)} web sources...")

    for url in WEB_SOURCES:
        print(f"   🌍 {url}")
        content = fetch_url(url)

        if not content.strip():
            print(f"   ⚠️  Skipped (empty): {url}")
            continue

        documents.append(Document(
            page_content=content,
            metadata={"source": "web", "url": url}
        ))
        print(f"   ✅ {len(content):,} chars")

    return documents


# ── Split into chunks ────────────────────────────────────────────────────────
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"\n✂️  Split into {len(chunks)} chunks")
    return chunks


# ── Build / rebuild ChromaDB ─────────────────────────────────────────────────
def create_vector_db(chunks: list[Document]):
    print("\n🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Wipe old collection so we don't accumulate stale chunks
    try:
        old_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        old_db.delete_collection()
        print("🗑️  Cleared old collection")
    except Exception:
        pass

    print("📥 Embedding and storing chunks (this takes a minute)...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    count = vectordb._collection.count()
    print(f"✅ Vector DB ready — {count} vectors stored at {DB_PATH}")
    return vectordb


# ── Main ─────────────────────────────────────────────────────────────────────
def run_ingestion():
    print("\n🔄 Starting web-only ingestion...")

    if not ENABLE_WEB:
        print("⚠️  ENABLE_WEB is False in config.py — nothing to ingest.")
        return

    docs   = load_web_documents()
    print(f"\n📄 Total documents loaded: {len(docs)}")

    chunks = split_documents(docs)
    create_vector_db(chunks)

    print("✅ Ingestion complete\n")


if __name__ == "__main__":
    run_ingestion()