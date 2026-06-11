"""
ingest.py — Telecom Co-Pilot Knowledge Ingestion
Run once manually:   python ingest.py
Or keep running:     python scheduler.py

All sources and tuning live in config.py — edit that file only.
"""

import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    PDF_SOURCES, WEB_SOURCES,
    ENABLE_PDF, ENABLE_WEB,
    MAX_PDF_PAGES, MAX_WEB_CHARS, WEB_TIMEOUT_SECS, WEB_WORKERS,
    CHUNK_SIZE, CHUNK_OVERLAP,
    DB_PATH, EMBEDDING_MODEL,
)


# ──────────────────────────────────────────────────────────────
#  PDF INGESTION
# ──────────────────────────────────────────────────────────────

def load_pdf_documents() -> list[Document]:
    """Load PDFs from every folder defined in PDF_SOURCES."""
    documents: list[Document] = []
    total_pages = 0

    for domain, path in PDF_SOURCES.items():
        print(f"\n📂 Loading PDFs — domain: {domain}  path: {path}")

        try:
            files = os.listdir(path)
        except FileNotFoundError:
            print(f"   ⚠️  Folder not found, skipping: {path}")
            continue

        pdf_files = [f for f in files if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"   ℹ️  No PDFs found in {path}")
            continue

        for file in pdf_files:
            if total_pages >= MAX_PDF_PAGES:
                print(f"⚠️  Page cap ({MAX_PDF_PAGES}) reached — stopping PDF load early")
                return documents

            file_path = os.path.join(path, file)
            print(f"   📄 {file}")

            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
            except Exception as e:
                print(f"   ❌ Could not load {file}: {e}")
                continue

            for page in pages:
                if total_pages >= MAX_PDF_PAGES:
                    return documents

                page.metadata["source"]    = "pdf"
                page.metadata["domain"]    = domain
                page.metadata["file_name"] = file

                documents.append(page)
                total_pages += 1

    print(f"\n✅ PDFs loaded: {len(documents)} pages")
    return documents


# ──────────────────────────────────────────────────────────────
#  WEB / URL INGESTION
# ──────────────────────────────────────────────────────────────

def _fetch_one_url(url: str) -> Document | None:
    """Fetch and parse a single URL. Returns a Document or None on failure."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (TelecomCoPilot/1.0)"}
        response = requests.get(
            url,
            timeout=WEB_TIMEOUT_SECS,
            verify=False,           # handles self-signed certs on internal sites
            headers=headers,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())          # collapse whitespace
        text = text[:MAX_WEB_CHARS]

        if not text:
            print(f"   ⚠️  Empty content: {url}")
            return None

        return Document(
            page_content=text,
            metadata={"source": "web", "url": url},
        )

    except Exception as e:
        print(f"   ❌ Failed to fetch {url}: {e}")
        return None


def load_web_documents() -> list[Document]:
    """Fetch all URLs from WEB_SOURCES in parallel."""
    if not WEB_SOURCES:
        return []

    print(f"\n🌐 Fetching {len(WEB_SOURCES)} URLs with {WEB_WORKERS} workers...")
    documents: list[Document] = []

    with ThreadPoolExecutor(max_workers=WEB_WORKERS) as executor:
        future_to_url = {executor.submit(_fetch_one_url, url): url for url in WEB_SOURCES}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
                    print(f"   ✅ {url}")
                else:
                    print(f"   ⏭️  Skipped: {url}")
            except Exception as e:
                print(f"   ❌ Error processing {url}: {e}")

    print(f"✅ Web docs loaded: {len(documents)}")
    return documents


# ──────────────────────────────────────────────────────────────
#  CHUNKING
# ──────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────────────────────
#  VECTOR DB
# ──────────────────────────────────────────────────────────────

def create_vector_db(chunks: list[Document]) -> None:
    print("🔄 Building embeddings and writing to ChromaDB...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Wipe old collection so we get a clean rebuild
    try:
        old_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        old_db.delete_collection()
        print("   🗑️  Old collection deleted")
    except Exception:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )
    print("✅ Vector DB written successfully")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    print("\n" + "=" * 60)
    print("  🚀 Telecom Co-Pilot — Knowledge Ingestion")
    print("=" * 60)

    all_docs: list[Document] = []

    if ENABLE_PDF:
        all_docs += load_pdf_documents()
    else:
        print("ℹ️  PDF ingestion disabled (ENABLE_PDF=False in config.py)")

    if ENABLE_WEB:
        all_docs += load_web_documents()
    else:
        print("ℹ️  Web ingestion disabled (ENABLE_WEB=False in config.py)")

    if not all_docs:
        print("⚠️  No documents loaded — check your sources in config.py")
        return

    print(f"\n📦 Total documents: {len(all_docs)}")
    chunks = split_documents(all_docs)
    create_vector_db(chunks)

    print("\n✅ Ingestion complete\n")


if __name__ == "__main__":
    run_ingestion()