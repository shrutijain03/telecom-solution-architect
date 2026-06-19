"""
ingest.py
---------
Loads PDFs + URLs → creates embeddings → stores in Pinecone

Run:
    python ingest.py
"""

import os
import re
import requests
from bs4 import BeautifulSoup

from config import PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from config import (
    WEB_SOURCES,
    ENABLE_WEB,
    ENABLE_PDF,
    PINECONE_INDEX_NAME
)

# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
MAX_CHARS_URL = 20000
PDF_FOLDERS = ["Architecture", "ETOM", "SID", "TMF_APIs"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ── Load Web ─────────────────────────────────────────────────────────
def fetch_url(url):
    try:
        resp = requests.get(url, timeout=15, verify=False, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:MAX_CHARS_URL]
    except Exception as e:
        print(f"❌ URL error {url}: {e}")
        return ""


def load_web_documents():
    documents = []
    if not ENABLE_WEB:
        return documents

    print(f"\n🌐 Loading {len(WEB_SOURCES)} URLs...")
    for url in WEB_SOURCES:
        text = fetch_url(url)
        if not text.strip():
            print(f"⚠️  Skipped (empty): {url}")
            continue

        documents.append(
            Document(page_content=text, metadata={"url": url, "source": "web"})
        )
        print(f"✅ {url}")

    return documents


# ── Load PDFs ─────────────────────────────────────────────────────────
def load_pdf_documents():
    """
    Loads every PDF in PDF_FOLDERS. Each PAGE becomes its own Document
    (not just the last page) — this was the bug: append() was previously
    outside the loops, so only one page total ever made it into Pinecone.
    """
    documents = []
    if not ENABLE_PDF:
        return documents

    print("\n📄 Loading PDFs...")

    for folder in PDF_FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue

        for file in os.listdir(folder):
            if not file.endswith(".pdf"):
                continue

            path = os.path.join(folder, file)

            try:
                loader = PyPDFLoader(path)
                pages = loader.load()
            except Exception as e:
                print(f"❌ {file}: failed to load — {e}")
                continue

            pages_kept = 0
            for page in pages:
                text = page.page_content
                text = re.sub(r"\s+", " ", text)
                text = text.strip()

                if len(text) < 50:
                    continue   # skip near-empty pages

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"file_name": file, "source": "pdf"}
                    )
                )
                pages_kept += 1

            print(f"✅ {file} ({pages_kept}/{len(pages)} pages kept)")

    return documents


# ── Split ─────────────────────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"\n✂️  {len(chunks)} chunks created")
    return chunks


# ── Store in Pinecone ────────────────────────────────────────────────
def create_vector_db(chunks):
    print("\n🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("🆕 Creating index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Clear old vectors so re-running doesn't pile up duplicates/stale data
    print("🗑️  Clearing old vectors...")
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
    except Exception as e:
        print(f"   (nothing to clear, or: {e})")

    print("📥 Uploading to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("✅ Pinecone ready!")


# ── Main ─────────────────────────────────────────────────────────────
def run_ingestion():
    print("🔄 Starting ingestion...")

    docs = []
    docs.extend(load_web_documents())
    docs.extend(load_pdf_documents())

    print(f"\n✅ Loaded {len(docs)} documents total")

    if not docs:
        print("❌ No data found — nothing to ingest")
        return

    chunks = split_documents(docs)
    create_vector_db(chunks)

    print("✅ Ingestion complete!\n")


if __name__ == "__main__":
    run_ingestion()