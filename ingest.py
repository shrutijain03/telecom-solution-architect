"""
ingest.py
─────────
Loads PDFs + URLs → creates embeddings → stores in Pinecone

Run:
    python ingest.py
"""
import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
import os
from config import PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

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
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)


# ── Config ──────────────────────────────────────────────────────────────────
MAX_CHARS_URL  = 20000
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 60
PDF_FOLDERS = ["Architecture", "ETOM", "SID", "TMF_APIs"]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ── Fetch URL ───────────────────────────────────────────────────────────────
def fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, verify=False, headers=HEADERS)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text[:MAX_CHARS_URL]

    except Exception as e:
        print(f"❌ URL error {url}: {e}")
        return ""


# ── Load WEB ────────────────────────────────────────────────────────────────
def load_web_documents():
    documents = []

    if not ENABLE_WEB:
        return documents

    print(f"\n🌐 Loading {len(WEB_SOURCES)} URLs...")

    for url in WEB_SOURCES:
        content = fetch_url(url)

        if not content.strip():
            continue

        documents.append(Document(
            page_content=content,
            metadata={"source": "web", "url": url}
        ))

        print(f"✅ {url}")

    return documents


# ── Load PDFs ───────────────────────────────────────────────────────────────
def load_pdf_documents():
    documents = []

    if not ENABLE_PDF:
        return documents

    print("\n📄 Loading PDFs...")

    for folder in PDF_FOLDERS:
        path = os.path.join("./", folder)

        if not os.path.exists(path):
            continue

        for file in os.listdir(path):
            if file.endswith(".pdf"):
                file_path = os.path.join(path, file)

                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    for d in docs:
                        d.metadata["source"] = "pdf"
                        d.metadata["file_name"] = file

                    documents.extend(docs)

                    print(f"✅ {file}")

                except Exception as e:
                    print(f"❌ {file}: {e}")

    return documents


# ── Split ───────────────────────────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    print(f"\n✂️ {len(chunks)} chunks created")
    return chunks


# ── Pinecone store ──────────────────────────────────────────────────────────
def create_vector_db(chunks):

    print("\n🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    import urllib3
    urllib3.disable_warnings()

    pc = Pinecone(
    api_key=PINECONE_API_KEY,
    ssl_verify=False   # 🔥 force bypass SSL
)


    # Create index if not exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("🆕 Creating index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print("📥 Uploading to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print("✅ Pinecone ready!")


# ── Main ────────────────────────────────────────────────────────────────────
def run_ingestion():
    print("\n🔄 Starting ingestion...")

    documents = []
    documents.extend(load_web_documents())
    documents.extend(load_pdf_documents())

    print(f"\n📄 Total documents: {len(documents)}")

    if not documents:
        print("❌ No data found")
        return

    chunks = split_documents(documents)
    create_vector_db(chunks)

    print("✅ DONE\n")


if __name__ == "__main__":
    run_ingestion()