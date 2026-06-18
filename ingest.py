"""
ingest.py
---------
Loads PDFs + URLs → creates embeddings → stores in Pinecone
"""

import os
from config import PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
import requests
from bs4 import BeautifulSoup
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

# ✅ config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 60
PDF_FOLDERS = ["Architecture", "ETOM", "SID", "TMF_APIs"]

# ── Load Web ─────────────────────────────────────────────────────────
def fetch_url(url):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except:
        return ""

def load_web_documents():
    documents = []

    for url in WEB_SOURCES:
        text = fetch_url(url)
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "url": url,
                        "source": "web"
                    }
                )
            )
    return documents

# ── Load PDFs ─────────────────────────────────────────────────────────
def load_pdf_documents():
    documents = []

    for folder in PDF_FOLDERS:
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            if file.endswith(".pdf"):
                path = os.path.join(folder, file)

                loader = PyPDFLoader(path)
                import re

                pages = loader.load()

                for page in pages:
                  text = page.page_content

    # ✅ CLEAN PDF TEXT
                  text = re.sub(r"\s+", " ", text)         # remove extra spaces
                  text = re.sub(r"\n", " ", text)          # remove line breaks
                  text = text.strip()

                if len(text) < 50:
                  continue   # skip useless chunks

    documents.append(
        Document(
            page_content=text,
            metadata={
                "file_name": file,
                "source": "pdf"
            }
        )
    )

    return documents

# ── Split ─────────────────────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# ── Store in Pinecone ────────────────────────────────────────────────
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

# ── Main ─────────────────────────────────────────────────────────────
def run_ingestion():
    print("🔄 Starting ingestion...")

    docs = []

    if ENABLE_WEB:
        docs.extend(load_web_documents())

    if ENABLE_PDF:
        docs.extend(load_pdf_documents())

    print(f"✅ Loaded {len(docs)} documents")

    chunks = split_documents(docs)

    print(f"✅ Created {len(chunks)} chunks")

    create_vector_db(chunks)

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
