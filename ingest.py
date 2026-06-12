import os
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import PDF_SOURCES, WEB_SOURCES, ENABLE_PDF, ENABLE_WEB

# ------------------------------ CONFIG ------------------------------

DB_PATH = "./tmforum_db"

# ✅ PERFORMANCE LIMIT (important)
MAX_PDF_PAGES = 2000


# ------------------------------ LOAD PDF DOCUMENTS ------------------------------

def load_pdf_documents():
    documents = []
    total_pages = 0

    for domain, path in PDF_SOURCES.items():
        print(f"\n📂 Loading from: {domain} ({path})")

        try:
            files = os.listdir(path)
        except:
            print(f"❌ Path not found: {path}")
            continue

        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(path, file)
                print(f"   📄 Processing: {file}")

                loader = PyPDFLoader(file_path)
                pages = loader.load()

                # ✅ LIMIT PAGES (performance boost)
                for page in pages:
                    if total_pages >= MAX_PDF_PAGES:
                        print("⚠️ PDF limit reached, stopping early")
                        return documents

                    page.metadata["source"] = "pdf"
                    page.metadata["file_name"] = file

                    documents.append(page)
                    total_pages += 1

    return documents

# ------------------------------ LOAD WEB DOCUMENTS ------------------------------

def load_web_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            timeout=10,              # ✅ FIX hang
            verify=False,            # ✅ FIX SSL error
            headers=headers
        )

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.extract()

        text = soup.get_text(separator=" ")

        # ✅ LIMIT SIZE
        text = text[:15000]

        return text

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return ""


def load_web_documents():
    documents = []

    print("\n🌐 Loading web documents...")

    for url in WEB_SOURCES:
        print(f"   🌍 Fetching: {url}")

        content = load_web_content(url)

        if not content.strip():
            print(f"⚠️ Skipped empty: {url}")
            continue

        doc = Document(
            page_content=content,
            metadata={
                "source": "web",
                "url": url
            }
        )

        documents.append(doc)

    return documents


# ------------------------------ SPLIT DOCUMENTS ------------------------------

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,   # ✅ smaller = faster
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


# ------------------------------ CREATE VECTOR DB ------------------------------

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    try:
        vectordb.delete_collection()
    except:
        pass

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("✅ Vector DB updated successfully!")

    # ------------------------------ MAIN INGEST FUNCTION ------------------------------

def run_ingestion():
    print("\n🔄 Starting full ingestion...")

    pdf_docs = load_pdf_documents()
    web_docs = load_web_documents()

    all_docs = pdf_docs + web_docs

    print(f"\n✅ Total documents loaded: {len(all_docs)}")

    print("🔄 Splitting documents...")
    chunks = split_documents(all_docs)

    print(f"✅ Created {len(chunks)} chunks")

    print("🔄 Creating vector database...")
    create_vector_db(chunks)

    print("✅ Ingestion complete\n")


# ------------------------------ MAIN ------------------------------

if __name__ == "__main__":
    run_ingestion()
