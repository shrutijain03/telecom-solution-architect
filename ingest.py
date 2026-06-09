import os
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ------------------------------ FOLDER PATHS ------------------------------

DATA_PATHS = {
    "ETOM": "./ETOM",
    "SID": "./SID",
    "TMF_APIs": "./TMF_APIs",
    "Architecture": "./Architecture",
    "ServiceNow": "./ServiceNow"
}

DB_PATH = "./tmforum_db"

# ✅ ADD WEB LINKS HERE
WEB_URLS = [
    #TM FORUM
    "https://www.tmforum.org/oda/open-apis/",
    "https://www.tmforum.org/open-digital-architecture/open-apis",
    "https://github.com/tmforum-apis",
    #ServiceNow
 "https://www.servicenow.com/standard/resource-center/data-sheet/ds-telecommunications-network-inventory.html",
 "https://www.servicenow.com/docs/r/telecom-network-inventory/telecommunications-network-inventory/telecom-network-inventory.html",
"https://www.servicenow.com/docs/"
]
# ------------------------------ LOAD PDF DOCUMENTS ------------------------------

def load_pdf_documents():
    documents = []

    for domain, path in DATA_PATHS.items():
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

                for page in pages:
                    page.metadata["source"] = domain
                    page.metadata["file_name"] = file

                documents.extend(pages)

    return documents


# ------------------------------ LOAD WEB DOCUMENTS ------------------------------

def load_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    text = soup.get_text(separator=" ")
    text = text[:20000]
    return text


def load_web_documents():
    documents = []

    print("\n🌐 Loading web documents...")

    for url in WEB_URLS:
        try:
            print(f"   🌍 Fetching: {url}")

            content = load_web_content(url)

            doc = Document(
                page_content=content,
                metadata={
                    "source": "web",
                    "url": url
                }
            )

            documents.append(doc)

        except Exception as e:
            print(f"❌ Failed: {url} → {e}")

    return documents


# ------------------------------ SPLIT DOCUMENTS ------------------------------

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# ------------------------------ CREATE VECTOR DB ------------------------------

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ IMPORTANT: delete old DB (for dynamic update)
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    vectordb.delete_collection()

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