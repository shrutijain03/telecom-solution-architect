import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ------------------------------ FOLDER PATHS ------------------------------

DATA_PATHS = {
    "ETOM": "./ETOM",
    "SID": "./SID",
    "TMF_APIs": "./TMF_APIs",
    "Architecture": "./Architecture",
    "ServiceNow": "./ServiceNow"
}
DB_PATH = "./tmforum_db"


# ------------------------------ LOAD DOCUMENTS ------------------------------

def load_documents():
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

                print(f"   📄 Processing: {file}")  # ✅ ADDED

                loader = PyPDFLoader(file_path)
                pages = loader.load()
                
                for page in pages:
                    page.metadata["source"] = domain
                    page.metadata["file_name"] = file

                documents.extend(pages)

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

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    #vectordb.persist()
    print("✅ Vector DB created successfully!")


# ------------------------------ MAIN ------------------------------

if __name__ == "__main__":
    print("🔄 Loading documents...")
    docs = load_documents()

    print(f"✅ Loaded {len(docs)} pages")

    print("🔄 Splitting documents...")
    chunks = split_documents(docs)

    print(f"✅ Created {len(chunks)} chunks")

    print("🔄 Creating vector database...")
    create_vector_db(chunks)