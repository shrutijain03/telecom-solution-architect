from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the embeddings used during ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the ChromaDB
vectordb = Chroma(
    persist_directory="./tmforum_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# Query
query = "Which eTOM process handles Service Problem Management?"

# ✅ NEW API (LangChain >= 0.2)
docs = retriever.invoke(query)

print(f"\nQUESTION: {query}\n")

for i, doc in enumerate(docs, 1):
    print(f"--- Result {i} ---")
    print(f"Source: {doc.metadata.get('source')} | File: {doc.metadata.get('file_name')}")
    print(doc.page_content[:500])
    print()