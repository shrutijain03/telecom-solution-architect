"""
rag_query.py
────────────
Test retrieval from Pinecone
"""

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


# ── Load embeddings ─────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── Connect Pinecone ────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

vectordb = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})


# ── Test query ──────────────────────────────────────────────────────────────
query = "Which eTOM process handles Service Problem Management?"

docs = retriever.invoke(query)

print(f"\nQUESTION:\n{query}\n")

for i, doc in enumerate(docs, 1):
    print(f"--- Result {i} ---")

    source = doc.metadata.get("source")
    file_name = doc.metadata.get("file_name")
    url = doc.metadata.get("url")

    print(f"Source: {source}")
    if file_name:
        print(f"File: {file_name}")
    if url:
        print(f"URL: {url}")

    print(doc.page_content[:500])
    print()
