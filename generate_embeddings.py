import uuid
from pathlib import Path

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)
print("Chroma collection 'documents' initialized")


def split_text(text, chunk_size=1000, overlap=200):
    if not text or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap

    return chunks


def generate_doc_embeddings(document_path):
    pdf = fitz.open(document_path)
    text = ""

    for page in pdf:
        text += page.get_text()

    pdf.close()

    chunks = split_text(text)
    if not chunks:
        print("No text extracted from PDF")
        return {"chunks": 0, "document": Path(document_path).name}

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    filename = Path(document_path).name

    metadatas = [
        {
            "source": filename,
            "chunk": idx,
        }
        for idx in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"done making the embeddings for {filename}")
    return {"chunks": len(chunks), "document": filename, "ids": ids}


if __name__ == "__main__":
    generate_doc_embeddings(r"C:\Users\lkidane\Downloads\DoYouSpeakGenerativeAI.pdf")
    print("done making the embeddings")
