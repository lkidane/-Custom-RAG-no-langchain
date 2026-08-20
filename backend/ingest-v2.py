import uuid
import fitz
import chromadb
from sentence_transformers import SentenceTransformer

# Embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ChromaDB
chroma = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)

    return chunks


def ingest_pdf(pdf_path):
    pdf = fitz.open(pdf_path)

    text = ""
    for page in pdf:
        text += page.get_text()

    pdf.close()

    if not text.strip():
        print("No text found in PDF")
        return

    chunks = split_text(text)

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]

    filename = pdf_path.split("\\")[-1].split("/")[-1]

    metadatas = [
        {
            "source": filename,
            "chunk": idx
        }
        for idx in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Successfully ingested {filename}"
    )
    print(
        f"Stored {len(chunks)} chunks"
    )


# if __name__ == "__main__":
#     ingest_pdf(
#         r"C:\Users\lkidane\Downloads\MachineLearning-Lecture01.pdf"
#     )