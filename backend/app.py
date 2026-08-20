import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag import ask_rag
from collections import Counter


app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str


chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="documents",
        metadata={"hnsw:space": "cosine"})
print(f"Chroma collection '{"documents"}' initialized")

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask(req: QueryRequest):
    
    return ask_rag(req.question)


@app.get("/documents")
def get_documents():
    try:
        result = collection.get(
            include=["metadatas"]
        )

        docs = sorted(
            list(
                {
                    m.get("source", "Unknown")
                    for m in result["metadatas"]
                }
            )
        )

        return {"documents": docs}

    except Exception as e:
        return {"error": str(e)}