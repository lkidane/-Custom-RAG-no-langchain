import shutil
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.rag import ask_rag
from generate_embeddings import generate_doc_embeddings

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str


chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)
print("Chroma collection 'documents' initialized")


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/ask")
def ask(req: QueryRequest):
    return ask_rag(req.question)


@app.get("/documents")
def get_documents():
    try:
        result = collection.get(include=["metadatas"])
        docs = sorted({m.get("source", "Unknown") for m in result.get("metadatas", [])})
        return {"documents": docs}
    except Exception as exc:
        return {"error": str(exc)}


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    results = collection.get(where={"source": document_name})
    ids = results.get("ids", [])

    if not ids:
        return {"status": "error", "message": "Document not found"}

    collection.delete(ids=ids)
    return {"status": "success", "deleted_chunks": len(ids)}


UPLOAD_DIR = (Path(__file__).resolve().parent.parent / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = generate_doc_embeddings(str(file_path))
        return {
            "message": "Document uploaded successfully",
            "file_name": file.filename,
            "chunks_created": result.get("chunks", 0),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/documents")
async def upload_document_alias(file: UploadFile = File(...)):
    return await upload_document(file)
