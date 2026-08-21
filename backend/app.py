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

# Delete document from the collection
# @app.delete("/documents/{document_path}")
# def delete_document(document_path: str):
#     from backend.delete_document import delete_document

#     success = delete_document(document_path)

#     if success:
#         return {"message": f"Document '{document_path}' deleted successfully."}
#     else:
#         return {"error": f"Failed to delete document '{document_path}'."}           

@app.delete("/documents/{document_name}")
def delete_document(document_name: str):

    results = collection.get(
        where={"source": document_name}
    )

    ids = results.get("ids", [])

    if not ids:
        return {
            "status": "error",
            "message": "Document not found"
        }

    collection.delete(ids=ids)

    return {
        "status": "success",
        "deleted_chunks": len(ids)
    }


# results = collection.get(
#             where={"source": document_path}
#         )

#         ids = results.get("ids", [])


######### Upload document endpoint #############

from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from generate_embeddings import generate_doc_embeddings

# app = FastAPI()

# UPLOAD_DIR = Path("uploads")
# UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    
    try:

        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        file_path = UPLOAD_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = generate_doc_embeddings(str(file_path))

        return {
            "message": "Document uploaded successfully",
            "file_name": file.filename,
            "chunks_created": result["chunks"]
        }

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )