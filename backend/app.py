from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag import ask_rag

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask(req: QueryRequest):
    
    return ask_rag(req.question)
