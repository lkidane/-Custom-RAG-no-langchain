import chromadb
from ollama import Client
from sentence_transformers import SentenceTransformer

ollama_client = Client(host="http://localhost:11434")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)


def ask_rag(question: str):
    try:
        if not question or not question.strip():
            return {"answer": "Please provide a question.", "citations": {}}

        query_embedding = embedding_model.encode(question, convert_to_numpy=True).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return {"answer": "No relevant documents found.", "citations": {}}

        context = "\n\n".join(documents)
        response = ollama_client.chat(
            model="mymodel",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only using the supplied context. "
                        "If the answer is not in the context, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}
""",
                },
            ],
        )

        citations = {
            f"DOC{idx + 1}": {
                "source": meta.get("source", "Unknown"),
                "chunk": meta.get("chunk", "Unknown"),
            }
            for idx, meta in enumerate(metadatas)
        }

        return {"answer": response["message"]["content"], "citations": citations}
    except Exception as exc:
        return {"answer": f"Error: {exc}", "citations": {}}
