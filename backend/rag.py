import chromadb
from ollama import Client
from sentence_transformers import SentenceTransformer

# Ollama client
ollama_client = Client(host="http://localhost:11434")

# Embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB
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

        context_parts = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            source = meta.get("source", "Unknown")
            chunk = meta.get("chunk", "Unknown")
            context_parts.append(
                f"""
[DOC{i}]
Source: {source}
Chunk: {chunk}

{doc}
"""
            )

        context = "\n\n".join(context_parts)
        response = ollama_client.chat(
            model="mymodel",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a retrieval-augmented assistant.

STRICT RULES:

1. Use ONLY information found in the provided context.
2. Every factual statement MUST include at least one citation in the form [n].
3. Never make a claim without a citation.
4. If a sentence cannot be supported by a retrieved chunk, do not include that sentence.
5. If the answer cannot be fully supported by the provided context, respond exactly:

"I cannot answer from the provided documents."

6. Do not use prior knowledge.
7. Do not infer, assume, summarize, or speculate beyond the retrieved context.
8. Every paragraph must contain at least one citation.
9. The final answer must contain citations. Answers without citations are invalid.
""",
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

        citation_map = {
            f"DOC{i}": {"source": meta.get("source"), "chunk": meta.get("chunk")}
            for i, meta in enumerate(metadatas, start=1)
        }
        return {"answer": response["message"]["content"], "citations": citation_map}
    except Exception as exc:
        return {"answer": f"Error: {exc}", "citations": {}}


# if __name__ == "__main__":

#     question = (
#         "What is machine learning?"
#     )

#     response = ask_rag(question)

#     print("\nANSWER:")
#     print(response["answer"])

#     print("\nCITATIONS:")
#     for doc_id, citation in response[
#         "citations"
#     ].items():

#         print(
#             f"{doc_id} -> "
#             f"{citation['source']} "
#             f"(chunk {citation['chunk']})"
#         )