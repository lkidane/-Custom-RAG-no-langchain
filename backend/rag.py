import chromadb
from ollama import Client
from sentence_transformers import SentenceTransformer

# Ollama client
ollama_client = Client(
    host="http://localhost:11434"
)

# Embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ChromaDB
chroma = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma.get_or_create_collection(
    name="documents"
)


def ask_rag(question: str):

    try:

        # Create query embedding
        query_embedding = embedding_model.encode(
            question,
            convert_to_numpy=True
        ).tolist()

        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # if (
        #     not results.get("documents")
        #     or len(results[- in range(len(chunks)) ingested {           return {
        #         "answer": "No relevant documents found.",
        #         "citations": {}
        #     }

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Build citation-aware context
        context_parts = []

        for i, (doc, meta) in enumerate(
            zip(docs, metas),
            start=1
        ):

            source = meta.get(
                "source",
                "Unknown"
            )

            chunk = meta.get(
                "chunk",
                "Unknown"
            )

            context_parts.append(
                f"""
[DOC{i}]
Source: {source}
Chunk: {chunk}

{doc}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        response = ollama_client.chat(
            model="mymodel",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful assistant.

Use ONLY the supplied context.

Whenever you use information from a retrieved
document, cite it using:

[DOC1]
[DOC2]
etc.

Example:

Machine learning learns patterns from data [DOC1].

If the answer cannot be found in the context,
reply exactly:

I don't know
"""
                },
                {
                    "role": "user",
                    "content": f"""
Context:

{context}

Question:

{question}
"""
                }
            ]
        )

        citation_map = {
            f"DOC{i}": {
                "source": meta.get("source"),
                "chunk": meta.get("chunk")
            }
            for i, meta in enumerate(
                metas,
                start=1
            )
        }

        return {
            "answer": response["message"]["content"],
            "citations": citation_map
        }

    except Exception as e:

        return {
            "answer": f"Error: {str(e)}",
            "citations": {}
        }


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