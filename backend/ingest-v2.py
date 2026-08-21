# import chromadb
# from ollama import Client
# from sentence_transformers import SentenceTransformer

# # Ollama
# ollama_client = Client(host="http://localhost:11434")

# # Embedding model
# embedding_model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )

# # ChromaDB
# chroma = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma.get_or_create_collection("documents")


# def ask_rag(question: str):

#     query_embedding = embedding_model.encode(
#         question
#     ).tolist()

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=5
#     )

#     docs = results["documents"][0]
#     metas = results["metadatas"][0]

#     context = "\n\n".join(docs)

#     response = ollama_client.chat(
#         model="mymodel",
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "Answer only using the supplied context. "
#                     "If the answer is not in the context, say so."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": f"""
#     Context:
#     {context}

#     Question:
#     {question}
#     """
#                 }
#             ]
#         )
   
#     return {
#         "answer": response["message"]["content"],
#         "sources": [
#             m.get("source", "")
#             for m in metas
#         ]
#     }

# new version
import chromadb
from ollama import Client
from sentence_transformers import SentenceTransformer

ollama_client = Client(
    host="http://localhost:11434"
)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chroma = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma.get_or_create_collection(
    name="documents"
)


def ask_rag(question: str):
    try:
        query_embedding = embedding_model.encode(
            question
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        if not results["documents"] or not results["documents"] return {
                "answer": "No relevant documents found.",
                "sources": []
            }

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        context = "\n\n".join(docs)

        response = ollama_client.chat(
            model="mymodel",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only using the supplied context. "
                        "If the answer is not in the context, say so."
                    )
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

        return {
            "answer": response["message"]["content"],
            "sources": [
                m.get("source", "")
                for m in metas
            ]
        }

    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "sources": []
        }