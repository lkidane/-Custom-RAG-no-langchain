# import chromadb
# from openai import OpenAI
# import os

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# chroma = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma.get_or_create_collection("documents")

# def ask_rag(question:str):
#     emb = client.embeddings.create(model="text-embedding-3-large", input=question)
#     query_embedding = emb.data[0].embedding

#     results = collection.query(query_embeddings=[query_embedding], n_results=5)
#     docs = results['documents'][0]
#     metas = results['metadatas'][0]

#     context = '\n\n'.join(docs)

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role":"system","content":"Answer using the provided context only."},
#             {"role":"user","content":f"Context:\n{context}\n\nQuestion:{question}"}
#         ]
#     )

#     return {
#         "answer": response.choices[0].message.content,
#         "sources": [m.get('source','') for m in metas]
#     }
