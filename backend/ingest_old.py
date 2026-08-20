# import os
# import uuid
# import chromadb
# from openai import OpenAI
# from dotenv import load_dotenv
# from azure.storage.blob import BlobServiceClient

# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# blob_service = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
# container = blob_service.get_container_client(os.getenv("AZURE_CONTAINER_NAME"))

# chroma = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma.get_or_create_collection("documents")

# for blob in container.list_blobs():
#     text = container.get_blob_client(blob.name).download_blob().readall().decode('utf-8')

#     chunks = [text[i:i+1000] for i in range(0, len(text), 800)]

#     for chunk in chunks:
#         emb = client.embeddings.create(model="text-embedding-3-large", input=chunk)
#         collection.add(
#             ids=[str(uuid.uuid4())],
#             documents=[chunk],
#             embeddings=[emb.data[0].embedding],
#             metadatas=[{"source": blob.name}]
#         )

# print('Ingestion complete')
