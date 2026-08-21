import chromadb
import ollama
from sentence_transformers import SentenceTransformer
import fitz
import uuid

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="documents",
        metadata={"hnsw:space": "cosine"})
print(f"Chroma collection '{"documents"}' initialized")



# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )
def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def generate_doc_embeddings():
    pdf = fitz.open(r"C:\Users\lkidane\Downloads\MachineLearning-Lecture01.pdf",  filetype="pdf")

    text = ""

    for page in pdf:
        text += page.get_text()
    if page is not None:
        print("----------------------file read successful------------------")

    pdf.close()
    chunks = split_text(text)
    # chunks = splitter.split_text(text)

    if not chunks:
        print("No text extracted from PDF")


    embeddings = embedding_model.encode(chunks)
    print("----------------------embeddings created------------------")
    # collection = get_chroma_collection()

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [
        {
            "source": "MachineLearning-Lecture01.pdf",
            "chunk": idx,
        }
        for idx in range(len(chunks))
    ]
    print("----------------------metadata created------------------")
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print("done making he embeddings")
    # return chunks



def ask_rag(question: str):
    # generate embedding
    # generate_doc_embeddings()
    # Generate query embedding
    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context
    context = "\n\n".join(docs)

    # Ask Ollama
    response = ollama.chat(
        model="mymodel",
        messages=[
            {
                "role": "system",
                "content": """
    You are a helpful assistant.

    Answer only using the provided context.
    If the answer is not available in the context,
    respond with 'I don't know'.
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

    return {
        "answer": response["message"]["content"],
        "sources": [m.get("source", "") for m in metas]
    }

# left side of the code snippet is from generate_embeddings.py, which is responsible for generating embeddings from a PDF document and storing them in a ChromaDB collection. It includes functions to split text into chunks, generate embeddings, and add them to the database.
# another aspect of the code snippet is from backend/ingest-v2.py, which is responsible for querying the ChromaDB collection using a question, generating an embedding for the question, retrieving relevant documents, and then using Ollama to generate a response based on the context provided by those documents.
# 1. git
# 2. fastapi
# 3. rag tool calling
# 4. 