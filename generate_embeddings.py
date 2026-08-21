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


def generate_doc_embeddings(documet):
    pdf = fitz.open(documet,  filetype="pdf")

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
            "source": documet,
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





if __name__ == "__main__":

    generate_doc_embeddings(  r"C:\Users\lkidane\Downloads\DoYouSpeakGenerativeAI.pdf")
    print("done making he embeddings")
# finally the embeddings are created and stored in the ChromaDB collection named "documents".