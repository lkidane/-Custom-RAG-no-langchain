import chromadb


# ---------------------------
# ChromaDB Configuration
# ---------------------------
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"


# ---------------------------
# Connect to ChromaDB
# ---------------------------
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


# ---------------------------
# Delete Document
# ---------------------------
def delete_document(document_path):
    """
    Deletes all chunks belonging to a specific document.
    """

    collection = get_collection()

    try:
        # Retrieve all chunks for the document
        results = collection.get(
            where={"source": document_path}
        )

        ids = results.get("ids", [])

        if not ids:
            print(f"No records found for:\n{document_path}")
            return False

        print(f"Found {len(ids)} chunks.")

        # Delete all matching chunks
        collection.delete(ids=ids)

        print("Document deleted successfully.")

        # Verify deletion
        verify = collection.get(
            where={"source": document_path}
        )

        remaining = len(verify.get("ids", []))

        if remaining == 0:
            print("Verification successful. No chunks remain.")
        else:
            print(f"Warning: {remaining} chunks still exist.")

        return True

    except Exception as e:
        print(f"Error deleting document: {e}")
        return False


# ---------------------------
# List Documents (Optional)
# ---------------------------
def list_documents():
    """
    Lists all unique document paths stored in the collection.
    """

    collection = get_collection()

    try:
        results = collection.get(
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        unique_docs = sorted(
            {
                meta["source"]
                for meta in metadatas
                if meta and "source" in meta
            }
        )

        print("\nIndexed Documents:")
        print("-" * 50)

        for doc in unique_docs:
            print(doc)

        print("-" * 50)
        print(f"Total documents: {len(unique_docs)}")

    except Exception as e:
        print(f"Error listing documents: {e}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    pdf_file = r"C:\Users\lkidane\Downloads\DoYouSpeakGenerativeAI.pdf"

    # Show documents before deletion
    list_documents()

    print("\nDeleting document...")
    delete_document(pdf_file)

    # Show documents after deletion
    print("\nUpdated document list:")
    list_documents()