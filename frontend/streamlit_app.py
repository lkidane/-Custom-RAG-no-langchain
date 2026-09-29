import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.title("RAG Chatbot")
question = st.text_input("Ask a question")

try:
    doc_response = requests.get(f"{API_BASE_URL}/documents", timeout=20)
    documents = doc_response.json().get("documents", [])
except Exception as exc:
    documents = []
    st.sidebar.error(f"Could not load documents: {exc}")

st.sidebar.header("Indexed Documents")
for doc in documents:
    st.sidebar.write(doc)

selected_doc = None
if documents:
    selected_doc = st.sidebar.selectbox("Select document", documents)
    st.sidebar.write(f"Selected: {selected_doc}")

if selected_doc and st.sidebar.button("Delete Document", type="primary"):
    delete_response = requests.delete(f"{API_BASE_URL}/documents/{selected_doc}", timeout=20)
    if delete_response.status_code == 200:
        st.sidebar.success(f"{selected_doc} deleted")
        st.rerun()
    else:
        st.sidebar.error("Delete failed")

if st.button("Submit") and question:
    response = requests.post(f"{API_BASE_URL}/ask", json={"question": question}, timeout=60)
    data = response.json()

    st.subheader("Answer")
    st.write(data.get("answer", "No answer returned."))

    st.subheader("Sources")
    citations = data.get("citations", {})
    for doc_id, info in citations.items():
        st.write(f"{doc_id}: {info.get('source', 'Unknown')} (Chunk {info.get('chunk', 'Unknown')})")

st.write("Upload PDF documents to ChromaDB")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.success(f"Selected: {uploaded_file.name}")

    if st.button("Add Document"):
        with st.spinner("Generating embeddings..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(
                f"{API_BASE_URL}/upload-document",
                files=files,
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                st.success(
                    f"✅ Document uploaded successfully\n\nChunks created: {data['chunks_created']}"
                )
                st.rerun()
            else:
                st.error(response.text)
