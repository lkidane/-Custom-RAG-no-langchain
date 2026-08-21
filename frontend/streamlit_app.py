import requests
import streamlit as st

st.title("RAG Chatbot (No LangChain)")
question = st.text_input("Ask a question")

# Load documents from Chroma
try:
    doc_response = requests.get(
        "http://localhost:8000/documents"
    )

    documents = doc_response.json().get(
        "documents",
        []
    )

    st.sidebar.header("Indexed Documents")

    for doc in documents:
        st.sidebar.write(doc)

except Exception as e:
    st.sidebar.error(
        f"Could not load documents: {e}"
    )

st.sidebar.header("Indexed Documents")

selected_doc = None

# delete documents
try:
    response = requests.get(
        "http://localhost:8000/documents"
    )

    documents = response.json().get(
        "documents",
        []
    )

    if documents:

        selected_doc = st.sidebar.selectbox(
            "Select document",
            documents
        )

        st.sidebar.write(
            f"Selected: {selected_doc}"
        )

except Exception as e:
    st.sidebar.error(
        f"Failed loading documents: {e}"
    )
if selected_doc:

    if st.sidebar.button(
        "Delete Document",
        type="primary"
    ):

        delete_response = requests.delete(
            f"http://localhost:8000/documents/{selected_doc}"
        )

        if delete_response.status_code == 200:
            st.sidebar.success(
                f"{selected_doc} deleted"
            )
            st.rerun()

        else:
            st.sidebar.error(
                "Delete failed"
            )



if st.button("Submit") and question:
    r = requests.post("http://localhost:8000/ask", json={"question":question})
    data = r.json()

    st.subheader("Answer")
    st.write(data["answer"])
    st.subheader("Sources")
    # st.subheader("Sources")

    citations = data.get("citations", {})

    for doc_id, info in citations.items():
        st.write(
            f"{doc_id}: {info['source']} (Chunk {info['chunk']})"
        )
    # st.subheader("Sources")
    # for s in data["sources"]:
    #     st.write(s)
# else: st.error(f"API Error: {data}")
    # if "answer" in data:
    #     st.subheader("Answer")
    #     st.write(data["answer"])

    #     st.subheader("Sources")
    #     for s in data.get("sources", []):
    #         st.write(s)
    # else:
    #     st.error(f"API Error: {data}")

## Citations mapping but only at the end of the context, after the answer. The citations should be numbered and correspond to the sources in the answer. For example, if the answer references two sources, it should look like this:
    # st.subheader("Answer with Citations")
    # answer = data["answer"]

    # citation_map = {}
    # citation_number = 1

    # for doc_id, info in data["citations"].items():
    #     citation_map[doc_id] = citation_number
    #     citation_number += 1

    # answer += " "

    # for doc_id, number in citation_map.items():
    #     answer += f"[{number}] "

    # st.markdown(answer)

    # citations are working ( but based on the llm prompt - we specify it exactly in the prompt to the llm - so it is not a problem of the frontend -
    #  it is a problem of the backend prompt to the llm)

################### Document upload to add to the collection ################




API_URL = "http://localhost:8000/documents/"

st.write("Upload PDF documents to ChromaDB")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file:

    st.success(f"Selected: {uploaded_file.name}")

    if st.button("Add Document"):

        with st.spinner("Generating embeddings..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            response = requests.post(
                API_URL,
                files=files
            )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    f"✅ Document uploaded successfully\n\n"
                    f"Chunks created: {data['chunks_created']}"
                )

            else:
                st.error(response.text)