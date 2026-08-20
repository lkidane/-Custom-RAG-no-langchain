import requests
import streamlit as st

st.title("RAG Chatbot (No LangChain)")
question = st.text_input("Ask a question")

if st.button("Submit") and question:
    r = requests.post("http://localhost:8000/ask", json={"question":question})
    data = r.json()

    st.subheader("Answer")
    st.write(data["answer"])
    st.subheader("Sources")
    st.subheader("Sources")

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