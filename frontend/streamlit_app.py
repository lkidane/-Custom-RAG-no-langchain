import requests
import streamlit as st

st.title("RAG Chatbot (No LangChain)")
question = st.text_input("Ask a question")

if st.button("Submit") and question:
    r = requests.post("http://localhost:8000/ask", json={"question":question})
    data = r.json()

    st.subheader("Answer")
    st.write(data["answer"])

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

