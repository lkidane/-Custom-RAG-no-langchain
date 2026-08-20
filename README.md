# RAG Without LangChain

Tech Stack:
- Streamlit
- FastAPI
- OpenAI text-embedding-3-large
- GPT-4o
- ChromaDB
- Azure Blob Storage

Install:
pip install -r backend/requirements.txt

Run ingestion:
python backend/ingest.py

Run API:
uvicorn backend.app:app --reload

Run UI:
streamlit run frontend/streamlit_app.py
