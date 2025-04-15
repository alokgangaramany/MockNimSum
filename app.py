import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pyairtable import Table
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="NimSum Insights AI", layout="wide")
st.title("🤖 NimSum Terminal: Deep Tech Insights Chatbot")

# --- LOAD SECRETS ---
OPENAI_API_KEY = st.secrets.get("openai_api_key")
AIRTABLE_API_KEY = st.secrets.get("AIRTABLE_API_KEY")
BASE_ID = "appXEj7umXOt9b2XP"  # Replace with your real Base ID
TABLE_NAME = "Report"           # Replace with your table name

# --- FETCH & CACHE AIRTABLE RECORDS ---
@st.cache_data(show_spinner=False)
def fetch_airtable_records(api_key, base_id, table_name):
    table = Table(api_key, base_id, table_name)
    records = table.all()
    st.write(f"✅ Retrieved {len(records)} rows from Airtable")
    return [r["fields"] for r in records]

# --- BUILD VECTORSTORE ---
@st.cache_resource(show_spinner=True)
def build_vectorstore(record_fields, api_key):
    docs = []
    for fields in record_fields:
        content = "\n".join([f"{k}: {v}" for k, v in fields.items() if v])
        docs.append(Document(page_content=content))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)

# --- MAIN EXECUTION ---
if not OPENAI_API_KEY or not AIRTABLE_API_KEY:
    st.error("🔐 Missing API keys. Please add them to Streamlit secrets.")
else:
    with st.spinner("📡 Fetching and embedding Airtable data..."):
        record_fields = fetch_airtable_records(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
        st.success(f"📄 Loaded {len(record_fields)} Airtable records.")
        vectorstore = build_vectorstore(record_fields, OPENAI_API_KEY)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = st.text_input("💬 Ask your question:", placeholder="e.g., What are the key trends in AI diagnostics?")
    if query:
        with st.spinner("🤖 Thinking..."):
            result = qa_chain.run(query)
            st.markdown("### 🧠 AI Insight")
            st.write(result)
