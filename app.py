import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pyairtable import Table
import pandas as pd
import os

# --- CONFIG ---
st.set_page_config(page_title="NimSum Insights Chatbot", layout="wide")
st.title("🤖 NimSum Terminal: Deep Tech Insights AI")

# Config
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")  # set in .env or Streamlit secrets
BASE_ID = "Main"
TABLE_NAME = "Report"

table = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
records = table.all()

# --- SIDEBAR SETUP ---
st.sidebar.header("🔐 API Setup")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# --- PROCESS CSV INTO VECTORSTORE ---
@st.cache_resource(show_spinner=False)
def process_csv(file, api_key):

    df = pd.DataFrame([r["fields"] for r in records])
#    df = pd.read_csv(file)
    docs = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])
        docs.append(Document(page_content=content))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# --- MAIN CHAT UI ---
if openai_api_key and uploaded_file:
    with st.spinner("Indexing CSV and building AI..."):
        vectorstore = process_csv(uploaded_file, openai_api_key)
        llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    st.success("AI is ready! Ask your question below.")
    query = st.text_input("💬 Ask a question about the market report:", placeholder="e.g., What are key investment trends in AI diagnostics?")
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
            st.markdown("### 🧠 Answer")
            st.write(result)
else:
    st.info("Please upload a CSV file and enter your OpenAI API key to get started.")
