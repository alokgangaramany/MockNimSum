import streamlit as st
from pyairtable import Table
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
import io
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="NimSum Insights AI", layout="wide")
st.title("🤖 NimSum Terminal: Unified Market Insights Chatbot")

# --- LOAD SECRETS ---
OPENAI_API_KEY = st.secrets.get("openai_api_key")
AIRTABLE_API_KEY = st.secrets.get("AIRTABLE_API_KEY")
BASE_ID = "appXEj7umXOt9b2XP"
TABLE_NAME = "Report"

# --- HELPER: Convert Airtable record (structured + PDF) into LangChain documents ---
def record_to_documents(record):
    fields = record["fields"]
    metadata_text = "\n".join([f"{k}: {v}" for k, v in fields.items() if k != "PDF Upload" and v])

    docs = [Document(page_content=metadata_text, metadata={"source": fields.get("Title", "Unnamed")})]

    if "PDF Upload" in fields:
        try:
            pdf_url = fields["PDF Upload"][0]["url"]
            response = requests.get(pdf_url)
            file_stream = io.BytesIO(response.content)
            reader = PdfReader(file_stream)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                docs.append(Document(page_content=text, metadata={
                    "source": fields.get("Title", "Unnamed"),
                    "page": i + 1
                }))
        except Exception as e:
            st.warning(f"⚠️ Failed to load PDF for {fields.get('Title', 'Unknown Report')} — {str(e)}")

    return docs

# --- FETCH + EMBED DATA FROM AIRTABLE ---
@st.cache_resource(show_spinner=True)
def build_vectorstore(api_key, base_id, table_name):
    table = Table(api_key, base_id, table_name)
    records = table.all()

    all_docs = []
    for r in records:
        all_docs.extend(record_to_documents(r))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)

# --- RUN APP ---
if not OPENAI_API_KEY or not AIRTABLE_API_KEY:
    st.error("🔐 Missing API keys. Please check your Streamlit secrets.")
else:
    with st.spinner("🔄 Loading market insights from Airtable and PDFs..."):
        vectorstore = build_vectorstore(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
#        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#        vectorstore = FAISS.load_local("vectorstore_index", embeddings)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = st.text_input("💬 Ask a question about the market reports:", placeholder="e.g., What are the growth trends in AI diagnostics?")
    if query:
        with st.spinner("🤖 Thinking..."):
            result = qa_chain.run(query)
            st.markdown("### 🧠 Insight")
            st.write(result)


# import streamlit as st
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from pyairtable import Table
# import pandas as pd
# import os

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="NimSum Insights AI", layout="wide")
# st.title("🤖 NimSum Terminal: Deep Tech Insights Chatbot")

# # --- LOAD SECRETS ---
# OPENAI_API_KEY = st.secrets.get("openai_api_key")
# AIRTABLE_API_KEY = st.secrets.get("AIRTABLE_API_KEY")
# BASE_ID = "appXEj7umXOt9b2XP"  # Replace with your real Base ID
# TABLE_NAME = "Report"           # Replace with your table name

# # --- FETCH & CACHE AIRTABLE RECORDS ---
# @st.cache_data(show_spinner=False)
# def fetch_airtable_records(api_key, base_id, table_name):
#     table = Table(api_key, base_id, table_name)
#     records = table.all()
#     st.write(f"✅ Retrieved {len(records)} rows from Airtable")
#     return [r["fields"] for r in records]

# # --- BUILD VECTORSTORE ---
# @st.cache_resource(show_spinner=True)
# def build_vectorstore(record_fields, api_key):
#     docs = []
#     for fields in record_fields:
#         content = "\n".join([f"{k}: {v}" for k, v in fields.items() if v])
#         docs.append(Document(page_content=content))

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = splitter.split_documents(docs)
#     embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#     return FAISS.from_documents(chunks, embeddings)

# # --- MAIN EXECUTION ---
# if not OPENAI_API_KEY or not AIRTABLE_API_KEY:
#     st.error("🔐 Missing API keys. Please add them to Streamlit secrets.")
# else:
#     with st.spinner("📡 Fetching and embedding Airtable data..."):
#         record_fields = fetch_airtable_records(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
#         st.success(f"📄 Loaded {len(record_fields)} Airtable records.")
#         vectorstore = build_vectorstore(record_fields, OPENAI_API_KEY)

#     llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

#     query = st.text_input("💬 Ask your question:", placeholder="e.g., What are the key trends in AI diagnostics?")
#     if query:
#         with st.spinner("🤖 Thinking..."):
#             result = qa_chain.run(query)
#             st.markdown("### 🧠 AI Insight")
#             st.write(result)
