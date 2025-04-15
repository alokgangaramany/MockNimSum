from pyairtable import Table
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests, io
import os

# Load API keys from environment (set these via CLI or .env/.toml)
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_ID = "appXEj7umXOt9b2XP"
TABLE_NAME = "Report"

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
            print(f"⚠️ PDF failed for {fields.get('Title', 'Unknown')} → {str(e)}")

    return docs

def main():
    table = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
    records = table.all()

    all_docs = []
    for r in records:
        all_docs.extend(record_to_documents(r))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("vectorstore_index")
    print("✅ Vectorstore saved to ./vectorstore_index")

if __name__ == "__main__":
    main()
