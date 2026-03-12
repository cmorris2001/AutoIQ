from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
 
# Load your API key from .env
load_dotenv()
 
def ingest_manual(pdf_path: str = "data/manual.pdf"):
    print("📄 Loading manual...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
 
    print("✂️  Chunking document...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Each chunk is ~1000 characters
        chunk_overlap=200,    # Chunks overlap so context isn't lost at edges
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
 
    print("🔢 Creating embeddings and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"  # Saves locally so you don't re-ingest every time
    )
    print("✅ Done! Manual stored in chroma_db/")
    return vectorstore
 
if __name__ == "__main__":
    ingest_manual()
    