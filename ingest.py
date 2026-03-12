from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_manual(pdf_path: str, collection_name: str):
    print(f"📄 Loading {collection_name} manual...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    print("✂️  Chunking document...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔢 Creating embeddings and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name=collection_name
    )
    print(f"✅ Done! Manual stored as {collection_name}")
    return vectorstore

if __name__ == "__main__":
    ingest_manual("data/manual.pdf", "toyota_aygo_2011")