import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import chromadb

load_dotenv()

st.set_page_config(page_title="AutoIQ", page_icon="🚗")
st.title("🚗 AutoIQ")
st.subheader("Your AI Car Manual Assistant")

def get_existing_manuals():
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = client.list_collections()
    return [col.name for col in collections]

def collection_exists(collection_name):
    existing = get_existing_manuals()
    return collection_name in existing

def ingest_manual(pdf_path, collection_name):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with st.spinner(f"📄 Reading and indexing {collection_name}..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db",
            collection_name=collection_name
        )
    st.success(f"✅ {collection_name} indexed and ready!")

def load_chain(collection_name):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=collection_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Sidebar ---
with st.sidebar:
    st.header("📚 My Manuals")

    existing_manuals = get_existing_manuals()

    if existing_manuals:
        selected_manual = st.selectbox(
            "Switch manual:",
            options=existing_manuals,
            format_func=lambda x: x.replace("_", " ").title()
        )
    else:
        selected_manual = None
        st.info("No manuals yet — add one below!")

    st.divider()
    st.subheader("➕ Add a New Manual")

    col1, col2, col3 = st.columns(3)
    with col1:
        make = st.text_input("Make", placeholder="Toyota")
    with col2:
        model = st.text_input("Model", placeholder="Aygo")
    with col3:
        year = st.text_input("Year", placeholder="2011")

    if make and model and year:
        collection_name = f"{make}_{model}_{year}".lower().replace(" ", "_")

        if collection_exists(collection_name):
            st.success(f"✅ {make} {model} {year} already in library!")
            if st.button("Switch to this manual"):
                selected_manual = collection_name
                st.rerun()
        else:
            uploaded_file = st.file_uploader("Upload PDF manual", type="pdf")
            if uploaded_file:
                if st.button("Add to AutoIQ"):
                    temp_path = f"data/{collection_name}.pdf"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    ingest_manual(temp_path, collection_name)
                    st.rerun()

# --- Main Chat ---
if selected_manual:
    st.caption(f"📖 Chatting with: **{selected_manual.replace('_', ' ').title()}**")

    if "messages" not in st.session_state or st.session_state.get("current_manual") != selected_manual:
        st.session_state.messages = []
        st.session_state.current_manual = selected_manual

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about your car..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching manual..."):
                chain = load_chain(selected_manual)
                answer = chain.invoke(question)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Add a manual in the sidebar to get started!")
    