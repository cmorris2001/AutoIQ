## AutoIQ — AI Car Manual Assistant

AutoIQ is a RAG (Retrieval Augmented Generation) application that lets you chat with any car manual. 
Upload a PDF, ask questions in plain English, and get accurate answers pulled directly from the manual.

##  Features
- **Multi-manual library** — upload multiple manuals and switch between them
- **Duplicate detection** — checks ChromaDB before ingesting, never stores the same manual twice
- **Accurate answers** — GPT-3.5 answers only from manual content, no hallucination
- **Clean chat UI** — Streamlit chat interface with conversation history per manual
- **Persistent storage** — manuals survive between sessions, ingest once and reuse forever

## How it works
1. User enters car make, model and year
2. App checks ChromaDB — if already indexed, loads instantly with no re-ingestion
3. If new, user uploads the PDF which is split into 1000 character chunks with 200 character overlap
4. Each chunk is converted to a vector embedding via OpenAI and stored in ChromaDB
5. When a question is asked, the 6 most semantically relevant chunks are retrieved
6. Those chunks are passed to GPT-3.5 as context along with the question
7. GPT answers based only on the retrieved content

## How to run it

### 1. Clone the repo
git clone https://github.com/cmorris2001/AutoIQ.git

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your OpenAI API key
Create a .env file:
OPENAI_API_KEY=your_key_here

### 4. Run the app
streamlit run app.py

### 5. Add your first manual
Enter the make, model and year in the sidebar, upload the PDF and start chatting.

## Known limitations
- Some maintenance intervals span multiple sections and may not always be retrieved in a single query
- Only works with text-based PDFs, not scanned images
- Manual switcher requires page refresh to take effect

## Future improvements
- Delete manual from library
- Support scanned PDFs via OCR
- Re-ranking retriever for improved accuracy on complex queries
- Deploy to cloud so anyone can use it without a local setup

## Built with
- LangChain
- ChromaDB
- OpenAI GPT-3.5
- Streamlit
- PyPDF

