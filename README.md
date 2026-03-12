AutoIQ — Car Manual Q&A App

AutoIQ lets you chat with your car manual. Upload a PDF and ask questions in plain English — it finds the right section and answers instantly.

## What it does
- Reads a car manual PDF and chunks it into searchable sections
- Converts chunks into vector embeddings and stores them in ChromaDB
- Uses GPT-3.5 to answer questions based only on the manual content
- Presents everything in a clean Streamlit chat interface

## How it works
1. The PDF is split into 1000 character chunks with a 200 character overlap so context isn't lost between sections
2. Each chunk is converted to a vector embedding using OpenAI
3. When you ask a question, the 6 most relevant chunks are retrieved from ChromaDB
4. Those chunks are sent to GPT-3.5 along with your question
5. GPT answers based only on the manual — no hallucination

## How to run it

### 1. Clone the repo
git clone https://github.com/cmorris2001/AutoIQ.git

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your OpenAI API key
Create a .env file and add:
OPENAI_API_KEY=your_key_here

### 4. Add your PDF
Drop your car manual PDF into the data/ folder and name it manual.pdf

### 5. Ingest the document
python ingest.py

### 6. Run the app
streamlit run app.py

## Known limitations
- Some maintenance intervals are spread across multiple sections and may not always be retrieved accurately
- Only works with text-based PDFs, not scanned images

## Built with
- LangChain
- ChromaDB
- OpenAI GPT-3.5
- Streamlit
- PyPDF
```

Then run:
```
git add .
git commit -m "Add README"
git push
