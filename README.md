# Personal RAG (Retrieval-Augmented Generation)

A custom AI-powered assistant that answers questions about your professional profile using your own documents and web content.

## Features
- Ingests data from portfolio HTML, resume PDF, LinkedIn summary, skills, certifications, and blog posts (RSS).
- Extracts text using BeautifulSoup, pdfplumber, pytesseract, and pdf2image.
- Generates embeddings with OpenAI and stores them in Pinecone for semantic search.
- FastAPI backend exposes a `/ask` endpoint for question answering.
- Uses GPT-4 to generate answers based on retrieved context.
- Rate limiting and CORS for secure API access.

## Tech Stack
- Python (core language)
- FastAPI (REST API backend)
- OpenAI API (embeddings & GPT-4 for answer generation)
- Pinecone (vector database for semantic search)
- BeautifulSoup, pdfplumber, pytesseract, pdf2image (data extraction from HTML/PDF/images)
- LangChain (RAG orchestration)
- Uvicorn (ASGI server)
- slowapi (rate limiting)
- dotenv (environment management)

## Setup
1. Clone the repository and install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Add your API keys and config to a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=your_index_name
   PINECONE_EMBED_DIM=1536
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   ```
3. Ingest your data:
   ```sh
   python ingest_data.py
   ```
4. Start the API server:
   ```sh
   uvicorn app:app --reload
   ```

## Usage
Send a POST request to `/ask` with your question:
```json
{
  "question": "What are my key skills?"
}
```
