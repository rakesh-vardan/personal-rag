### Architecture Diagram

```mermaid

flowchart TD
    %% Ingestion Pipeline
    subgraph Ingestion
        A["Data Sources: 
        Portfolio HTML, Resume PDF, LinkedIn, RSS Blogs"] --> B["Extract Text
        (BeautifulSoup, pdfplumber, pdf2image, pytesseract, feedparser)"]
        B --> C["Split into Chunks (LangChain)"]
        C --> D["Generate Embeddings
        (OpenAI Embeddings API)"]
        D --> E["Store Vectors
        (Pinecone Vector DB)"]
    end

    %% API & Query Flow
    subgraph Query
        UA["User Question
        (Browser)"] --> F["Send POST Request (FastAPI Endpoint - '/ask')"]
        F --> G["Embed Question
        (OpenAI Embeddings API)"]
        G --> H["Semantic Search
        (Query Pinecone Vector DB)"]
        H --> I["Retrieve Relevant Context
        (Top-K Chunks)"]
        I --> J["Generate Answer
        (GPT‑4 Completion API)"]
        J --> K["Format & Return Response
        (FastAPI)"]
        K --> UB["Display Answer
        (Browser)"]
    end

    %% External Services
    G -. "Embed API Call" .-> OA["OpenAI API"]
    H -. "Vector Search" .-> PC["Pinecone DB"]
    J -. "Completion API Call" .-> OAGPT["OpenAI GPT-4 API"]
