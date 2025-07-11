### Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion
        A[Local HTML, PDFs, RSS] --> B["Extract Text\n(BeautifulSoup, pdfplumber,\npdf2image, pytesseract, feedparser)"]
        B --> C[Split to Chunks]
        C --> D["Generate Embeddings\n(OpenAI)"]
        D --> E[Pinecone Vector DB]
    end

    subgraph API
        F[User Question] --> G["Embed Question\n(OpenAI)"]
        G --> H[Query Pinecone]
        H --> I[Retrieve Context]
        I --> J["Generate Answer\n(GPT‑4)"]
        J --> K[Return JSON Response]
    end

    F -. request .-> M[/FastAPI /ask/]
    M -. response .-> K
