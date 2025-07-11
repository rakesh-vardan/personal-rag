%%{init: {'theme': 'base', 'themeVariables': {
    'primaryColor': '#f0f8ff',
    'primaryBorderColor': '#0288d1',
    'primaryTextColor': '#003c57'
}}}%%
classDef ingestion fill:#e6f7ff,stroke:#0288d1,color:#001428;
classDef api fill:#fce4ec,stroke:#d81b60,color:#330014;

flowchart TD
    subgraph Ingestion
        direction TB
        A([fa:fa-file-code HTML/PDFs/RSS]):::ingestion
        B([fa:fa-scissors Extract Text<br>(BeautifulSoup, pdfplumber,<br>pdf2image, pytesseract, feedparser)]):::ingestion
        C([fa:fa-align-left Split to Chunks]):::ingestion
        D([fa:fa-brain Generate Embeddings<br>(OpenAI)]):::ingestion
        E([fa:fa-database Pinecone Vector DB]):::ingestion
        A --> B --> C --> D --> E
    end

    subgraph API
        direction TB
        F([fa:fa-user User Question]):::api
        G([fa:fa-brain Embed Question<br>(OpenAI)]):::api
        H([fa:fa-search Query Pinecone]):::api
        I([fa:fa-book Retrieve Context]):::api
        J([fa:fa-magic Generate Answer<br>(GPT‑4)]):::api
        K([fa:fa-reply Return JSON Response]):::api
        F --> G --> H --> I --> J --> K
    end

    F -. request .-> M[/fa:fa-paper-plane FastAPI /ask/]:::api
    M -. response .-> K
