# app.py

# ---------------------
# Step 1: Install dependencies
# ---------------------
# pip install openai pinecone-client fastapi uvicorn langchain pydantic beautifulsoup4 pdfplumber

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import openai
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# ---------------------
# Step 2: Configure OpenAI and Pinecone
# ---------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a new index with dimension from env if it doesn't exist
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_DIM = int(os.getenv("PINECONE_EMBED_DIM", "1536"))
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )
index = pc.Index(INDEX_NAME)

# ---------------------
# Step 3: FastAPI app setup
# ---------------------
app = FastAPI()

# Enable CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://rakeshvardan.com", "https://www.rakeshvardan.com/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


class QuestionRequest(BaseModel):
    question: str


# ---------------------
# Step 4: Helper - Embed question
# ---------------------
def get_embedding(text: str):
    response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


# ---------------------
# Step 5: Helper - Query Pinecone
# ---------------------
def query_vector_db(embedding):
    query_result = index.query(vector=embedding, top_k=5, include_metadata=True)
    contexts = [match["metadata"]["text"] for match in query_result["matches"]]
    return "\n---\n".join(contexts)


# ---------------------
# Step 6: Helper - Generate answer with GPT-4
# ---------------------
def generate_answer(context: str, question: str):
    prompt = f"""
You are Rakesh's professional assistant. Answer the user's question using the context below.
If the context doesn't contain the answer, say "I don't have information about that."

Context:
{context}

Question: {question}
Answer:
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You answer based on Rakesh's professional background.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


# ---------------------
# Step 7: API Route - Ask Question
# ---------------------
@app.post("/ask")
@limiter.limit("5/minute")
def ask_question(request: Request, req: QuestionRequest):
    try:
        embedding = get_embedding(req.question)
        context = query_vector_db(embedding)
        answer = generate_answer(context, req.question)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------
# Step 8: Run server
# ---------------------
# Run with: uvicorn app:app --reload


def split_text(text, max_length=500):
    words = text.split()
    return [
        " ".join(words[i : i + max_length]) for i in range(0, len(words), max_length)
    ]


if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
