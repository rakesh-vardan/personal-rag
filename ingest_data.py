from pinecone import Pinecone, ServerlessSpec
import openai
from bs4 import BeautifulSoup
import pdfplumber
import os
from dotenv import load_dotenv
import feedparser
import requests
import pytesseract
from pdf2image import convert_from_path

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_DIM = int(os.getenv("PINECONE_EMBED_DIM", "1536"))
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
index = pc.Index(INDEX_NAME)


def get_embedding(text: str):
    response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def split_text(text, max_length=500):
    words = text.split()
    return [
        " ".join(words[i : i + max_length]) for i in range(0, len(words), max_length)
    ]


def fetch_blog_posts(rss_url):
    feed = feedparser.parse(rss_url)
    posts = []
    blog_titles = []

    for entry in feed.entries:
        title = entry.title
        blog_titles.append(title)
        summary = entry.get("summary", "")

        # Try to extract full content from <content:encoded>
        content = ""
        if hasattr(entry, "content") and entry.content:
            html_content = entry.content[0].value
            # Convert HTML to plain text
            soup = BeautifulSoup(html_content, "html.parser")
            content = soup.get_text(separator=" ", strip=True)
        else:
            # Fallback to summary if content is not present
            content = summary

        # Add publication date and link (optional, but improves answers)
        pub_date = entry.get("published", "")
        link = entry.get("link", "")

        full_text = (
            f"Blog Title: {title}\nPublished: {pub_date}\nLink: {link}\n\n{content}"
        )
        posts.append(("blog", full_text))

    summary_text = f"Rakesh Vardan has written {len(blog_titles)} blogs:\n" + "\n".join(
        f"- {title}" for title in blog_titles
    )
    posts.append(("blog_summary", summary_text))

    return posts


def extract_text_from_pdf_image(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text


def ingest_data():
    data_sources = []

    # Load and parse a local HTML file (your website export)
    with open("mydata/portfolio.html", "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        data_sources.append(("portfolio", text))

    # Extract text from a PDF resume
    with pdfplumber.open("mydata/resume.pdf") as pdf:
        resume_text = " ".join(page.extract_text() or "" for page in pdf.pages)
        data_sources.append(("resume", resume_text))

    # Portfolio PDF
    portfolio_pdf_text = extract_text_from_pdf_image("mydata/portfolio.pdf")
    data_sources.append(("portfolio_pdf", portfolio_pdf_text))

    # GitHub profile PDF
    github_profile_text = extract_text_from_pdf_image("mydata/github_profile.pdf")
    data_sources.append(("github_profile", github_profile_text))

    # LinkedIn Profile Screenshot PDF
    linkedin_profile_text = extract_text_from_pdf_image(
        "mydata/rakesh_linkedin_summary.pdf"
    )
    data_sources.append(("linkedin_profile", linkedin_profile_text))

    # Skills Screenshot PDF
    skills_text = extract_text_from_pdf_image("mydata/rakesh_skills.pdf")
    data_sources.append(("skills", skills_text))

    # Certificates Screenshot PDF
    certificates_text = extract_text_from_pdf_image("mydata/rakesh_certifications.pdf")
    data_sources.append(("certificates", certificates_text))

    # Fetch blog posts from RSS feed
    blog_posts = fetch_blog_posts("https://blog.rakeshvardan.com/rss.xml")
    data_sources.extend(blog_posts)

    # Split and upload chunks
    for doc_index, (source, content) in enumerate(data_sources):
        chunks = split_text(content)
        for chunk_index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            # Unique ID = source + doc index + chunk index
            record_id = f"{source}-{doc_index}-{chunk_index}"
            metadata = {"source": source, "chunk": chunk_index, "text": chunk}
            index.upsert([(record_id, embedding, metadata)])

    print("Data ingestion complete.")


if __name__ == "__main__":
    ingest_data()
