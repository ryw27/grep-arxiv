import urllib
import feedparser
import pymupdf
import requests
import re
import uuid
from sentence_transformers import SentenceTransformer
# import psycopg2

# Substitute for sql for now
# Key by arxiv ID
papers = {}
chunks = {}
vector_db = {}


base_url = 'http://export.arxiv.org/api/query?'

def getPapers(topic, max_results):
    link = base_url + 'search_query=cat:%s&start=%i&max_results=%i' % (topic, 0, max_results)
    response = urllib.request.urlopen(link).read()

    # Extract metadata and download pdf

    feed = feedparser.parse(response)

    # Loop through each paper
    for paper in feed.entries:
        # TODO: replace with postgres
        papers[paper.id] = {
            "author": paper.author,
            "authors": paper.authors,
            "title": paper.title,
            "published": paper.published,
            "summary": paper.summary,
            "status": "pending"
        }
        assert(paper.links[1].title == "pdf")

        pdf_link = paper.links[1].href
        response = requests.get(pdf_link, stream=True)
        response.raise_for_status()

        pdf = pymupdf.open(stream=response.content, filetype="pdf")

        full_text = []
        for page in pdf:
            full_text.append(page.get_text())
        
        paper_chunks = chunkPaper(full_text)

        for idx, chunk in enumerate(paper_chunks):
            chunk_id = uuid.uuid4()

            model = SentenceTransformer("all-MiniLM-L6-V2")
            embedded_chunk = model.encode(chunk)

            chunks[chunk_id] = {
                "arxiv_id": paper.id, # Foreign Key, one to many
                "chunk_index": idx,
                "chunk_text": chunk,
                "token_count": len(chunk),
                "embedded": embedded_chunk
            }
    return true

def chunkPaper(full_text, chunk_size=400, overlap=50):
    # Tokenize by whitespace
    tokens = re.split(r"\s+", full_text)
    chunks = []
    
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def pipeline():
    # 1. Fetch 
    # Get 3 topics for now

    # cs.AI
    csAI = getPapers("cs.AI", 1)

    # cs.LG
    csLG = getPapers("cs.LG", 1)

    # cs.CV
    csCV = getPapers("cs.CV", 1)



if __name__ == "__main__":
    # conn = psycopg2.connect()
    # cursor = conn.cursor()
    # cursor.execute("CREATE TABLE IF NOT EXISTS papers")
    pipeline()