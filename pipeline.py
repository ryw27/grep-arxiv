import urllib
import feedparser
import pymupdf
import requests
import re
import uuid
from sentence_transformers import SentenceTransformer
# import psycopg2

# Config Variables
BASE_URL = 'http://export.arxiv.org/api/query?'
MODEL = SentenceTransformer("all-MiniLM-L6-V2")
CATEGORIES = ["cs.AI", "cs.LG", "cs.CV"]
MAX_RESULTS = 10

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Data models
class PaperMetaData:
    arxiv_id: str
    title: str
    author: str
    authors: list[str]
    categories: list[str]
    summary: str
    published: int
    pdf_url: str

class Chunk:
    arxiv_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str
    token_count: int

# Placeholder for postgres
papers = {}
chunks = {}
vector_db = {}


# def chunkPaper(full_text, chunk_size=400, overlap=50):
#     # Tokenize by whitespace
#     tokens = re.split(r"\s+", full_text)
#     chunks = []
    
#     i = 0
#     while i < len(tokens):
#         chunk = tokens[i : i + chunk_size]
#         if not chunk:
#             break
#         chunks.append(chunk)
#         i += (chunk_size - overlap)
#     return chunks

def fetchMetaData():
    for cat in CATEGORIES:
        link = BASE_URL + 'search_query=cat:%s&start=%i&max_results=%i&sortBy=submittedDate&sortOrder=descending' % (cat, 0, MAX_RESULTS)
        response = urllib.request.urlopen(link).read()
        feed = feedparser.parse(response)
        for paper in feed.entries:
            urls = [link for link in paper.links if link.title == "pdf"]
            # Ensure that a pdf link exists here
            if not urls:
                continue
            pdf_url = urls[0].href
            
            metadata = PaperMetaData(
                arxiv_id = paper.id,
                title=e.title,
                authors=e.authors,
                categories=paper.categories,
                published=paper.published,
                summary=paper.summary,
                pdf_url=pdf_url,
            )

            papers[cat].append(metadata)


def pipeline():
    print("Fetching metadata")
    fetchMetaData()


if __name__ == "__main__":
    pipeline()