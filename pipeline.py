import urllib.request
import feedparser
import pymupdf
import requests
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict
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

@dataclass
class PaperMetaData:
    arxiv_id: str
    title: str
    authors: List[str]
    categories: List[str]
    summary: str
    published: str  # arXiv 'published' field is an ISO 8601 date string, not int
    pdf_url: str

@dataclass
class Chunk:
    arxiv_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str
    token_count: int

# Placeholder for postgres
papers: Dict[str, List["PaperMetaData"]] = {cat: [] for cat in CATEGORIES}
chunks: Dict[str, List["Chunk"]] = {}
vector_db: Dict[str, List[float]] = {}


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

def fetchMetaData() -> None:
    for cat in CATEGORIES:
        link = (
            BASE_URL
            + 'search_query=cat:%s&start=%i&max_results=%i&sortBy=submittedDate&sortOrder=descending'
            % (cat, 0, MAX_RESULTS)
        )
        response = urllib.request.urlopen(link).read()
        feed = feedparser.parse(response)
        for paper in feed.entries:
            links = getattr(paper, "links", [])
            urls = [link for link in links if getattr(link, "title", "") == "pdf"]
            # Ensure that a pdf link exists here
            if not urls:
                continue

            link = urls[0]
            href = link.get("href") 
            if not href:
                continue
            
            pdf_url = str(href)

            authors = [a.name for a in getattr(paper, "authors", [])]

            paper_categories = [str(tag["term"]) for tag in paper.tags] 
            
            metadata = PaperMetaData(
                arxiv_id=str(getattr(paper, "id", "")),
                title=str(getattr(paper, "title", "")),
                authors=authors,
                categories=paper_categories,
                published=str(getattr(paper, "published", "")),
                summary=str(getattr(paper, "summary", "")),
                pdf_url=pdf_url,
            )

            papers[cat].append(metadata)


def pipeline() -> None:
    print("Fetching metadata")
    fetchMetaData()
    print("Done fetching metadata")


if __name__ == "__main__":
    pipeline()