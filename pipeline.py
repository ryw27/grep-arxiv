import urllib.request
import feedparser
import pymupdf
import requests
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import asyncio
import aiohttp
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

async def download_all_pdfs():
    download_queue = asyncio.Queue()

    async def download_pdf(session: aiohttp.ClientSession, meta: PaperMetaData):
        try:
            async with session.get(url=meta.pdf_url) as response:
                resp = await response.read()
                await download_queue.put((meta, resp))
                print(f"Successfully fetched pdf {meta.pdf_url}")
        except Exception as e:
            print(f"Unable to get url {meta.pdf_url} with error {e}")


    async with aiohttp.ClientSession() as session:
        tasks = [
            download_pdf(session, paper)
            for cat in CATEGORIES
            for paper in papers[cat]
        ]

        await asyncio.gather(*tasks)
        # Close the channel
        await download_queue.put(None)

    return download_queue

async def pipeline() -> None:
    print("Fetching metadata")
    fetchMetaData()
    print("Done fetching metadata")

    # Don't wait for pdfs to finish downloading
    print("Fetching pdfs")
    pdfs_to_download = await download_all_pdfs()
    print("Done fetching pdfs")


# Data pipeline
## Fetch metadata
## Download pdfs and obtain text
### Asyncronous, async fetch since it's network i/o
## Chunk pdfs (CPU heavy -> threaded pool)
### Send through queue as well
## Embed (use gpu?)

## Figure out what kind of data shape I need for fast search

# download (async) -> download_queue -> chunk with threaded pool, pulling from download_queeu like a channel -> queue -> embed (with GPU?) -> store into DB



if __name__ == "__main__":
    asyncio.run(pipeline())