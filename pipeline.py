import urllib.request
import feedparser
import fitz 
import re
from dataclasses import dataclass
import uuid
from typing import List, Dict 
from sentence_transformers import SentenceTransformer
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
# import torch
# import psycopg2

# Config Variables
BASE_URL = 'http://export.arxiv.org/api/query?'
MODEL = SentenceTransformer("all-MiniLM-L6-V2")
CATEGORIES = ["cs.AI", "cs.LG", "cs.CV"]
MAX_RESULTS = 3 
MAX_DOWNLOAD_Q = 50
MAX_CHUNK_Q = 200

MAX_THREAD_WORKERS = 8

# Concurrency tuning (derived from CPU)
DOWNLOAD_CONCURRENCY = 10
CHUNK_WORKERS = 4
EMBED_WORKERS = 1
EMBED_BATCH_SIZE = 32

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

async def download_all_pdfs(download_queue: asyncio.Queue[tuple[PaperMetaData, bytes] | None]) -> None:

    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async def download_pdf(session: aiohttp.ClientSession, meta: PaperMetaData):
        try:
            async with sem:
                async with session.get(url=meta.pdf_url) as response:
                    resp = await response.read()
                    await download_queue.put((meta, resp))
                    print(f"Successfully fetched pdf {meta.pdf_url} with category {meta.categories[0]}")
        except Exception as e:
            print(f"Unable to get url {meta.pdf_url} with error: {e}")


    timeout = aiohttp.ClientTimeout(total=180, sock_connect=30, sock_read=150)
    connector = aiohttp.TCPConnector(limit=DOWNLOAD_CONCURRENCY, limit_per_host=DOWNLOAD_CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            download_pdf(session, paper)
            for cat in CATEGORIES
            for paper in papers[cat]
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print("Error in downloading pdfs", e)
        finally:
            # Close the channel
            await download_queue.put(None)

async def chunkPDFs(downloaded_pdfs: asyncio.Queue[tuple[PaperMetaData, bytes] | None], chunk_queue: asyncio.Queue[tuple[PaperMetaData, Chunk] | None]) -> None:
    # Chunker worker function
    def chunker(metadata: PaperMetaData, fetched_pdf_bytes: bytes):
        full_text = []

        with fitz.open(stream=fetched_pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                full_text.append(page.get_text())
        
        tokens = re.split(r"\s+", "\n".join(full_text))
        pdf_chunks = []

        i = 0
        idx = 0
        while i < len(tokens):
            chunk = tokens[i : i + CHUNK_SIZE]
            if not chunk:
                break
            
            pdf_chunks.append(chunk)
            i += (CHUNK_SIZE - CHUNK_OVERLAP)
            idx += 1

        return pdf_chunks

        # return pdf_chunks 
    loop = asyncio.get_running_loop() 
    with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as executor:
        while True:
            item = await downloaded_pdfs.get()
            if item is None:
                # Close chunk queue channel
                await chunk_queue.put(None)
                break

            metadata, pdf = item
            try:
                result = await loop.run_in_executor(executor, chunker, metadata, pdf)
            except Exception as exc:
                print(f"Chunking failed for {metadata.pdf_url}: {exc}")
                continue
            

            print("Successfully chunked ", metadata.pdf_url)

            for idx, chunk in enumerate(result):
                await chunk_queue.put((metadata, Chunk(
                    arxiv_id=metadata.arxiv_id,
                    chunk_id=str(uuid.uuid4()),
                    chunk_index=idx,
                    chunk_text=" ".join(chunk),
                    token_count=len(chunk)
                )))


async def embedChunks(chunk_queue: asyncio.Queue[tuple[PaperMetaData, Chunk] | None]) -> None:
    def embed_batch(texts: List[str]):
        return MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        batch: List[tuple[PaperMetaData, Chunk]] = []
        ch_closed = False
        while True:
            item = await chunk_queue.get()
            if item is None:
                ch_closed = True
            else:
                batch.append(item)

            # Try to fill the batch quickly without blocking
            while len(batch) < EMBED_BATCH_SIZE:
                try:
                    next_item = chunk_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_item is None:
                    ch_closed = True
                    break
                batch.append(next_item)

            if batch:
                try:
                    texts = [c.chunk_text for _, c in batch]
                    embeddings = await loop.run_in_executor(executor, embed_batch, texts)
                    for (_, c), emb in zip(batch, embeddings):
                        vector_db[c.chunk_id] = emb
                except Exception as e:
                    print(f"Embedding batch failed with {len(batch)} items: {e}")
                finally:
                    batch.clear()

            if ch_closed:
                break

async def pipeline() -> None:
    download_queue = asyncio.Queue(maxsize=MAX_DOWNLOAD_Q)
    chunk_queue = asyncio.Queue(maxsize=MAX_CHUNK_Q)

    print("Fetching metadata")
    fetchMetaData()
    print("Done fetching metadata")

    pdfs_to_download = asyncio.create_task(download_all_pdfs(download_queue))
    pdf_chunks = asyncio.create_task(chunkPDFs(download_queue, chunk_queue))
    embeddings_task = asyncio.create_task(embedChunks(chunk_queue))

    await asyncio.gather(pdfs_to_download, pdf_chunks, embeddings_task)


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