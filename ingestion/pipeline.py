# import urllib.request
# import feedparser
import fitz 
import re
import uuid
from typing import List, Dict 
from models import PaperMetaData, Chunk
import arxiv

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

# Chunk config
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


# Placeholder for postgres
papers: Dict[str, List["PaperMetaData"]] = {cat: [] for cat in CATEGORIES}
chunks: Dict[str, List["Chunk"]] = {}
vector_db: Dict[str, List[float]] = {}

class PaperFetcher:
    def __init__(self):
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )

        self.download_queue = asyncio.Queue(maxsize=MAX_DOWNLOAD_Q)
        self.chunk_queue = asyncio.Queue(maxsize=MAX_CHUNK_Q)

    async def run(self):
        self.fetchMetaData()

        pdfs_to_download = asyncio.create_task(self.download_all_pdfs())
        pdf_chunks = asyncio.create_task(self.chunkPDFs())
        embeddings_task = asyncio.create_task(self.embedChunks())

        await asyncio.gather(pdfs_to_download, pdf_chunks, embeddings_task)
    
    def fetchMetaData(self) -> None:
        # link = (
        #     BASE_URL
        #     + 'search_query=cat:%s&start=%i&max_results=%i&sortBy=submittedDate&sortOrder=descending'
        #     % (category, 0, max_results)
        # )
        for cat in CATEGORIES:
            search = arxiv.Search(
                query=cat,
                max_results=MAX_RESULTS,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            for paper in self.client.results(search):
                if not paper.pdf_url:
                    continue

                metadata = PaperMetaData(
                    arxiv_id=paper.entry_id,
                    title=paper.title,
                    authors=[author.name for author in paper.authors],
                    categories=paper.categories,
                    published=paper.published,
                    summary=paper.summary,
                    pdf_url=paper.pdf_url,
                )

                papers[cat].append(metadata)
    
    async def download_all_pdfs(self) -> None:
        
        sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

        async def download_pdf(meta: PaperMetaData):
            try:
                async with sem:
                    # async with session.get(url=meta.pdf_url) as response:
                    #     resp = await response.read()
                    search = arxiv.Search(id_list=[meta.arxiv_id])
                    paper = next(self.client.results(search))
                    paper.download_pdf(dirpath="./papers", filename=f"{meta.arxiv_id}.pdf")

                    path = f"./papers/{meta.arxiv_id}.pdf"
                    await self.download_queue.put((meta, path))
                    print(f"Successfully fetched pdf {meta.pdf_url} with category {meta.categories[0]}")
            except Exception as e:
                print(f"Unable to get url {meta.pdf_url} with error: {e}")


        timeout = aiohttp.ClientTimeout(total=180, sock_connect=30, sock_read=150)
        connector = aiohttp.TCPConnector(limit=DOWNLOAD_CONCURRENCY, limit_per_host=DOWNLOAD_CONCURRENCY)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector):
            tasks = [
                download_pdf(paper)
                for cat in CATEGORIES
                for paper in papers[cat]
            ]

            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                print("Error in downloading pdfs", e)
            finally:
                # Close the channel
                await self.download_queue.put(None)


    async def chunkPDFs(self) -> None:
        # Chunker worker function
        def chunker(metadata: PaperMetaData, filepath: str):
            full_text = []

            with fitz.open(filename=filepath) as doc:
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
                item = await self.download_queue.get()
                if item is None:
                    # Close chunk queue channel
                    await self.chunk_queue.put(None)
                    break

                metadata, pdf = item
                try:
                    result = await loop.run_in_executor(executor, chunker, metadata, pdf)
                except Exception as exc:
                    print(f"Chunking failed for {metadata.pdf_url}: {exc}")
                    continue
                

                print("Successfully chunked ", metadata.pdf_url)

                for idx, chunk in enumerate(result):
                    await self.chunk_queue.put((metadata, Chunk(
                        arxiv_id=metadata.arxiv_id,
                        chunk_id=str(uuid.uuid4()),
                        chunk_index=idx,
                        chunk_text=" ".join(chunk),
                        token_count=len(chunk)
                    )))


    async def embedChunks(self) -> None:
        def embed_batch(texts: List[str]):
            return MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
            batch: List[tuple[PaperMetaData, Chunk]] = []
            ch_closed = False
            while True:
                item = await self.chunk_queue.get()
                if item is None:
                    ch_closed = True
                else:
                    batch.append(item)

                # Try to fill the batch quickly without blocking
                while len(batch) < EMBED_BATCH_SIZE:
                    try:
                        next_item = self.chunk_queue.get_nowait()
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

# async def pipeline() -> None:
#     download_queue = asyncio.Queue(maxsize=MAX_DOWNLOAD_Q)
#     chunk_queue = asyncio.Queue(maxsize=MAX_CHUNK_Q)

#     print("Fetching metadata")
#     fetchMetaData()
#     print("Done fetching metadata")

#     pdfs_to_download = asyncio.create_task(download_all_pdfs(download_queue))
#     pdf_chunks = asyncio.create_task(chunkPDFs(download_queue, chunk_queue))
#     embeddings_task = asyncio.create_task(embedChunks(chunk_queue))

#     await asyncio.gather(pdfs_to_download, pdf_chunks, embeddings_task)


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
    paperfetcher = PaperFetcher()
    asyncio.run(paperfetcher.run())