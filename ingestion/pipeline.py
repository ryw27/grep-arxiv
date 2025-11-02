from fetcher import PaperFetcher
from processor import PaperProcessor
from typing import List, Dict
from models import Chunk, Embedding, Paper
import uuid
import asyncio

# Placeholder for postgres/open search
chunks: List[Chunk] = []
vector_db: List[Embedding] = []
papers: List[Paper] = []

class Pipeline:
    def __init__(self):
        # self.metadata_cache: Dict[str, dict] = {}
        self.fetcher = PaperFetcher()
        self.processor = PaperProcessor()

    async def ingest_papers(self, query: str, max_results: int = 10) -> None:
        # 1. Fetch papers and store in directory
        pdf_files = await self.fetcher.fetch_pdfs(query, max_results)

        # 2. Process them 
        for pdf in pdf_files:
            filename, metadata = pdf
            papers.append(metadata)
            text_chunks, embedded_chunks = await self.processor.process_pdf(filename)

            for idx, (text, embed) in enumerate(zip(text_chunks, embedded_chunks)):
                chunk_id = uuid.uuid4()

                chunks.append(Chunk(
                    arxiv_id=metadata.arxiv_id,
                    chunk_id=chunk_id,
                    chunk_index=idx,
                    chunk_text=text,
                    token_count=len(text)
                ))

                vector_db.append(Embedding(
                    chunk_id=chunk_id,
                    vector=embed
                ))

if __name__ == "__main__":
    pipeline = Pipeline()
    asyncio.run(pipeline.ingest_papers("cs.AI"))
    

