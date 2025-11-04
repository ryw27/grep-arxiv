from fetcher import PaperFetcher
from processor import PaperProcessor
from typing import List 
from models import Chunk, Embedding, Paper
import asyncio
from db import Database

# Placeholder for postgres/open search
chunks: List[Chunk] = []
vector_db: List[Embedding] = []
papers: List[Paper] = []

class Pipeline:
    def __init__(self):
        # self.metadata_cache: Dict[str, dict] = {}
        self.fetcher = PaperFetcher()
        self.processor = PaperProcessor()

        self.database = Database()

    async def ingest_papers(self, query: str, max_results: int = 10) -> None:
        # 1. Fetch papers and store in directory
        pdf_files = await self.fetcher.fetch_pdfs(query, max_results)

        # 2. Process them 
        for pdf in pdf_files:
            filename, metadata = pdf

            self.database.insert_paper(metadata)

            text_chunks, embedded_chunks = await self.processor.process_pdf(filename)
            for text, embed in zip(text_chunks, embedded_chunks):
                self.database.insert_chunk(text)
                self.database.insert_embedding(embed)
                # chunk_id = uuid.uuid4()

                # chunks.append(Chunk(
                #     arxiv_id=metadata.arxiv_id,
                #     chunk_id=chunk_id,
                #     chunk_index=idx,
                #     chunk_text=text,
                #     token_count=len(text)
                # ))

                # vector_db.append(Embedding(
                #     chunk_id=chunk_id,
                #     embedding_id=uuid.uuid4(),
                #     embedding=embed
                # ))

if __name__ == "__main__":
    pipeline = Pipeline()
    asyncio.run(pipeline.ingest_papers("cs.AI"))
    

