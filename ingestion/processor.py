from sentence_transformers import SentenceTransformer
import fitz
import re
from typing import List


class PaperProcessor:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap  

        self.model = SentenceTransformer("all-MiniLM-L6-V2")

    async def process_pdf(self, filename: str, dirpath: str = "./papers", cuda: bool = False):
        pdf_path = dirpath + filename

        full_text = await self._get_pdf_text(pdf_path)

        text_chunks = await self._chunk_pdf(full_text)

        embedded_chunks = await self._embed_chunks(text_chunks, cuda)

        return text_chunks, embedded_chunks

    async def _get_pdf_text(self, pdf_path: str):
        full_text = []
        with fitz.open(filename=pdf_path) as doc:
            for page in doc:
                full_text.append(page.get_text())

        return full_text
            
    async def _chunk_pdf(self, full_text: List[str]):
        tokens = re.split(r"\s+", "\n".join(full_text))
        pdf_chunks = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i : i + self.chunk_size]
            if not chunk:
                break
            
            pdf_chunks.append(chunk)
            i += (self.chunk_size - self.chunk_overlap)

        return pdf_chunks

    def _embed_chunks(self, chunkText: List[str], cuda: bool = False):
        return self.model.encode(chunkText, convert_to_numpy=True, normalize_embeddings=True).tolist()