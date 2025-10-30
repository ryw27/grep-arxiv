from dataclasses import dataclass
from typing import List
import arxiv 
import datetime


# Data models
@dataclass
class PaperMetaData:
    arxiv_id: str
    title: str
    authors: List[arxiv.Result.Author]
    categories: List[str]
    summary: str
    published: datetime.datetime
    pdf_url: str

@dataclass
class Chunk:
    arxiv_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str
    token_count: int

