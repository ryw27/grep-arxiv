from dataclasses import dataclass
from typing import List
from uuid import UUID
import arxiv 
import datetime


# Data models
@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    categories: List[str]
    summary: str
    published: datetime.datetime
    pdf_url: str

@dataclass
class Chunk:
    arxiv_id: str
    chunk_id: UUID 
    chunk_index: int
    chunk_text: str
    token_count: int

@dataclass
class Embedding:
    chunk_id: UUID
    vector: List[float]

# For open search
@dataclass
class SearchResult:
    arxiv_id: str
    chunk_id: UUID
    title: str
    score: float
    snippet: str
    categories: List[str]
    published: datetime.datetime

