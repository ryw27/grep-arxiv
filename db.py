import psycopg2
from dotenv import load_dotenv
import os
from ingestion.models import Chunk, Paper, Embedding

class Database:
    def __init__(self):
        load_dotenv()

        self.conn = psycopg2.connect(
            host=os.getenv("DATABASE_HOST"),
            database=os.getenv("DATABASE_NAME"),
            user=os.getenv("DATABASE_USER"),
            password=os.getenv("DATABASE_PASSWORD"),
            port=os.getenv("DATABASE_PORT"),
        )

        self.cursor = self.conn.cursor()
        self._create_tables()

    def close_database(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def _create_tables(self):
        # Make sure required extensions are available
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS \"pgvector\";")
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";")

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT[],
                categories TEXT[],
                summary TEXT,
                published TIMESTAMP,
                pdf_url TEXT
            );
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                arxiv_id TEXT REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                chunk_index INT,
                chunk_text TEXT,
                token_count INT
            );
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings  (
                embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                embedding vector(384)
            );
            """
        )
        self.conn.commit()

    def insert_chunk(self, chunk: Chunk):
        self.cursor.execute(
            "INSERT INTO chunks (chunk_id, arxiv_id, chunk_index, chunk_text, token_count) VALUES (%s, %s, %s, %s, %s)",
            (str(chunk.chunk_id), chunk.arxiv_id, chunk.chunk_index, chunk.chunk_text, chunk.token_count) 
        )
        self.conn.commit()

    def insert_paper(self, paper: Paper):
        self.cursor.execute(
            """
            INSERT INTO papers (arxiv_id, title, authors, categories, summary, published, pdf_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO NOTHING;
            """,
            (
                paper.arxiv_id,
                paper.title,
                paper.authors,
                paper.categories,
                paper.summary,
                paper.published,
                paper.pdf_url
            )
        )
        self.conn.commit()

    def insert_embedding(self, embedding: Embedding):
        self.cursor.execute(
            """
            INSERT INTO embeddings (embedding_id, chunk_id, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (embedding_id) DO NOTHING;
            """,
            (
                str(embedding.embedding_id),
                str(embedding.chunk_id),
                embedding.embedding  # Should be a list/array of floats
            )
        )
        self.conn.commit()

    def get_paper(self, arxiv_id: str):
        self.cursor.execute(
            "SELECT arxiv_id, title, authors, categories, summary, published, pdf_url FROM papers WHERE arxiv_id = %s",
            (arxiv_id,)
        )
        return self.cursor.fetchone()

    def get_chunk(self, chunk_id: str):
        self.cursor.execute(
            "SELECT chunk_id, arxiv_id, chunk_index, chunk_text, token_count FROM chunks WHERE chunk_id = %s",
            (chunk_id,)
        )
        return self.cursor.fetchone()

    def get_embedding(self, embedding_id: str):
        self.cursor.execute(
            "SELECT embedding_id, chunk_id, embedding FROM embeddings WHERE embedding_id = %s",
            (embedding_id,)
        )
        return self.cursor.fetchone()

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def __del__(self):
        self.close_database()

