"""Postgres/pgvector connection management.

Responsibility: own the single connection pool the product index is
read from and written to, register pgvector's Python adapter on every
connection (so a Python `list[float]` round-trips as a `VECTOR`
column transparently), and ensure the `products` table + ANN index
exist. No other module talks to psycopg directly -- `IndexingService`
is the only consumer of the pool, exactly as it was the only consumer
of `embeddings.json` before this.
"""

from __future__ import annotations

import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.config import DATABASE_URL, EMBEDDING_DIM

logger = logging.getLogger(__name__)

_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    category TEXT NOT NULL,
    color TEXT,
    image_path TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS products_embedding_idx
    ON products USING hnsw (embedding vector_cosine_ops);
"""


def _configure(conn: Connection) -> None:
    """Per-connection setup run by the pool for every new connection.

    Requires the `vector` extension to already exist in the database
    -- `register_vector` looks up its type OIDs -- so this only ever
    runs after `create_pool` has created the extension below.
    """
    register_vector(conn)


def create_pool(database_url: str = DATABASE_URL) -> ConnectionPool:
    """Open a connection pool to Postgres and ensure the schema exists.

    Args:
        database_url: SQLAlchemy-style Postgres DSN. Defaults to
            `DATABASE_URL` from `app.config`.

    Returns:
        An open `ConnectionPool`, ready to use. The caller owns it and
        must call `.close()` at shutdown.

    Raises:
        psycopg.OperationalError: If Postgres is unreachable.
    """
    # A plain, unpooled connection first: the `vector` extension must
    # exist before any pooled connection can `register_vector` on
    # itself (that lookup needs the extension's types to be present).
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    pool = ConnectionPool(database_url, min_size=1, max_size=5, configure=_configure)
    pool.wait()

    with pool.connection() as conn:
        conn.execute(_TABLE_SQL)

    logger.info("Connected to Postgres and ensured the pgvector schema exists.")
    return pool
