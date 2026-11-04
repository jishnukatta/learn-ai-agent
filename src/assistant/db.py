"""db.py — connect to the PostgreSQL database (RDS) that stores our embeddings.

Connection settings come from environment variables (loaded from a local .env
file), never from code:

    DB_HOST      the RDS endpoint
    DB_PASSWORD  the master password
    DB_NAME      defaults to "assistant"
    DB_USER      defaults to "postgres"
    DB_PORT      defaults to 5432
"""

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def connect() -> psycopg.Connection:
    """Open a new connection. Use it as a context manager:

        with connect() as conn:
            conn.execute(...)

    On leaving the `with` block psycopg commits the transaction (or rolls back
    on error) and closes the connection.
    """
    return psycopg.connect(
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "assistant"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
        sslmode="require",   # encrypt the connection, as we did with psql
    )
