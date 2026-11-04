"""rag.py — the retrieval half of RAG, backed by PostgreSQL + pgvector.

Two operations:
    index_notes()   read notes/*.md, embed each one, store it in the documents table
    search(q)       embed the question, return the closest stored notes

Run from the project folder:
    uv run python -m assistant.rag index
    uv run python -m assistant.rag search "how does my app get deployed?"
"""

import sys
from pathlib import Path

from assistant.db import connect
from assistant.embeddings import embed

# notes/ sits at the project root: this file is src/assistant/rag.py,
# so three parents up from the file is the project root.
NOTES_DIR = Path(__file__).resolve().parents[2] / "notes"


def _vector_literal(values: list[float]) -> str:
    """Turn [0.1, 0.2] into the text '[0.1,0.2]' that Postgres accepts for a vector."""
    return "[" + ",".join(str(v) for v in values) + "]"


def index_notes() -> int:
    """Embed every notes/*.md file (one chunk per file) and store it. Returns the count."""
    count = 0
    with connect() as conn:
        # Start clean so running this twice doesn't create duplicates.
        conn.execute("TRUNCATE documents")
        for path in sorted(NOTES_DIR.glob("*.md")):
            text = path.read_text().strip()
            conn.execute(
                "INSERT INTO documents (source, content, embedding) VALUES (%s, %s, %s::vector)",
                (path.name, text, _vector_literal(embed(text))),
            )
            count += 1
    return count


def search(question: str, k: int = 3) -> list[tuple[str, str, float]]:
    """Return the k notes closest in meaning to `question`: (source, content, distance)."""
    qvec = _vector_literal(embed(question))
    with connect() as conn:
        return conn.execute(
            """
            SELECT source, content, embedding <=> %s::vector AS distance
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, k),
        ).fetchall()


if __name__ == "__main__":
    command = sys.argv[1:2]
    if command == ["index"]:
        print(f"Indexed {index_notes()} notes")
    elif command == ["search"]:
        for source, content, distance in search(" ".join(sys.argv[2:])):
            print(f"{distance:.3f}  {source}: {content[:90]}")
    else:
        print('usage: python -m assistant.rag index | search "your question"')
