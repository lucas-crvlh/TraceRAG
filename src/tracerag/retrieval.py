"""Lexical retrieval baseline: SQLite FTS5 in memory, ranked by ``bm25()``.

The index holds only the chunk text (tokenizer ``unicode61``, no stemming and
no stop list); metadata is neither indexed nor boosted. The user query never
reaches FTS5 as free syntax: it is reduced to alphanumeric terms, each term is
quoted and the terms are joined with OR, then passed as an SQL parameter.
"""

from __future__ import annotations

import re
import sqlite3

TOKENIZER = "unicode61"
QUERY_RULE = (
    "alphanumeric terms, case-folded, de-duplicated in order of first occurrence, "
    "each quoted as an FTS5 string and joined with OR"
)
RANKING = "FTS5 bm25() ascending (lower is a better match); ties broken by chunk_id ascending"
SCORE_NOTE = (
    "score is the FTS5 bm25() ranking value (lower means a better lexical match); "
    "it is not a probability that the passage supports an answer"
)
_TERM_RE = re.compile(r"[^\W_]+")
_HIT_FIELDS = (
    "chunk_id", "document_id", "publication_id", "title", "section", "unit_locator", "part",
    "source_url", "source_version", "source_sha256", "snapshot_path", "retrieved_at",
    "text_sha256", "text", "attribution",
)


def query_terms(query: str | None) -> list[str]:
    """Alphanumeric terms of ``query``, case-folded and de-duplicated in order."""
    terms, seen = [], set()
    for match in _TERM_RE.findall(query or ""):
        term = match.casefold()
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def match_expression(terms: list[str]) -> str:
    """FTS5 MATCH expression with every term quoted and joined by OR."""
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


class LexicalIndex:
    """In-memory FTS5 index rebuilt from frozen chunks on every run."""

    def __init__(self, chunks: list[dict]) -> None:
        ids = [chunk["chunk_id"] for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_id values must be unique")
        self._chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        self._connection = sqlite3.connect(":memory:")
        self._connection.execute(
            f"CREATE VIRTUAL TABLE chunk_fts USING fts5(text, chunk_id UNINDEXED, tokenize = '{TOKENIZER}')"
        )
        self._connection.executemany(
            "INSERT INTO chunk_fts (text, chunk_id) VALUES (?, ?)",
            [(chunk["text"], chunk["chunk_id"]) for chunk in chunks],
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str | None, top_k: int) -> dict:
        """Return up to ``top_k`` hits with score, provenance and full chunk text."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        terms = query_terms(query)
        result = {"query": query, "terms": terms, "match": None, "top_k": top_k, "hits": []}
        if not terms:
            result["status"] = "no_query_terms"
            return result
        expression = match_expression(terms)
        rows = self._connection.execute(
            "SELECT chunk_id, bm25(chunk_fts) AS score FROM chunk_fts "
            "WHERE chunk_fts MATCH ? ORDER BY score ASC, chunk_id ASC LIMIT ?",
            (expression, top_k),
        ).fetchall()
        result["match"] = expression
        result["hits"] = [self._hit(rank, chunk_id, score) for rank, (chunk_id, score) in enumerate(rows, start=1)]
        result["status"] = "ok" if rows else "no_lexical_match"
        return result

    def _hit(self, rank: int, chunk_id: str, score: float) -> dict:
        chunk = self._chunks[chunk_id]
        hit = {"rank": rank, "score": score}
        hit.update({key: chunk.get(key) for key in _HIT_FIELDS})
        return hit
