"""Golden-set validation, Recall@3 and RR@3 per query, and an analytic chance reference.

Recall@k = relevant chunks found in the first k hits / relevant chunks judged.
RR@k = 1 / rank of the first relevant hit when that rank is at most k, else 0.
MRR@k is the mean RR@k over the answerable queries. Queries without support
in the corpus are reported apart and never enter these denominators.
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import comb
from pathlib import Path

from .retrieval import LexicalIndex

TOP_K = 3
_GOLDEN_KEYS = ("id", "query", "language", "answerable", "relevant_chunk_ids", "evidence_locators", "judgment_notes")


class GoldenSetError(ValueError):
    """The golden set violates its contract."""


def load_golden_set(path: Path, chunk_ids) -> list[dict]:
    """Read the JSONL golden set and check every judgment against existing chunk ids."""
    lines = Path(path).read_bytes().decode("utf-8").splitlines()
    items, seen = [], set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise GoldenSetError(f"line {number} is blank")
        item = json.loads(line)
        _validate_item(item, f"line {number}", chunk_ids)
        if item["id"] in seen:
            raise GoldenSetError(f"duplicate query id {item['id']!r}")
        seen.add(item["id"])
        items.append(item)
    if not items:
        raise GoldenSetError("the golden set is empty")
    return items


def _validate_item(item, where: str, chunk_ids) -> None:
    if not isinstance(item, dict) or set(item) != set(_GOLDEN_KEYS):
        raise GoldenSetError(f"{where}: keys must be exactly {list(_GOLDEN_KEYS)}")
    for key in ("id", "query", "judgment_notes"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise GoldenSetError(f"{where}: {key} must be a non-empty string")
    if item["language"] != "en":
        raise GoldenSetError(f"{where}: language must be 'en'")
    if not isinstance(item["answerable"], bool):
        raise GoldenSetError(f"{where}: answerable must be true or false")
    for key in ("relevant_chunk_ids", "evidence_locators"):
        values = item[key]
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise GoldenSetError(f"{where}: {key} must be a list of non-empty strings")
    relevant = item["relevant_chunk_ids"]
    if item["answerable"] and not relevant:
        raise GoldenSetError(f"{where}: an answerable query needs at least one relevant chunk")
    if not item["answerable"] and relevant:
        raise GoldenSetError(f"{where}: a query without support must have no relevant chunks")
    unknown = [chunk_id for chunk_id in relevant if chunk_id not in chunk_ids]
    if unknown:
        raise GoldenSetError(f"{where}: unknown relevant chunk ids {unknown}")


def _unique(values) -> list:
    return list(dict.fromkeys(values))


def recall_at_k(ranked_ids, relevant_ids, k: int = TOP_K) -> Fraction:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Recall@k is undefined without relevant chunks")
    return Fraction(len(relevant & set(_unique(ranked_ids)[:k])), len(relevant))


def reciprocal_rank_at_k(ranked_ids, relevant_ids, k: int = TOP_K) -> Fraction:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("RR@k is undefined without relevant chunks")
    for rank, chunk_id in enumerate(_unique(ranked_ids)[:k], start=1):
        if chunk_id in relevant:
            return Fraction(1, rank)
    return Fraction(0)


def chance_reference(n_chunks: int, n_relevant: int, k: int = TOP_K) -> dict:
    """Expectations for a uniformly random ranking without replacement (not a measurement).

    With N chunks, R relevant and depth k' = min(k, N): expected Recall@k = k'/N;
    probability of at least one relevant hit = 1 - C(N-R, k')/C(N, k');
    expected RR@k = sum over r=1..k' of (1/r) * C(N-r, R-1) / C(N, R).
    """
    if n_chunks < 1 or not 1 <= n_relevant <= n_chunks:
        raise ValueError("chance reference needs 1 <= R <= N")
    depth = min(k, n_chunks)
    total = comb(n_chunks, n_relevant)
    expected_rr = sum(
        (Fraction(1, rank) * Fraction(comb(n_chunks - rank, n_relevant - 1), total) for rank in range(1, depth + 1)),
        Fraction(0),
    )
    return {
        "n_chunks": n_chunks,
        "n_relevant": n_relevant,
        "k": depth,
        "expected_recall": Fraction(depth, n_chunks),
        "hit_probability": 1 - Fraction(comb(n_chunks - n_relevant, depth), comb(n_chunks, depth)),
        "expected_rr": expected_rr,
    }


def _fraction(value: Fraction) -> dict:
    return {"fraction": f"{value.numerator}/{value.denominator}", "value": float(value)}


def run_evaluation(chunks: list[dict], items: list[dict], k: int = TOP_K) -> dict:
    """Search every golden query once and compute per-query and aggregate metrics."""
    index = LexicalIndex(chunks)
    results, references = [], []
    recall_sum = rr_sum = Fraction(0)
    answerable = 0
    for item in items:
        search = index.search(item["query"], k)
        relevant = _unique(item["relevant_chunk_ids"])
        hits = [
            {
                "rank": hit["rank"], "score": hit["score"], "chunk_id": hit["chunk_id"],
                "document_id": hit["document_id"], "unit_locator": hit["unit_locator"], "part": hit["part"],
                "relevant": hit["chunk_id"] in relevant, "source_sha256": hit["source_sha256"],
                "text_sha256": hit["text_sha256"], "text": hit["text"],
            }
            for hit in search["hits"]
        ]
        entry = {
            "id": item["id"], "query": item["query"], "answerable": item["answerable"],
            "terms": search["terms"], "match": search["match"], "status": search["status"],
            "relevant_chunk_ids": relevant, "evidence_locators": item["evidence_locators"], "hits": hits,
        }
        if item["answerable"]:
            ranked = [hit["chunk_id"] for hit in hits]
            recall = recall_at_k(ranked, relevant, k)
            rr = reciprocal_rank_at_k(ranked, relevant, k)
            entry.update({
                "relevant_total": len(relevant),
                "relevant_retrieved": sum(1 for hit in hits if hit["relevant"]),
                "first_relevant_rank": next((hit["rank"] for hit in hits if hit["relevant"]), None),
                "recall_at_3": _fraction(recall),
                "rr_at_3": _fraction(rr),
            })
            recall_sum += recall
            rr_sum += rr
            answerable += 1
            references.append((item["id"], chance_reference(len(chunks), len(relevant), k)))
        else:
            entry["retrieved_any"] = bool(hits)
            entry["note"] = "lexical candidates only; no supporting evidence was judged in this corpus"
        results.append(entry)
    if not answerable:
        raise GoldenSetError("no answerable queries to evaluate")
    expectations = ("expected_recall", "hit_probability", "expected_rr")
    means = {key: sum((ref[key] for _id, ref in references), Fraction(0)) / answerable for key in expectations}
    chance_rows = [
        {"id": query_id, "n_relevant": ref["n_relevant"], **{key: _fraction(ref[key]) for key in expectations}}
        for query_id, ref in references
    ]
    return {
        "results": results,
        "aggregates": {
            "answerable_queries": answerable,
            "negative_queries": len(items) - answerable,
            "k": k,
            "recall_at_3_sum": str(recall_sum),
            "rr_at_3_sum": str(rr_sum),
            "mean_recall_at_3": _fraction(recall_sum / answerable),
            "mrr_at_3": _fraction(rr_sum / answerable),
        },
        "chance_reference": {
            "description": "analytic expectation for a uniformly random ranking without replacement; not a measurement",
            "n_chunks": len(chunks),
            "k": min(k, len(chunks)),
            "per_query": chance_rows,
            "mean_expected_recall": _fraction(means["expected_recall"]),
            "mean_hit_probability": _fraction(means["hit_probability"]),
            "mean_expected_rr": _fraction(means["expected_rr"]),
        },
    }
