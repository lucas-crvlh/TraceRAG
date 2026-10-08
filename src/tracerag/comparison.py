"""Paired evaluation of BM25 and semantic retrieval over frozen judgments (M006).

Two golden sets are evaluated separately and never averaged together:

* ``evaluation/golden_set.jsonl`` (M004): ten English queries with the legacy labels,
  one relevant chunk per answerable query;
* ``evaluation/golden_set_m006.jsonl``: 24 queries in twelve Portuguese/English
  translation pairs, 16 positive and 8 negative, judged under ``direct-contribution-v2``.

Recall@3 = relevant chunks in the first three hits / R, and RR@3 = 1 / rank of the
first relevant hit within three, else 0. Means are macro averages over positive
queries only; negative queries have no metric (N/A) and are shown with their
candidates. Methods are compared in pairs inside each set: semantic minus BM25 per
query, with better/equal/worse counts on RR@3 (primary) and, separately, on Recall@3.
The chance reference is analytic and uses the real R of each query. Labels are
applied here, after the searches; the search functions never receive them.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from fractions import Fraction
from pathlib import Path

from . import semantic
from .corpus import json_bytes, read_chunks, sha256_hex, write_new_file
from .evaluation import TOP_K, chance_reference, load_golden_set, recall_at_k, reciprocal_rank_at_k, run_evaluation
from .retrieval import QUERY_RULE, RANKING, TOKENIZER, LexicalIndex

FREEZE_SCHEMA = 1
REPORT_SCHEMA = 1
RELEVANCE_POLICY = "direct-contribution-v2"
LANGUAGES = ("pt", "en")
M006_KEYS = (
    "id", "query", "language", "answerable", "relevant_chunk_ids", "evidence_locators", "judgment_notes",
    "pair_id", "query_provenance", "judgment_provenance", "relevance_policy", "question_scope",
    "relevance_basis", "non_relevant_notes",
)
_LOCATOR_KEYS = ("document_id", "unit_locator", "part", "source_sha256", "source_url")
_PAIR_SHARED = ("answerable", "relevant_chunk_ids", "evidence_locators", "judgment_notes", "relevance_policy",
                "question_scope", "relevance_basis", "non_relevant_notes")
FROZEN_INPUTS = {
    "golden_set_m006": "evaluation/golden_set_m006.jsonl",
    "golden_set_m004": "evaluation/golden_set.jsonl",
    "chunks": "data/m004/chunks.jsonl",
    "corpus_manifest": "corpus_manifest.yaml",
    "model_lock": "config/m006_model.json",
    "download_lock": "config/m006_downloads.json",
    "requirements_lock": "requirements-m006-win-cpu.lock",
    "snapshots_json": "data/m004/snapshots.json",
    "snapshot_tn1297_s2": "data/m004/raw/tn1297-s2.html",
    "snapshot_tn1297_s3": "data/m004/raw/tn1297-s3.html",
    "snapshot_tn1297_s4": "data/m004/raw/tn1297-s4.html",
    "m004_evaluation": "data/m004/evaluation.json",
}
RESOURCE_SOURCES = {
    "requirements_lock": "M006_REQUIREMENTS_WIN_CPU_v2.lock",
    "model_lock": "M006_MODEL_LOCK_v2.json",
    "download_lock": "M006_DOWNLOADS_LOCK_v2.json",
    "golden_set_m006": "M006_GOLDEN_SET_v2.jsonl",
}
RESOURCES_FROZEN_BY = "M006-SEMANTIC-v2"
LIMITATIONS = (
    "Fifteen chunks from three sections of one NIST publication, in English; Portuguese queries are "
    "compared with English text and nothing is translated at run time.",
    "The 24 M006 queries are twelve translation pairs, not 24 independent observations, and were written "
    "knowing the corpus; there is no blind split.",
    "Judgments are proposals of the Architect checked by the Auditor under direct-contribution-v2; five "
    "Portuguese questions are Lucas's originals, the other texts were written by the Codex Architect.",
    "The M004 HTML extraction lost some formulas; Recall@3 counts judged chunks, not complete answers, and "
    "one relevant hit does not cover every aspect of a question.",
    "The pretrained model may have seen NIST material during training.",
    "BM25 and cosine scores have different natures and are never compared with each other; a high score "
    "on a negative query is not evidence of support or of a failure.",
    "No claim of gain, generalization or significance: this is a diagnostic to decide whether a hybrid "
    "mission or a larger evaluation is worth it.",
)


class ComparisonError(ValueError):
    """A golden set, the freeze or a comparison violates the M006 contract."""


def _fraction(value: Fraction) -> dict:
    return {"fraction": f"{value.numerator}/{value.denominator}", "value": float(value)}


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- golden sets


def load_m006_golden_set(path: Path, chunks: list[dict]) -> list[dict]:
    """Read and validate the M006 golden set against the frozen chunks (no retrieval involved)."""
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    text = Path(path).read_bytes().decode("utf-8")
    if not text.endswith("\n") or "\r" in text:
        raise ComparisonError("golden set must use LF line endings, including the last line")
    items, seen = [], set()
    for number, line in enumerate(text[:-1].split("\n"), start=1):
        if not line.strip():
            raise ComparisonError(f"line {number} is blank")
        item = json.loads(line)
        _validate_m006_item(item, f"line {number}", by_id)
        if item["id"] in seen:
            raise ComparisonError(f"duplicate query id {item['id']!r}")
        seen.add(item["id"])
        items.append(item)
    if not items:
        raise ComparisonError("the golden set is empty")
    pairs: dict = {}
    for item in items:
        pairs.setdefault(item["pair_id"], []).append(item)
    for pair_id, members in pairs.items():
        if sorted(member["language"] for member in members) != sorted(LANGUAGES):
            raise ComparisonError(f"{pair_id} must have exactly one pt and one en query")
        first, second = members
        for key in _PAIR_SHARED:
            if first[key] != second[key]:
                raise ComparisonError(f"{pair_id}: {key} differs between the two languages")
        if first["query_provenance"].get("original_user_question") != second["query_provenance"].get("original_user_question"):
            raise ComparisonError(f"{pair_id}: original_user_question differs between the two languages")
    return items


def _validate_m006_item(item, where: str, by_id: dict) -> None:
    if not isinstance(item, dict) or set(item) != set(M006_KEYS):
        raise ComparisonError(f"{where}: keys must be exactly {list(M006_KEYS)}")
    for key in ("id", "query", "judgment_notes", "pair_id", "question_scope", "non_relevant_notes"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise ComparisonError(f"{where}: {key} must be a non-empty string")
    if item["language"] not in LANGUAGES:
        raise ComparisonError(f"{where}: language must be one of {LANGUAGES}")
    if not isinstance(item["answerable"], bool):
        raise ComparisonError(f"{where}: answerable must be true or false")
    if item["relevance_policy"] != RELEVANCE_POLICY:
        raise ComparisonError(f"{where}: relevance_policy must be {RELEVANCE_POLICY!r}")
    relevant = item["relevant_chunk_ids"]
    if not isinstance(relevant, list) or not all(isinstance(value, str) and value for value in relevant):
        raise ComparisonError(f"{where}: relevant_chunk_ids must be a list of non-empty strings")
    if len(set(relevant)) != len(relevant):
        raise ComparisonError(f"{where}: relevant_chunk_ids has duplicates")
    if item["answerable"] != bool(relevant):
        raise ComparisonError(f"{where}: answerable queries need relevant chunks and negatives must have none")
    unknown = [chunk_id for chunk_id in relevant if chunk_id not in by_id]
    if unknown:
        raise ComparisonError(f"{where}: unknown relevant chunk ids {unknown}")
    locators = item["evidence_locators"]
    if not isinstance(locators, list) or len(locators) != len(relevant):
        raise ComparisonError(f"{where}: one evidence locator is needed per relevant chunk")
    for chunk_id, locator in zip(relevant, locators):
        chunk = by_id[chunk_id]
        if not isinstance(locator, dict) or set(locator) != set(_LOCATOR_KEYS):
            raise ComparisonError(f"{where}: evidence locator keys must be {list(_LOCATOR_KEYS)}")
        if any(locator[key] != chunk[key] for key in _LOCATOR_KEYS):
            raise ComparisonError(f"{where}: evidence locator does not match chunk {chunk_id}")
    basis = item["relevance_basis"]
    if not isinstance(basis, dict) or list(basis) != relevant:
        raise ComparisonError(f"{where}: relevance_basis must have one entry per relevant chunk, in order")
    if not all(isinstance(value, str) and value.strip() for value in basis.values()):
        raise ComparisonError(f"{where}: relevance_basis entries must be non-empty strings")
    provenance = item["query_provenance"]
    if not isinstance(provenance, dict) or not {"author", "origin", "original_user_question"} <= set(provenance):
        raise ComparisonError(f"{where}: query_provenance needs author, origin and original_user_question")
    if not isinstance(item["judgment_provenance"], dict) or "author" not in item["judgment_provenance"]:
        raise ComparisonError(f"{where}: judgment_provenance needs an author")


def recall_ceiling(n_relevant: int, k: int = TOP_K) -> Fraction:
    """Highest Recall@k reachable when R relevant chunks exist: min(k, R) / R."""
    if n_relevant < 1:
        raise ValueError("the ceiling is undefined without relevant chunks")
    return Fraction(min(k, n_relevant), n_relevant)


def m006_summary(items: list[dict], k: int = TOP_K) -> dict:
    """Counts, R per pair, references and the macro Recall@k ceiling (per language and overall)."""
    positives = [item for item in items if item["answerable"]]
    if not positives:
        raise ComparisonError("no positive queries")
    ceilings = {}
    for name, members in (("all", positives), *((lang, [i for i in positives if i["language"] == lang]) for lang in LANGUAGES)):
        if members:
            ceilings[name] = _fraction(sum((recall_ceiling(len(i["relevant_chunk_ids"]), k) for i in members), Fraction(0)) / len(members))
    r_by_pair = {}
    for item in positives:
        r_by_pair.setdefault(item["pair_id"], len(item["relevant_chunk_ids"]))
    return {
        "cases": len(items), "languages": {lang: sum(i["language"] == lang for i in items) for lang in LANGUAGES},
        "positives": len(positives), "negatives": len(items) - len(positives),
        "pairs": len({item["pair_id"] for item in items}),
        "relevance_references": sum(len(item["relevant_chunk_ids"]) for item in items),
        "distinct_relevant_chunks": len({chunk for item in items for chunk in item["relevant_chunk_ids"]}),
        "r_by_pair": dict(sorted(r_by_pair.items())), "macro_recall_ceiling": ceilings,
        "negative_ids": [item["id"] for item in items if not item["answerable"]],
    }


# ---------------------------------------------------------------- freeze


def input_hashes(repo_root: Path) -> dict:
    return {name: semantic.file_info(Path(repo_root) / path, repo_root) for name, path in FROZEN_INPUTS.items()}


def create_freeze(repo_root: Path, output: Path, context: dict) -> dict:
    """Record inputs, locks, parameters, authorship and execution context in a new freeze file."""
    for key in ("packet_sha256", "pre_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", context.get(key) or ""):
            raise ComparisonError(f"{key} must be a lowercase SHA-256")
    for key in ("execution_revision", "semantic_t0"):
        if not isinstance(context.get(key), str) or not context[key].strip():
            raise ComparisonError(f"{key} must be a non-empty string")
    repo_root = Path(repo_root)
    chunks = read_chunks(repo_root / FROZEN_INPUTS["chunks"])
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    legacy = load_golden_set(repo_root / FROZEN_INPUTS["golden_set_m004"], chunk_ids)
    items = load_m006_golden_set(repo_root / FROZEN_INPUTS["golden_set_m006"], chunks)
    lock = semantic.load_model_lock(repo_root / FROZEN_INPUTS["model_lock"])
    downloads = json.loads((repo_root / FROZEN_INPUTS["download_lock"]).read_bytes().decode("utf-8"))
    requirements_header = (repo_root / FROZEN_INPUTS["requirements_lock"]).read_bytes().decode("utf-8").split("\n", 1)[0]
    inputs = input_hashes(repo_root)
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "execution": {
            "revision": context["execution_revision"], "packet_sha256": context["packet_sha256"],
            "pre_sha256": context["pre_sha256"], "semantic_t0": context["semantic_t0"],
            "note": "frozen before the first NIST search, real encoding or measurement of either method",
        },
        "frozen_at": dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "inputs": inputs,
        "resource_provenance": {
            "frozen_by_revision": RESOURCES_FROZEN_BY,
            "files": [{"source_name": RESOURCE_SOURCES[key], "repository_path": inputs[key]["path"],
                       "bytes": inputs[key]["bytes"], "sha256": inputs[key]["sha256"]} for key in RESOURCE_SOURCES],
            "mission_fields": {"model_lock": lock.get("mission"), "download_lock": downloads.get("mission"),
                               "requirements_lock_header": requirements_header},
            "note": "copied byte for byte; the v2 mission fields identify the revision that froze these inputs",
        },
        "golden_set_m006": {**m006_summary(items), "relevance_policy": RELEVANCE_POLICY,
                            "authorship": _authorship(items)},
        "golden_set_m004": {"cases": len(legacy), "positives": sum(i["answerable"] for i in legacy),
                            "negatives": sum(not i["answerable"] for i in legacy),
                            "labels": "legacy M004 labels, one relevant chunk per answerable query; the "
                                      "direct-contribution-v2 rubric is not applied to them"},
        "chunks": {"count": len(chunks), "ids": sorted(chunk_ids)},
        "parameters": parameters(),
        "limitations": list(LIMITATIONS),
    }
    write_new_file(Path(output), json_bytes(freeze))
    return {"status": "written", "path": Path(output).resolve().relative_to(repo_root.resolve()).as_posix(),
            "sha256": sha256_hex(Path(output).read_bytes()), "freeze": freeze}


def _authorship(items: list[dict]) -> dict:
    users = [i["id"] for i in items if str(i["query_provenance"]["author"]).startswith("Lucas")]
    return {"user_original_questions": users, "codex_texts": [i["id"] for i in items if i["id"] not in users],
            "judgments": sorted({i["judgment_provenance"]["author"] for i in items}),
            "note": "authorship as declared in the golden set; judgments are not attributed to the user"}


def verify_freeze(repo_root: Path, freeze_path: Path) -> dict:
    """Require every frozen input to keep its SHA-256 before any search, encoding or measurement."""
    data = Path(freeze_path).read_bytes()
    freeze = json.loads(data.decode("utf-8"))
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise ComparisonError("freeze file has another schema")
    current = input_hashes(repo_root)
    if sorted(freeze.get("inputs") or {}) != sorted(current):
        raise ComparisonError("freeze file lists other inputs")
    changed = [name for name, info in current.items() if freeze["inputs"][name]["sha256"] != info["sha256"]]
    if changed:
        raise ComparisonError(f"inputs changed after the freeze: {changed}")
    return {"path": Path(freeze_path).resolve().relative_to(Path(repo_root).resolve()).as_posix(),
            "sha256": sha256_hex(data), "execution_revision": freeze["execution"]["revision"],
            "frozen_at": freeze["frozen_at"], "inputs_verified": len(current)}


def parameters() -> dict:
    return {
        "top_k": TOP_K,
        "bm25": {"index": "SQLite FTS5 in memory", "tokenizer": TOKENIZER, "query_rule": QUERY_RULE, "ranking": RANKING},
        "semantic": {
            "model": semantic.MODEL_ID, "revision": semantic.MODEL_REVISION, "query_prefix": semantic.QUERY_PREFIX,
            "passage_prefix": semantic.PASSAGE_PREFIX, "tokenization": semantic.TOKENIZATION,
            "pooling": semantic.POOLING, "normalization": semantic.NORMALIZATION, "score": semantic.SCORE,
            "tie_policy": semantic.TIE_POLICY, "dtype": "float32", "device": "cpu", "batch_size_chunks": 1,
            "batch_size_queries": 1, "threads": {"intra_op": 1, "inter_op": 1, **semantic.THREAD_ENV},
            "seed": 0, "deterministic_algorithms": True, "attn_implementation": "eager",
            "tolerance": {"atol": semantic.ABS_TOL, "rtol": 0, "unit_norm_atol": semantic.UNIT_NORM_TOL},
        },
        "metrics": {"recall": "relevant retrieved in top 3 / R", "rr": "1 / rank of first relevant in top 3, else 0",
                    "means": "macro over positive queries; negatives are N/A",
                    "primary_paired_comparison": "RR@3", "secondary_paired_comparison": "Recall@3"},
        "relevance_policy_m006": RELEVANCE_POLICY,
    }


# ---------------------------------------------------------------- BM25 history


HISTORY_FIELDS = ("results", "aggregates", "chance_reference")


def reproduce_bm25_history(chunks: list[dict], legacy_items: list[dict], historical_path: Path, repo_root: Path) -> dict:
    """Run the M004 evaluation again and compare its deterministic fields with the historical file."""
    historical = json.loads(Path(historical_path).read_bytes().decode("utf-8"))
    current = run_evaluation(chunks, legacy_items, TOP_K)
    differences = [field for field in HISTORY_FIELDS if current[field] != historical[field]]
    expected_parameters = {"top_k": TOP_K, "tokenizer": TOKENIZER, "query_rule": QUERY_RULE, "ranking": RANKING}
    if any(historical["parameters"].get(key) != value for key, value in expected_parameters.items()):
        differences.append("parameters")
    return {
        "historical_file": semantic.file_info(historical_path, repo_root),
        "compared_fields": [*HISTORY_FIELDS, "parameters (top_k, tokenizer, query_rule, ranking)"],
        "status": "identical" if not differences else "differs", "differences": differences,
        "mean_recall_at_3": current["aggregates"]["mean_recall_at_3"], "mrr_at_3": current["aggregates"]["mrr_at_3"],
        "per_query": [{"id": row["id"], "hits": [[hit["rank"], hit["chunk_id"], hit["score"]] for hit in row["hits"]]}
                      for row in current["results"]],
    }


# ---------------------------------------------------------------- paired comparison


def score_ranking(ranked_ids: list[str], relevant: list[str], k: int = TOP_K) -> dict:
    """Recall@k, RR@k, hit, retrieved count and first relevant rank for one ranking."""
    top = ranked_ids[:k]
    wanted = set(relevant)
    retrieved = sum(1 for chunk_id in top if chunk_id in wanted)
    return {"recall_at_3": recall_at_k(ranked_ids, relevant, k), "rr_at_3": reciprocal_rank_at_k(ranked_ids, relevant, k),
            "hit": retrieved > 0, "relevant_retrieved": retrieved,
            "first_relevant_rank": next((rank for rank, cid in enumerate(top, start=1) if cid in wanted), None)}


def outcome(semantic_value: Fraction, bm25_value: Fraction) -> str:
    if semantic_value > bm25_value:
        return "semantic_better"
    if semantic_value < bm25_value:
        return "bm25_better"
    return "equal"


def evaluate_set(items: list[dict], chunks_by_id: dict, run_bm25, run_semantic, k: int = TOP_K) -> dict:
    """Run both methods for every query of one set, then apply the frozen labels."""
    n_chunks = len(chunks_by_id)
    rows, internal = [], []
    for item in items:
        relevant = list(dict.fromkeys(item["relevant_chunk_ids"]))
        methods = {"bm25": run_bm25(item["query"], k), "semantic": run_semantic(item["query"], k)}
        provenance = item.get("query_provenance") or {}
        row = {"id": item["id"], "language": item["language"], "pair_id": item.get("pair_id"),
               "query_author": provenance.get("author"), "original_user_question": provenance.get("original_user_question"),
               "query": item["query"], "answerable": item["answerable"], "relevant_chunk_ids": relevant, "R": len(relevant)}
        scores = {}
        for name, result in methods.items():
            for hit in result["hits"]:
                chunk = chunks_by_id[hit["chunk_id"]]
                hit.update({"document_id": chunk["document_id"], "unit_locator": chunk["unit_locator"],
                            "relevant": hit["chunk_id"] in relevant})
            if item["answerable"]:
                scores[name] = score_ranking([hit["chunk_id"] for hit in result["hits"]], relevant, k)
                result.update({key: (_fraction(value) if isinstance(value, Fraction) else value)
                               for key, value in scores[name].items()})
            row[name] = result
        if item["answerable"]:
            reference = chance_reference(n_chunks, len(relevant), k)
            row["recall_ceiling"] = _fraction(recall_ceiling(len(relevant), k))
            row["chance"] = {key: _fraction(reference[key]) for key in ("expected_recall", "hit_probability", "expected_rr")}
            row["difference_semantic_minus_bm25"] = {
                key: _fraction(scores["semantic"][key] - scores["bm25"][key]) for key in ("rr_at_3", "recall_at_3")}
            row["outcome_rr_at_3"] = outcome(scores["semantic"]["rr_at_3"], scores["bm25"]["rr_at_3"])
            row["outcome_recall_at_3"] = outcome(scores["semantic"]["recall_at_3"], scores["bm25"]["recall_at_3"])
            internal.append((item, scores, reference, recall_ceiling(len(relevant), k)))
        else:
            row["metrics"] = "N/A: no supporting evidence was judged in this corpus; candidates are shown only"
        rows.append(row)
    return {"questions": rows, "aggregates": aggregate(internal, items)}


def aggregate(internal: list[tuple], items: list[dict]) -> dict:
    slices = {"all_positive": internal}
    for lang in LANGUAGES:
        members = [entry for entry in internal if entry[0]["language"] == lang]
        if members:
            slices[f"{lang}_positive"] = members
    out = {}
    for name, members in slices.items():
        n = len(members)
        block = {"queries": n, "ids": [entry[0]["id"] for entry in members]}
        for method in ("bm25", "semantic"):
            recall = sum((entry[1][method]["recall_at_3"] for entry in members), Fraction(0))
            rr = sum((entry[1][method]["rr_at_3"] for entry in members), Fraction(0))
            hits = sum(1 for entry in members if entry[1][method]["hit"])
            block[method] = {"mean_recall_at_3": _fraction(recall / n), "mrr_at_3": _fraction(rr / n),
                             "queries_with_relevant_hit": hits, "hit_rate": _fraction(Fraction(hits, n))}
        for metric in ("rr_at_3", "recall_at_3"):
            counts = {"semantic_better": 0, "equal": 0, "bm25_better": 0}
            for entry in members:
                counts[outcome(entry[1]["semantic"][metric], entry[1]["bm25"][metric])] += 1
            block[f"paired_{metric}"] = counts
        block["mean_recall_ceiling"] = _fraction(sum((entry[3] for entry in members), Fraction(0)) / n)
        block["chance"] = {key: _fraction(sum((entry[2][key] for entry in members), Fraction(0)) / n)
                           for key in ("expected_recall", "hit_probability", "expected_rr")}
        out[name] = block
    negatives = [item for item in items if not item["answerable"]]
    out["negatives"] = {"queries": len(negatives), "ids": [item["id"] for item in negatives],
                        "metrics": "N/A, outside every denominator"}
    return out


def bm25_runner(chunks: list[dict]):
    index = LexicalIndex(chunks)

    def run(query: str, k: int) -> dict:
        result = index.search(query, k)
        return {"status": result["status"], "terms": result["terms"],
                "hits": [{"rank": hit["rank"], "score": hit["score"], "chunk_id": hit["chunk_id"]} for hit in result["hits"]]}
    return run


def semantic_runner(sem_index, encoder):
    def run(query: str, k: int) -> dict:
        vector, info = encoder.encode(query, "query")
        ranked = semantic.rank_scores(sem_index.embeddings, sem_index.chunk_ids, vector, k, encoder.np)
        return {"query_tokens": info,
                "hits": [{"rank": rank, "cosine_score": score, "chunk_id": chunk_id} for rank, chunk_id, score in ranked]}
    return run


def comparison_sets(chunks: list[dict], legacy_items: list[dict], m006_items: list[dict], run_bm25, run_semantic) -> dict:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    return {
        "m004_legacy": {"golden_set": FROZEN_INPUTS["golden_set_m004"],
                        "labels": "legacy M004 labels (one relevant chunk per answerable query)",
                        **evaluate_set(legacy_items, by_id, run_bm25, run_semantic)},
        "m006": {"golden_set": FROZEN_INPUTS["golden_set_m006"], "labels": RELEVANCE_POLICY,
                 **evaluate_set(m006_items, by_id, run_bm25, run_semantic)},
    }


# ---------------------------------------------------------------- reproduction


def compare_reports(first: dict, second: dict, atol: float = semantic.ABS_TOL) -> dict:
    """Same ids, ranks, labels and metrics; BM25 scores equal; cosine scores within atol (rtol 0)."""
    differences, worst = [], 0.0
    for set_name in ("m004_legacy", "m006"):
        a, b = first["sets"][set_name], second["sets"][set_name]
        if a["aggregates"] != b["aggregates"]:
            differences.append(f"{set_name}: aggregates differ")
        if [q["id"] for q in a["questions"]] != [q["id"] for q in b["questions"]]:
            differences.append(f"{set_name}: question order differs")
            continue
        for qa, qb in zip(a["questions"], b["questions"]):
            for method, score_key in (("bm25", "score"), ("semantic", "cosine_score")):
                ha, hb = qa[method]["hits"], qb[method]["hits"]
                if [(h["rank"], h["chunk_id"], h["relevant"]) for h in ha] != [(h["rank"], h["chunk_id"], h["relevant"]) for h in hb]:
                    differences.append(f"{set_name}/{qa['id']}/{method}: ranking differs")
                    continue
                for x, y in zip(ha, hb):
                    gap = abs(x[score_key] - y[score_key])
                    if method == "semantic":
                        worst = max(worst, gap)
                    if (method == "bm25" and gap != 0) or gap > atol:
                        differences.append(f"{set_name}/{qa['id']}/{method}: score differs by {gap}")
                for key in ("recall_at_3", "rr_at_3", "first_relevant_rank"):
                    if qa[method].get(key) != qb[method].get(key):
                        differences.append(f"{set_name}/{qa['id']}/{method}: {key} differs")
    if first.get("bm25_history_reproduction", {}).get("status") != second.get("bm25_history_reproduction", {}).get("status"):
        differences.append("bm25 history status differs")
    return {"status": "reproduced" if not differences else "differs", "differences": differences,
            "max_cosine_score_difference": worst, "tolerance": {"atol": atol, "rtol": 0},
            "compared": "question ids/order, ranks, chunk ids, labels, Recall@3, RR@3, first relevant rank, aggregates; "
                        "BM25 scores exactly, cosine scores within atol"}
