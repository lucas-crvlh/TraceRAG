"""Command line interface: ``python -m tracerag {ingest,build,search,evaluate}``.

Run from the repository root with ``PYTHONPATH=src``. Output is JSON on
stdout, encoded as UTF-8. Only ``ingest`` uses the network; ``build``,
``search`` and ``evaluate`` work offline on the local snapshots and chunks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sqlite3
import sys
import time
from pathlib import Path

from . import MISSION, __version__
from .corpus import (
    EXTRACTOR_VERSION, MAX_CHUNK_WORDS, CorpusError, build_corpus_chunks, chunks_jsonl, ingest,
    json_bytes, load_manifest, load_snapshots, read_chunks, sha256_hex, write_new_file,
)
from .evaluation import TOP_K, GoldenSetError, load_golden_set, run_evaluation
from .retrieval import QUERY_RULE, RANKING, SCORE_NOTE, TOKENIZER, LexicalIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "corpus_manifest.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "m004"
SNAPSHOTS = OUTPUT_DIR / "snapshots.json"
CHUNKS = OUTPUT_DIR / "chunks.jsonl"
GOLDEN = REPO_ROOT / "evaluation" / "golden_set.jsonl"
LIMITATIONS = (
    "Ten queries written together with this small corpus: a diagnostic development set, "
    "not a blind test and not evidence of generalization.",
    "Three HTML sections of one publication (the NIST web adaptation of TN 1297), in English; "
    "formulas, images and tables are not interpreted.",
    "A lexical match is not proof that a passage supports an answer; queries without support "
    "can still return candidates. No answer generation or abstention is implemented.",
    "bm25() scores are ranking values of this index, not calibrated probabilities; no "
    "confidence threshold is defined.",
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _file_info(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": _relative(path), "bytes": len(data), "sha256": sha256_hex(data)}


def _environment() -> dict:
    connection = sqlite3.connect(":memory:")
    options = [row[0] for row in connection.execute("PRAGMA compile_options")]
    connection.close()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "sqlite": sqlite3.sqlite_version,
        "fts5_compiled": "ENABLE_FTS5" in options,
        "platform": platform.platform(),
        "package_version": __version__,
    }


def command_ingest(_args) -> dict:
    documents = load_manifest(MANIFEST, REPO_ROOT)
    return ingest(documents, REPO_ROOT, SNAPSHOTS, mission=MISSION)


def command_build(_args) -> dict:
    documents = load_manifest(MANIFEST, REPO_ROOT)
    records = load_snapshots(SNAPSHOTS, documents, REPO_ROOT)
    chunks = build_corpus_chunks(documents, records, REPO_ROOT)
    data = chunks_jsonl(chunks)
    summary = {
        "path": _relative(CHUNKS),
        "chunks": len(chunks),
        "chunks_sha256": sha256_hex(data),
        "extractor": EXTRACTOR_VERSION,
        "max_chunk_words": MAX_CHUNK_WORDS,
        "units": [
            {"chunk_id": c["chunk_id"], "unit_locator": c["unit_locator"], "part": c["part"], "word_count": c["word_count"]}
            for c in chunks
        ],
    }
    if CHUNKS.exists():
        # Reproduction: compare in memory with the frozen file, never rewrite it.
        summary["status"] = "reproduced_identical" if CHUNKS.read_bytes() == data else "differs_from_existing"
    else:
        write_new_file(CHUNKS, data)
        summary["status"] = "written"
    return summary


def command_search(args) -> dict:
    result = LexicalIndex(read_chunks(CHUNKS)).search(args.query, args.top_k)
    result["score_note"] = SCORE_NOTE
    return result


def evaluation_report() -> dict:
    """Deterministic evaluation content (everything except the ``run`` timing block)."""
    documents = load_manifest(MANIFEST, REPO_ROOT)
    records = load_snapshots(SNAPSHOTS, documents, REPO_ROOT)
    chunks = read_chunks(CHUNKS)
    items = load_golden_set(GOLDEN, {chunk["chunk_id"] for chunk in chunks})
    code = sorted((REPO_ROOT / "src" / "tracerag").glob("*.py"))
    report = {
        "mission": MISSION,
        "inputs": {
            "manifest": _file_info(MANIFEST),
            "snapshots_json": _file_info(SNAPSHOTS),
            "snapshots": [
                {key: records[d.id][key] for key in ("document_id", "local_path", "bytes", "sha256", "retrieved_at")}
                for d in documents
            ],
            "chunks": {**_file_info(CHUNKS), "count": len(chunks)},
            "golden_set": {**_file_info(GOLDEN), "count": len(items)},
            "code": [_file_info(path) for path in code],
        },
        "environment": _environment(),
        "parameters": {
            "top_k": TOP_K,
            "index": "SQLite FTS5 in memory, rebuilt from the frozen chunks on every run",
            "tokenizer": TOKENIZER,
            "query_rule": QUERY_RULE,
            "ranking": RANKING,
            "extractor": EXTRACTOR_VERSION,
            "max_chunk_words": MAX_CHUNK_WORDS,
        },
        "limitations": list(LIMITATIONS),
    }
    report.update(run_evaluation(chunks, items, TOP_K))
    return report


def command_evaluate(args) -> dict:
    started = time.perf_counter()
    report = evaluation_report()
    report["run"] = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "note": "timing is a local observation, excluded from reproducibility comparisons",
    }
    if args.output:
        target = Path(args.output)
        target = (target if target.is_absolute() else REPO_ROOT / target).resolve()
        if target.suffix != ".json" or not target.is_relative_to(OUTPUT_DIR.resolve()):
            raise CorpusError("--output must name a new .json file under data/m004/")
        write_new_file(target, json_bytes(report))
        report["run"]["output"] = _relative(target)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tracerag", description="TraceRAG M004 lexical retrieval pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="validate the manifest and capture (or verify) the reviewed snapshots")
    commands.add_parser("build", help="extract units and write (or reproduce) data/m004/chunks.jsonl")
    search = commands.add_parser("search", help="lexical search over the frozen chunks (offline)")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=TOP_K)
    evaluate = commands.add_parser("evaluate", help="evaluate the golden set (offline)")
    evaluate.add_argument("--output", help="create this new JSON file under data/m004/ (never overwritten)")
    return parser


COMMANDS = {"ingest": command_ingest, "build": command_build, "search": command_search, "evaluate": command_evaluate}


def _write(stream, payload) -> None:
    data = json_bytes(payload)
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write(data)
    else:
        stream.write(data.decode("utf-8"))
    stream.flush()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = COMMANDS[args.command](args)
    except (CorpusError, GoldenSetError, ValueError, OSError) as exc:
        _write(sys.stderr, {"status": "error", "command": args.command, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _write(sys.stdout, payload)
    return 1 if payload.get("status") == "differs_from_existing" else 0


if __name__ == "__main__":
    sys.exit(main())
