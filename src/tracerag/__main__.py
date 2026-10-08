"""Command line interface: ``python -m tracerag <command>``.

Run from the repository root with ``PYTHONPATH=src``. Output is JSON on stdout,
encoded as UTF-8.

M004 lexical commands: ``ingest``, ``build``, ``search`` and ``evaluate``. Only
``ingest`` uses the network; the others work offline on the local snapshots and
chunks, and none of them imports torch, transformers or numpy.

M006 semantic commands (data/m006, environment data/.venv-m006): ``freeze``,
``semantic-build``, ``semantic-search``, ``compare`` and ``observe-costs``. They work
offline on the verified local model. A process that loads the model first checks the
memory guard, then fixes the offline and one-thread settings and blocks the network,
and only then imports the ML libraries. Their files are created only under
data/m006/ and never replace an existing path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
M006_MISSION = "M006-SEMANTIC-v3"
M006_DIR = REPO_ROOT / "data" / "m006"
GOLDEN_M006 = REPO_ROOT / "evaluation" / "golden_set_m006.jsonl"
MODEL_LOCK = REPO_ROOT / "config" / "m006_model.json"
FREEZE = M006_DIR / "freeze.json"
EVALUATION_M004 = OUTPUT_DIR / "evaluation.json"


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


# ---------------------------------------------------------------- M006 semantic commands


def _m006_path(value: str, *, new: bool, suffix: str | None = None) -> Path:
    """Resolve a path that must stay inside data/m006/ (links and junctions included)."""
    path = Path(value)
    resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    root = M006_DIR.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{value} must name a path inside data/m006/")
    if suffix and resolved.suffix != suffix:
        raise ValueError(f"{value} must end with {suffix}")
    if new and (resolved.exists() or resolved.is_symlink()):
        raise ValueError(f"{value} already exists; M006 outputs are never overwritten")
    return resolved


def _begin_model_process() -> dict:
    """Memory guard before any ML import, then offline/one-thread settings and the network block."""
    from . import performance, semantic

    guard = performance.memory_guard()
    process = semantic.prepare_process()
    memory = {"process_start": performance.process_memory()}
    return {"resource_guard": guard, "process_settings": process, "memory": memory}


def _load_encoder(session: dict, model_dir: Path):
    from . import performance, semantic

    memory = session["memory"]
    memory["system_before_imports"] = performance.system_memory()
    ml = semantic.import_ml()
    memory["process_after_imports"] = performance.process_memory()
    lock = semantic.load_model_lock(MODEL_LOCK)
    memory["system_before_model_load"] = performance.system_memory()
    encoder = semantic.Encoder(ml, model_dir, lock)
    memory["system_after_model_load"] = performance.system_memory()
    memory["process_after_model_load"] = performance.process_memory()
    return ml, encoder


def _model_record(ml, encoder) -> dict:
    return {"files": encoder.verification, "load_checks": encoder.load_checks, "versions": ml.versions,
            "threads": encoder.thread_report()}


def _open_index(index_dir: Path, chunks: list[dict], ml, encoder):
    """Load an index only if it was built from the current chunks, locks and model files."""
    from . import comparison, semantic

    current = comparison.input_hashes(REPO_ROOT)
    expected = {name: current[name]["sha256"] for name in ("chunks", "model_lock", "download_lock", "requirements_lock")}
    index = semantic.load_index(index_dir, chunks, ml.np, expected)
    recorded = ((index.metadata.get("model_record") or {}).get("files") or {}).get("files")
    if recorded != encoder.verification["files"]:
        raise semantic.SemanticError("the index was built from other model files")
    return index


def _finish(session: dict) -> dict:
    """Final memory reading and the network audit; any network attempt fails the command."""
    from . import performance, semantic

    session["memory"]["process_end"] = performance.process_memory()
    session["memory"]["system_end"] = performance.system_memory()
    network = semantic.network_report()
    if network["attempts"]:
        raise semantic.NetworkBlockedError(f"network attempts during an offline command: {network['attempts']}")
    return {"resource_guard": session["resource_guard"], "process_settings": session["process_settings"],
            "memory": session["memory"], "network_guard": network,
            "peak_working_set_bytes": session["memory"]["process_end"]["peak_working_set_bytes"]}


def command_freeze(args) -> dict:
    from . import comparison

    target = _m006_path(args.output, new=True, suffix=".json")
    context = {"execution_revision": args.execution_revision, "packet_sha256": args.packet_sha256,
               "pre_sha256": args.pre_sha256, "semantic_t0": args.semantic_t0}
    return comparison.create_freeze(REPO_ROOT, target, context)


def command_semantic_build(args) -> dict:
    from . import comparison, performance, semantic

    output_dir = _m006_path(args.output_dir, new=True)
    reference = _m006_path(args.reference_index, new=False) if args.reference_index else None
    model_dir = _m006_path(args.model_dir, new=False)
    started = time.perf_counter()
    session = _begin_model_process()
    freeze = comparison.verify_freeze(REPO_ROOT, FREEZE)
    chunks = read_chunks(CHUNKS)
    ml, encoder = _load_encoder(session, model_dir)
    provenance = {"freeze": freeze, "inputs": comparison.input_hashes(REPO_ROOT),
                  "code": semantic.functional_code(REPO_ROOT), "model_record": _model_record(ml, encoder),
                  "environment": _environment()}

    def after_encoding() -> dict:
        session["memory"]["process_after_encoding"] = performance.process_memory()
        session["memory"]["system_after_encoding"] = performance.system_memory()
        return {"created_at": dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "run": {**_finish(session), "encoding_and_load_seconds": round(time.perf_counter() - started, 3)}}

    summary = semantic.build_index(chunks, encoder, output_dir, provenance, after_encoding)
    result = {"status": "written", **summary, "index_dir": _relative(output_dir)}
    if reference:
        first = semantic.load_index(reference, chunks, ml.np)
        second = semantic.load_index(output_dir, chunks, ml.np)
        check = semantic.compare_indexes(first, second, ml.np)
        check.update({"reference_index": _relative(reference), "reproduction_index": _relative(output_dir)})
        write_new_file(output_dir / "reproduction-check.json", json_bytes(check))
        result["reproduction"] = check
        if check["status"] != "reproduced_within_tolerance":
            result["status"] = "not_reproduced"
    result["duration_seconds"] = round(time.perf_counter() - started, 3)
    return result


def command_semantic_search(args) -> dict:
    from . import comparison, semantic

    index_dir = _m006_path(args.index_dir, new=False)
    model_dir = _m006_path(args.model_dir, new=False)
    session = _begin_model_process()
    comparison.verify_freeze(REPO_ROOT, FREEZE)
    chunks = read_chunks(CHUNKS)
    ml, encoder = _load_encoder(session, model_dir)
    index = _open_index(index_dir, chunks, ml, encoder)
    result = semantic.search(args.query, index, encoder, args.top_k)
    result["run"] = _finish(session)
    return result


def command_compare(args) -> dict:
    from . import comparison, semantic

    target = _m006_path(args.output, new=True, suffix=".json") if args.output else None
    reference = _m006_path(args.reference, new=False, suffix=".json") if args.reference else None
    index_dir = _m006_path(args.index_dir, new=False)
    model_dir = _m006_path(args.model_dir, new=False)
    started = time.perf_counter()
    session = _begin_model_process()
    freeze = comparison.verify_freeze(REPO_ROOT, FREEZE)
    chunks = read_chunks(CHUNKS)
    legacy = load_golden_set(GOLDEN, {chunk["chunk_id"] for chunk in chunks})
    items = comparison.load_m006_golden_set(GOLDEN_M006, chunks)
    history = comparison.reproduce_bm25_history(chunks, legacy, EVALUATION_M004, REPO_ROOT)
    if history["status"] != "identical":
        raise comparison.ComparisonError(f"the historical BM25 baseline was not reproduced: {history['differences']}")
    ml, encoder = _load_encoder(session, model_dir)
    index = _open_index(index_dir, chunks, ml, encoder)
    sets = comparison.comparison_sets(chunks, legacy, items, comparison.bm25_runner(chunks),
                                      comparison.semantic_runner(index, encoder))
    report = {
        "schema_version": comparison.REPORT_SCHEMA, "mission": M006_MISSION, "freeze": freeze,
        "inputs": comparison.input_hashes(REPO_ROOT), "code": semantic.functional_code(REPO_ROOT),
        "model": _model_record(ml, encoder), "index": {"path": _relative(index_dir), **index.reference()},
        "environment": _environment(), "parameters": comparison.parameters(),
        "golden_set_m006_summary": comparison.m006_summary(items), "bm25_history_reproduction": history,
        "sets": sets, "limitations": list(comparison.LIMITATIONS),
    }
    if reference:
        report["reproduction_check"] = comparison.compare_reports(json.loads(reference.read_bytes().decode("utf-8")), report)
        report["reproduction_check"]["reference"] = _relative(reference)
    report["run"] = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                     "duration_seconds": round(time.perf_counter() - started, 3), **_finish(session),
                     "note": "run block is a local observation, excluded from reproducibility comparisons"}
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_new_file(target, json_bytes(report))
        report["run"]["output"] = _relative(target)
    if reference and report["reproduction_check"]["status"] != "reproduced":
        report["status"] = "differs"
    return report


def command_observe_costs(args) -> dict:
    from . import comparison, performance, semantic

    if args.child:
        return _observe_child(args)
    target = _m006_path(args.output, new=True, suffix=".json") if args.output else None
    index_dir = _m006_path(args.index_dir, new=False)
    model_dir = _m006_path(args.model_dir, new=False)
    started = time.perf_counter()
    freeze = comparison.verify_freeze(REPO_ROOT, FREEZE)
    chunks = read_chunks(CHUNKS)
    items = comparison.load_m006_golden_set(GOLDEN_M006, chunks)
    env = {**os.environ, **semantic.OFFLINE_ENV, **semantic.THREAD_ENV}
    bm25 = performance.run_child("bm25", [], REPO_ROOT, env)
    guard = performance.memory_guard()
    model = performance.run_child(
        "semantic", ["--index-dir", _relative(index_dir), "--model-dir", _relative(model_dir)], REPO_ROOT, env)
    report = {
        "schema_version": 1, "mission": M006_MISSION, "method": performance.METHOD, "freeze": freeze,
        "inputs": comparison.input_hashes(REPO_ROOT), "code": semantic.functional_code(REPO_ROOT),
        "environment": {**performance.environment(), **_environment()},
        "parameters": {"queries": [item["id"] for item in items], "query_order": "golden_set_m006.jsonl order",
                       "warmup_sweeps": performance.WARMUP_SWEEPS, "measured_sweeps": performance.MEASURED_SWEEPS,
                       "top_k": TOP_K, "batch_size_queries": 1, "threads": semantic.THREAD_ENV,
                       "index_dir": _relative(index_dir), "model_dir": _relative(model_dir)},
        "children": {"bm25": bm25, "semantic": model},
        "resource_guard_before_semantic_child": guard,
        "parent": {"torch_imported": "torch" in sys.modules, "memory_end": performance.process_memory()},
        "storage": performance.storage_report(REPO_ROOT),
        "limitations": list(performance.LIMITATIONS),
        "run": {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_seconds": round(time.perf_counter() - started, 3)},
    }
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_new_file(target, json_bytes(report))
        report["run"]["output"] = _relative(target)
    return report


def _observe_child(args) -> dict:
    from . import comparison, performance, semantic

    process = semantic.prepare_process()
    chunks = read_chunks(CHUNKS)
    items = comparison.load_m006_golden_set(GOLDEN_M006, chunks)
    if args.child == "bm25":
        result = performance.bm25_child(chunks, items, LexicalIndex)
        if result["torch_imported"]:
            raise performance.PerformanceError("the BM25 child imported torch")
    else:
        index_dir = _m006_path(args.index_dir, new=False)
        model_dir = _m006_path(args.model_dir, new=False)
        lock = semantic.load_model_lock(MODEL_LOCK)
        state = {}

        def load_encoder(ml):
            state["encoder"] = semantic.Encoder(ml, model_dir, lock)
            return state["encoder"]

        def load_index(ml):
            state["np"] = ml.np
            return _open_index(index_dir, chunks, ml, state["encoder"])

        def rank(index, vector, k):
            return semantic.rank_scores(index.embeddings, index.chunk_ids, vector, k, state["np"])

        result = performance.semantic_child(items, semantic.import_ml, load_encoder, load_index, rank)
    network = semantic.network_report()
    if network["attempts"]:
        raise semantic.NetworkBlockedError(f"network attempts in the {args.child} child: {network['attempts']}")
    result.update({"process_settings": process, "network_guard": network})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tracerag", description="TraceRAG retrieval pilots (M004 lexical, M006 semantic)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="validate the manifest and capture (or verify) the reviewed snapshots")
    commands.add_parser("build", help="extract units and write (or reproduce) data/m004/chunks.jsonl")
    search = commands.add_parser("search", help="lexical search over the frozen chunks (offline)")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=TOP_K)
    evaluate = commands.add_parser("evaluate", help="evaluate the golden set (offline)")
    evaluate.add_argument("--output", help="create this new JSON file under data/m004/ (never overwritten)")
    freeze = commands.add_parser("freeze", help="M006: record frozen inputs, parameters and execution context")
    freeze.add_argument("--execution-revision", required=True)
    freeze.add_argument("--packet-sha256", required=True)
    freeze.add_argument("--pre-sha256", required=True)
    freeze.add_argument("--semantic-t0", required=True)
    freeze.add_argument("--output", default="data/m006/freeze.json")
    sbuild = commands.add_parser("semantic-build", help="M006: encode the frozen chunks into a new index directory")
    sbuild.add_argument("--output-dir", default="data/m006/index")
    sbuild.add_argument("--model-dir", default="data/m006/model")
    sbuild.add_argument("--reference-index", help="compare the new vectors with this index (reproduction)")
    ssearch = commands.add_parser("semantic-search", help="M006: semantic search over the persisted index (offline)")
    ssearch.add_argument("--query", required=True)
    ssearch.add_argument("--top-k", type=int, default=TOP_K)
    ssearch.add_argument("--index-dir", default="data/m006/index")
    ssearch.add_argument("--model-dir", default="data/m006/model")
    compare = commands.add_parser("compare", help="M006: paired BM25 x semantic evaluation of both golden sets")
    compare.add_argument("--output", help="create this new JSON file under data/m006/ (never overwritten)")
    compare.add_argument("--index-dir", default="data/m006/index")
    compare.add_argument("--model-dir", default="data/m006/model")
    compare.add_argument("--reference", help="earlier comparison JSON under data/m006/ to check the reproduction")
    costs = commands.add_parser("observe-costs", help="M006: time and memory of both backends in separate processes")
    costs.add_argument("--output", help="create this new JSON file under data/m006/ (never overwritten)")
    costs.add_argument("--index-dir", default="data/m006/index")
    costs.add_argument("--model-dir", default="data/m006/model")
    costs.add_argument("--child", choices=("bm25", "semantic"), help=argparse.SUPPRESS)
    return parser


COMMANDS = {
    "ingest": command_ingest, "build": command_build, "search": command_search, "evaluate": command_evaluate,
    "freeze": command_freeze, "semantic-build": command_semantic_build, "semantic-search": command_semantic_search,
    "compare": command_compare, "observe-costs": command_observe_costs,
}


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
    except (CorpusError, GoldenSetError, ValueError, OSError, ImportError) as exc:
        _write(sys.stderr, {"status": "error", "command": args.command, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _write(sys.stdout, payload)
    return 1 if payload.get("status") in ("differs_from_existing", "not_reproduced", "differs") else 0


if __name__ == "__main__":
    sys.exit(main())
