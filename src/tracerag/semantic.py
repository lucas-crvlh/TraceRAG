"""Local multilingual embeddings for M006: encoder, numeric index and semantic search.

Model: intfloat/multilingual-e5-small at a pinned revision, loaded only from the
verified local directory ``data/m006/model`` (safetensors weights, fast tokenizer from
tokenizer.json, no remote code, no download). Protocol, as fixed by the model card and
the M006 packet: ``"passage: "`` + original chunk text and ``"query: "`` + original
query, also in Portuguese, with no other text change; tokenization with padding and
truncation at 512 tokens; mean pooling over the tokens marked by the attention mask;
L2 normalization. The score is the dot product of two unit vectors (cosine
similarity, not multiplied by 100), ranked in descending order; only exactly equal
scores are ordered by chunk_id ascending.

A high score ranks a passage; it is not a probability that the passage supports an
answer, and no threshold is defined. torch, transformers and numpy are imported only
inside the functions that need them, so the lexical commands keep working without them.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from .corpus import json_bytes, sha256_hex, write_new_file

MISSION = "M006-SEMANTIC-v3"
MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
PREFIXES = {"query": QUERY_PREFIX, "passage": PASSAGE_PREFIX}
MAX_LENGTH = 512
DIMENSION = 384
TOP_K = 3
ABS_TOL = 1e-5
UNIT_NORM_TOL = 1e-5
INDEX_SCHEMA = 1
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "index.json"
POOLING = "attention-mask mean over all non-padding tokens"
NORMALIZATION = "L2 (vector divided by its Euclidean norm; zero or non-finite norms are errors)"
TOKENIZATION = "fast tokenizer (tokenizer.json), padding=True, truncation=True, max_length=512, batch of one text"
SCORE = ("dot product of the stored float32 unit vectors, accumulated in float64; cosine similarity, "
         "not multiplied by 100; ranked in descending order")
TIE_POLICY = "score descending, chunk_id ascending only for exactly equal scores"
SCORE_NOTE = (
    "cosine_score is the cosine similarity between the query and passage embeddings; it orders "
    "passages and is not a probability that the passage supports an answer (this model places most "
    "scores between about 0.7 and 1.0)"
)
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false",
}
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
ML_MODULES = ("numpy", "torch", "transformers", "tokenizers", "safetensors", "huggingface_hub")
_MODEL_FILES = 9
_LOCK_EXPECTED = {
    "model_id": MODEL_ID, "revision": MODEL_REVISION, "local_directory": "data/m006/model",
    "embedding_dimension": DIMENSION, "max_length_tokens": MAX_LENGTH, "query_prefix": QUERY_PREFIX,
    "passage_prefix": PASSAGE_PREFIX, "dtype": "float32", "device": "cpu", "batch_size": 1,
    "intraop_threads": 1, "interop_threads": 1, "use_fast_tokenizer": True, "use_safetensors": True,
    "trust_remote_code": False, "local_files_only": True, "attn_implementation": "eager",
}
_NETWORK_EVENTS = frozenset({
    "socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
    "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg", "urllib.Request", "http.client.connect",
})
_NETWORK = {"hook_installed": False, "enabled": False, "attempts": []}
_ML: dict = {}


class SemanticError(ValueError):
    """The model, an index, a vector or a semantic search violates the M006 contract."""


class NetworkBlockedError(RuntimeError):
    """A network operation was attempted in an offline M006 process."""


# ---------------------------------------------------------------- process setup


def _network_audit(event, _args):
    if _NETWORK["enabled"] and event in _NETWORK_EVENTS:
        _NETWORK["attempts"].append(event)
        raise NetworkBlockedError(f"network access attempted in an offline M006 process ({event})")


def block_network() -> None:
    """Install (once) an audit hook that rejects socket/HTTP operations while enabled."""
    if not _NETWORK["hook_installed"]:
        sys.addaudithook(_network_audit)
        _NETWORK["hook_installed"] = True
    _NETWORK["enabled"] = True


def release_network_block() -> None:
    """Disable the hook for the rest of the process (used by tests; the hook stays installed)."""
    _NETWORK["enabled"] = False


def network_report() -> dict:
    return {
        "mechanism": "sys.addaudithook raising NetworkBlockedError on socket and HTTP audit events",
        "events": sorted(_NETWORK_EVENTS), "enabled": _NETWORK["enabled"], "attempts": list(_NETWORK["attempts"]),
        "note": "covers Python-level networking in this process; it is not proof that the computer was isolated",
    }


def prepare_process(require_fresh: bool = True) -> dict:
    """Fix offline and one-thread settings, then block the network, before any ML import."""
    loaded = [name for name in ML_MODULES if name in sys.modules]
    if require_fresh and loaded:
        raise SemanticError(f"ML modules were imported before the offline/thread settings: {loaded}")
    for key, value in {**OFFLINE_ENV, **THREAD_ENV}.items():
        os.environ[key] = value
    block_network()
    return {"environment": {key: os.environ[key] for key in (*OFFLINE_ENV, *THREAD_ENV)},
            "ml_modules_loaded_before": loaded}


def import_ml() -> SimpleNamespace:
    """Import numpy/torch/transformers once and fix one thread, seed 0 and deterministic algorithms."""
    if "ns" in _ML:
        return _ML["ns"]
    import huggingface_hub
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if torch.get_num_interop_threads() != 1:
            raise SemanticError(f"cannot fix one inter-op thread: {exc}") from exc
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    versions = {
        "numpy": numpy.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__, "huggingface_hub": huggingface_hub.__version__,
    }
    _ML["ns"] = SimpleNamespace(np=numpy, torch=torch, transformers=transformers, versions=versions)
    return _ML["ns"]


# ---------------------------------------------------------------- model files


def load_model_lock(path: Path) -> dict:
    """Read the model lock and require the parameters fixed by the packet."""
    lock = json.loads(Path(path).read_bytes().decode("utf-8"))
    for key, value in _LOCK_EXPECTED.items():
        if lock.get(key) != value:
            raise SemanticError(f"model lock {key} is {lock.get(key)!r}, expected {value!r}")
    tolerance = lock.get("numerical_tolerance") or {}
    if (tolerance.get("absolute"), tolerance.get("relative"), tolerance.get("unit_norm_absolute")) != (ABS_TOL, 0, UNIT_NORM_TOL):
        raise SemanticError("model lock numerical_tolerance differs from atol=1e-5, rtol=0, unit norm 1e-5")
    files = lock.get("files")
    if not isinstance(files, list) or len(files) != _MODEL_FILES or len({f.get("path") for f in files}) != _MODEL_FILES:
        raise SemanticError(f"model lock must list {_MODEL_FILES} distinct files")
    return lock


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_dir(model_dir: Path, lock: dict) -> dict:
    """Require exactly the locked files, with their sizes and hashes, and no custom code."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise SemanticError(f"model directory {model_dir} does not exist")
    present = sorted(
        path.relative_to(model_dir).as_posix() for path in model_dir.rglob("*") if path.is_file() or path.is_symlink()
    )
    expected = sorted(entry["path"] for entry in lock["files"])
    if present != expected:
        raise SemanticError(f"model directory files {present} differ from the locked files {expected}")
    files = []
    for entry in sorted(lock["files"], key=lambda e: e["path"]):
        path = model_dir / entry["path"]
        if path.is_symlink():
            raise SemanticError(f"{entry['path']} is a link; only regular files are accepted")
        size, digest = path.stat().st_size, _sha256_file(path)
        if (size, digest) != (entry["size_bytes"], entry["sha256"]):
            raise SemanticError(f"{entry['path']}: size or SHA-256 differs from the model lock")
        files.append({"path": entry["path"], "bytes": size, "sha256": digest})
    config = json.loads((model_dir / "config.json").read_bytes().decode("utf-8"))
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_bytes().decode("utf-8"))
    if "auto_map" in config or "auto_map" in tokenizer_config:
        raise SemanticError("auto_map found: custom model or tokenizer code is not accepted")
    checks = {"model_type": "bert", "architectures": ["BertModel"], "hidden_size": DIMENSION,
              "num_hidden_layers": 12, "max_position_embeddings": MAX_LENGTH, "torch_dtype": "float32"}
    for key, value in checks.items():
        if config.get(key) != value:
            raise SemanticError(f"config.json {key} is {config.get(key)!r}, expected {value!r}")
    pooling = json.loads((model_dir / "1_Pooling" / "config.json").read_bytes().decode("utf-8"))
    if not pooling.get("pooling_mode_mean_tokens") or pooling.get("word_embedding_dimension") != DIMENSION:
        raise SemanticError("1_Pooling/config.json does not describe mean pooling of 384 dimensions")
    return {"model_dir": model_dir.as_posix(), "files": files, "config_checks": checks,
            "custom_code": "none (no .py file, no auto_map)", "weights": "model.safetensors only"}


# ---------------------------------------------------------------- vectors


def mean_pool(last_hidden_state, attention_mask):
    """Average the token states marked by the attention mask (padding excluded), as in the model card."""
    if last_hidden_state.dim() != 3 or tuple(attention_mask.shape) != tuple(last_hidden_state.shape[:2]):
        raise SemanticError("expected hidden states [batch, tokens, dim] and a mask [batch, tokens]")
    counts = attention_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise SemanticError("a sequence has no non-padding token")
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / counts[..., None]


def unit_vectors(pooled):
    """L2-normalize each row; zero or non-finite norms are errors, never silently fixed."""
    torch = import_ml().torch
    norms = pooled.norm(p=2, dim=-1, keepdim=True)
    if not bool(torch.isfinite(pooled).all()) or not bool(torch.isfinite(norms).all()):
        raise SemanticError("embedding has non-finite values")
    if bool((norms == 0).any()):
        raise SemanticError("embedding has zero norm")
    return pooled / norms


def check_vectors(matrix, np, dimension: int = DIMENSION) -> list[float]:
    """Validate a float32 matrix of unit vectors and return the float64 norms."""
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] < 1:
        raise SemanticError(f"expected a non-empty matrix with {dimension} columns, got {getattr(matrix, 'shape', None)}")
    if matrix.dtype != np.float32:
        raise SemanticError(f"expected float32 vectors, got {matrix.dtype}")
    if not bool(np.isfinite(matrix).all()):
        raise SemanticError("vectors contain NaN or Inf")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if bool((norms == 0).any()):
        raise SemanticError("a vector has zero norm")
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > UNIT_NORM_TOL:
        raise SemanticError(f"a vector is not unit length (|norm - 1| = {worst:.3g} > {UNIT_NORM_TOL})")
    return [float(value) for value in norms]


def rank_scores(embeddings, chunk_ids: list[str], query_vector, top_k: int, np) -> list[tuple]:
    """Cosine scores of all chunks, best first; exactly equal scores ordered by chunk_id."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise SemanticError("top_k must be a positive integer")
    query = np.asarray(query_vector)
    if query.shape != (embeddings.shape[1],):
        raise SemanticError(f"query vector has shape {query.shape}, expected ({embeddings.shape[1]},)")
    scores = embeddings.astype(np.float64) @ query.astype(np.float64)
    values = [float(value) for value in scores]
    order = sorted(range(len(chunk_ids)), key=lambda i: (-values[i], chunk_ids[i]))
    return [(rank, chunk_ids[i], values[i]) for rank, i in enumerate(order[:top_k], start=1)]


# ---------------------------------------------------------------- encoder


class Encoder:
    """Tokenizer and BertModel loaded from the verified local directory, CPU float32, eval mode."""

    def __init__(self, ml, model_dir: Path, lock: dict) -> None:
        self.ml = ml
        self.np, self.torch = ml.np, ml.torch
        self.verification = verify_model_dir(model_dir, lock)
        transformers = ml.transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_dir), use_fast=True, local_files_only=True, trust_remote_code=False,
        )
        self.model = transformers.AutoModel.from_pretrained(
            str(model_dir), use_safetensors=True, local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", dtype=ml.torch.float32,
        )
        self.model.eval()
        self.load_checks = self._check_loaded()

    def _check_loaded(self) -> dict:
        model, torch = self.model, self.torch
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        devices = sorted({p.device.type for p in model.parameters()})
        checks = {
            "tokenizer_class": type(self.tokenizer).__name__, "tokenizer_is_fast": bool(self.tokenizer.is_fast),
            "model_class": type(model).__name__, "parameter_dtypes": dtypes, "parameter_devices": devices,
            "training_mode": bool(model.training), "attn_implementation": model.config._attn_implementation,
            "hidden_size": model.config.hidden_size, "num_hidden_layers": model.config.num_hidden_layers,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
        expected = {"tokenizer_is_fast": True, "model_class": "BertModel", "parameter_dtypes": [str(torch.float32)],
                    "parameter_devices": ["cpu"], "training_mode": False, "attn_implementation": "eager",
                    "hidden_size": DIMENSION, "num_hidden_layers": 12}
        for key, value in expected.items():
            if checks[key] != value:
                raise SemanticError(f"loaded model check {key}: {checks[key]!r}, expected {value!r}")
        return checks

    def thread_report(self) -> dict:
        torch = self.torch
        return {"torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "environment": {key: os.environ.get(key) for key in (*THREAD_ENV, "TOKENIZERS_PARALLELISM")}}

    def encode(self, text: str, kind: str):
        """Return the float32 unit vector of one text and its token provenance."""
        if kind not in PREFIXES:
            raise SemanticError(f"kind must be one of {sorted(PREFIXES)}")
        if not isinstance(text, str) or not text.strip():
            raise SemanticError("text to encode must be a non-empty string")
        model_input = PREFIXES[kind] + text
        before = len(self.tokenizer(model_input, truncation=False)["input_ids"])
        batch = self.tokenizer([model_input], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        after = int(batch["attention_mask"].sum())
        with self.torch.inference_mode():
            output = self.model(**batch)
            vector = unit_vectors(mean_pool(output.last_hidden_state, batch["attention_mask"]))[0]
            array = vector.numpy().astype(self.np.float32, copy=True)
        check_vectors(array.reshape(1, -1), self.np)
        info = {"prefix": PREFIXES[kind], "text_sha256": sha256_hex(text.encode("utf-8")),
                "tokens_before_truncation": before, "tokens_after_truncation": after, "truncated": before > after}
        return array, info


# ---------------------------------------------------------------- index


class SemanticIndex:
    """Validated embeddings, their metadata and the chunks they encode."""

    def __init__(self, chunk_ids, embeddings, metadata, chunks_by_id, index_json_sha256: str) -> None:
        self.chunk_ids = chunk_ids
        self.embeddings = embeddings
        self.metadata = metadata
        self.chunks_by_id = chunks_by_id
        self.index_json_sha256 = index_json_sha256
        self.passages = {row["chunk_id"]: row for row in metadata["passages"]}

    def reference(self) -> dict:
        return {"index_json_sha256": self.index_json_sha256,
                "embeddings_sha256": self.metadata["embeddings"]["sha256"],
                "model": f"{MODEL_ID}@{MODEL_REVISION}"}


def build_index(chunks: list[dict], encoder: Encoder, output_dir: Path, provenance: dict, after_encoding=None) -> dict:
    """Encode every chunk (batch of one, chunk_id order) and create a new index directory.

    ``after_encoding`` may return extra metadata observed once all vectors exist
    (for example memory and timing); it is merged before anything is written.
    """
    np = encoder.np
    ordered = sorted(chunks, key=lambda chunk: chunk["chunk_id"])
    ids = [chunk["chunk_id"] for chunk in ordered]
    if len(set(ids)) != len(ids):
        raise SemanticError("chunk_id values must be unique")
    vectors, rows = [], []
    for chunk in ordered:
        if sha256_hex(chunk["text"].encode("utf-8")) != chunk["text_sha256"]:
            raise SemanticError(f"{chunk['chunk_id']}: text does not match text_sha256")
        vector, info = encoder.encode(chunk["text"], "passage")
        vectors.append(vector)
        rows.append({"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                     "unit_locator": chunk["unit_locator"], "part": chunk["part"],
                     "word_count": chunk["word_count"], **info})
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    norms = check_vectors(matrix, np)
    for row, norm in zip(rows, norms):
        row["vector_norm"] = norm
    extra = after_encoding() if after_encoding else {}
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    data = buffer.getvalue()
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    write_new_file(output_dir / EMBEDDINGS_FILE, data)
    metadata = {
        "schema_version": INDEX_SCHEMA, "mission": MISSION, "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "embeddings": {"file": EMBEDDINGS_FILE, "bytes": len(data), "sha256": sha256_hex(data),
                       "shape": list(matrix.shape), "dtype": "float32", "format": "numpy .npy, allow_pickle=False"},
        "chunk_ids": ids,
        "protocol": {"passage_prefix": PASSAGE_PREFIX, "query_prefix": QUERY_PREFIX, "tokenization": TOKENIZATION,
                     "max_length": MAX_LENGTH, "pooling": POOLING, "normalization": NORMALIZATION,
                     "batch_size": 1, "order": "chunk_id ascending", "score": SCORE, "tie_policy": TIE_POLICY},
        "passages": rows,
        "truncated_passages": [row["chunk_id"] for row in rows if row["truncated"]],
        **provenance,
        **extra,
    }
    meta_bytes = json_bytes(metadata)
    write_new_file(output_dir / METADATA_FILE, meta_bytes)
    return {"index_dir": output_dir.as_posix(), "index_json_sha256": sha256_hex(meta_bytes),
            "embeddings_sha256": metadata["embeddings"]["sha256"], "embeddings_bytes": len(data),
            "shape": list(matrix.shape), "truncated_passages": metadata["truncated_passages"], "passages": rows}


def load_index(index_dir: Path, chunks: list[dict], np, expected_inputs: dict | None = None) -> SemanticIndex:
    """Read and validate an index directory against the frozen chunks before any query."""
    index_dir = Path(index_dir)
    meta_bytes = (index_dir / METADATA_FILE).read_bytes()
    metadata = json.loads(meta_bytes.decode("utf-8"))
    if metadata.get("schema_version") != INDEX_SCHEMA or metadata.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise SemanticError("index metadata has another schema or model")
    protocol = metadata.get("protocol") or {}
    for key, value in (("passage_prefix", PASSAGE_PREFIX), ("query_prefix", QUERY_PREFIX), ("max_length", MAX_LENGTH),
                       ("pooling", POOLING), ("batch_size", 1)):
        if protocol.get(key) != value:
            raise SemanticError(f"index protocol {key} is {protocol.get(key)!r}, expected {value!r}")
    data = (index_dir / EMBEDDINGS_FILE).read_bytes()
    stored = metadata.get("embeddings") or {}
    if len(data) != stored.get("bytes") or sha256_hex(data) != stored.get("sha256"):
        raise SemanticError("embeddings.npy bytes do not match the index metadata")
    matrix = np.load(io.BytesIO(data), allow_pickle=False)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ids = metadata.get("chunk_ids")
    expected_ids = sorted(by_id)
    if ids != expected_ids:
        unknown = sorted(set(ids or []) - set(by_id))
        missing = sorted(set(by_id) - set(ids or []))
        raise SemanticError(f"index chunk ids differ from the frozen chunks (unknown {unknown}, missing {missing}, or order)")
    if tuple(matrix.shape) != (len(ids), DIMENSION) or list(matrix.shape) != stored.get("shape"):
        raise SemanticError(f"embeddings shape {tuple(matrix.shape)} differs from ({len(ids)}, {DIMENSION})")
    check_vectors(matrix, np)
    passages = {row["chunk_id"]: row for row in metadata.get("passages") or []}
    for chunk_id in ids:
        if passages.get(chunk_id, {}).get("text_sha256") != by_id[chunk_id]["text_sha256"]:
            raise SemanticError(f"{chunk_id}: index was built from another text")
    for key, value in (expected_inputs or {}).items():
        found = (metadata.get("inputs") or {}).get(key, {}).get("sha256")
        if found != value:
            raise SemanticError(f"index input {key} has SHA-256 {found}, expected {value}")
    return SemanticIndex(ids, matrix, metadata, by_id, sha256_hex(meta_bytes))


def compare_indexes(first: SemanticIndex, second: SemanticIndex, np) -> dict:
    """Reproduction check: same ids, shape and finite unit vectors within atol=1e-5, rtol=0."""
    same_ids = first.chunk_ids == second.chunk_ids
    same_shape = first.embeddings.shape == second.embeddings.shape
    difference = (float(np.max(np.abs(first.embeddings.astype(np.float64) - second.embeddings.astype(np.float64))))
                  if same_ids and same_shape else None)
    close = bool(same_ids and same_shape and np.allclose(first.embeddings, second.embeddings, rtol=0, atol=ABS_TOL))
    return {"same_chunk_ids": same_ids, "same_shape": same_shape, "max_abs_difference": difference,
            "allclose_rtol0_atol1e-5": close, "bytes_identical": first.metadata["embeddings"]["sha256"]
            == second.metadata["embeddings"]["sha256"], "first_embeddings_sha256": first.metadata["embeddings"]["sha256"],
            "second_embeddings_sha256": second.metadata["embeddings"]["sha256"],
            "status": "reproduced_within_tolerance" if close else "not_reproduced"}


# ---------------------------------------------------------------- search


_HIT_FIELDS = (
    "document_id", "publication_id", "title", "section", "unit_locator", "part", "source_url", "source_version",
    "source_sha256", "snapshot_path", "retrieved_at", "text_sha256", "text", "attribution",
)


def search(query: str, index: SemanticIndex, encoder: Encoder, top_k: int = TOP_K) -> dict:
    """Return up to ``top_k`` passages by cosine score, with full text and provenance."""
    vector, info = encoder.encode(query, "query")
    hits = []
    for rank, chunk_id, score in rank_scores(index.embeddings, index.chunk_ids, vector, top_k, encoder.np):
        chunk = index.chunks_by_id[chunk_id]
        passage = index.passages[chunk_id]
        hit = {"rank": rank, "cosine_score": score, "chunk_id": chunk_id}
        hit.update({key: chunk.get(key) for key in _HIT_FIELDS})
        hit["encoder_tokens"] = {key: passage[key] for key in ("tokens_before_truncation", "tokens_after_truncation", "truncated")}
        hits.append(hit)
    return {"query": query, "query_tokens": info, "top_k": top_k, "hits": hits, "index": index.reference(),
            "score_note": SCORE_NOTE,
            "text_note": "text is the full chunk returned to the reader; encoder_tokens tells how many of its "
                         "tokens the encoder actually saw"}


# ---------------------------------------------------------------- provenance helpers


def file_info(path: Path, repo_root: Path) -> dict:
    data = Path(path).read_bytes()
    return {"path": Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix(),
            "bytes": len(data), "sha256": sha256_hex(data)}


def functional_code(repo_root: Path) -> list[dict]:
    """Hashes of the functional modules (src/tracerag/*.py) that produce indexes and measurements."""
    return [file_info(path, repo_root) for path in sorted((Path(repo_root) / "src" / "tracerag").glob("*.py"))]
