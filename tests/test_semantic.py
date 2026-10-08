"""Offline tests for the M006 semantic retrieval comparison.

Synthetic tests use invented texts and vectors (no NIST text), need no model and run
no search over the corpus. The real tests run only with TRACERAG_M006_REAL=1, after
the freeze: they load the verified local model with the network blocked, rebuild the
index into a new directory, repeat queries and evaluation and compare them with the
first measurement within atol=1e-5 (rtol=0). Test files live under data/m006/tests
(or TRACERAG_M006_TEST_ROOT, which must stay inside data/m006), never under
data/m004/test-temp.
"""

from __future__ import annotations

import datetime as dt
import email.message
import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tracerag import comparison, evaluation, performance, semantic
from tracerag import __main__ as cli

REPO_ROOT = Path(__file__).resolve().parents[1]
M006_ROOT = REPO_ROOT / "data" / "m006"
_ROOT = Path(os.environ.get("TRACERAG_M006_TEST_ROOT") or M006_ROOT / "tests")
TEST_ROOT = _ROOT if _ROOT.is_absolute() else REPO_ROOT / _ROOT
REAL = os.environ.get("TRACERAG_M006_REAL") == "1"
HAS_ML = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "transformers"))
CHUNKS = REPO_ROOT / "data" / "m004" / "chunks.jsonl"
GOLDEN_M006 = REPO_ROOT / "evaluation" / "golden_set_m006.jsonl"
MODEL_LOCK = REPO_ROOT / "config" / "m006_model.json"
DOWNLOAD_LOCK = REPO_ROOT / "config" / "m006_downloads.json"
FIRST_INDEX = M006_ROOT / "index"
FIRST_COMPARISON = M006_ROOT / "results" / "comparison.json"
MODEL_DIR = M006_ROOT / "model"


def setUpModule():
    root = TEST_ROOT.resolve()
    if root == M006_ROOT.resolve() or not root.is_relative_to(M006_ROOT.resolve()):
        raise RuntimeError("TRACERAG_M006_TEST_ROOT must name a directory inside data/m006")
    (TEST_ROOT / "tmp").mkdir(parents=True, exist_ok=True)


def temp_dir():
    return tempfile.TemporaryDirectory(dir=TEST_ROOT / "tmp")


def load_acquire():
    spec = importlib.util.spec_from_file_location("acquire_m006", REPO_ROOT / "tools" / "acquire_m006.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "document_id": "doc", "publication_id": "pub", "title": "Synthetic",
            "section": "7", "unit_locator": chunk_id.split(":")[-1], "part": 1, "source_url": "https://example.org/s",
            "source_version": "snapshot:" + "0" * 64, "source_sha256": "0" * 64, "snapshot_path": "data/x.html",
            "retrieved_at": "2026-10-08T10:00:00-03:00", "text": text, "word_count": len(text.split()),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "attribution": "Synthetic."}


SYNTHETIC = [synthetic_chunk("doc:7.1", "kestrel lantern measures granite"),
             synthetic_chunk("doc:7.2", "falcon marigold heron"),
             synthetic_chunk("doc:7.3", "osprey pelican harrier")]


class FakeEncoder:
    """Deterministic synthetic unit vectors (no model)."""

    def __init__(self, np) -> None:
        self.np = np

    def encode(self, text, kind):
        np = self.np
        seed = int.from_bytes(hashlib.sha256((kind + text).encode("utf-8")).digest()[:8], "little")
        vector = np.random.default_rng(seed).standard_normal(semantic.DIMENSION)
        vector = (vector / np.linalg.norm(vector)).astype(np.float32)
        info = {"prefix": semantic.PREFIXES[kind], "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "tokens_before_truncation": len(text.split()) + 2, "tokens_after_truncation": len(text.split()) + 2,
                "truncated": False}
        return vector, info


# ---------------------------------------------------------------- vectors and pooling


@unittest.skipUnless(HAS_ML, "needs numpy and torch")
class PoolingTests(unittest.TestCase):
    def setUp(self):
        self.torch = semantic.import_ml().torch

    def test_mean_pool_ignores_padding_tokens(self):
        torch = self.torch
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [1e6, -1e6]], [[5.0, 7.0], [9.0, 1.0], [2.0, 4.0]]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        pooled = semantic.mean_pool(hidden, mask)
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 3.0], [16.0 / 3, 4.0]])))

    def test_all_padding_and_shape_errors(self):
        torch = self.torch
        with self.assertRaisesRegex(semantic.SemanticError, "no non-padding"):
            semantic.mean_pool(torch.ones(1, 2, 3), torch.zeros(1, 2, dtype=torch.long))
        with self.assertRaises(semantic.SemanticError):
            semantic.mean_pool(torch.ones(1, 2, 3), torch.ones(1, 3, dtype=torch.long))

    def test_unit_vectors_and_invalid_norms(self):
        torch = self.torch
        unit = semantic.unit_vectors(torch.tensor([[3.0, 4.0]]))
        self.assertTrue(torch.allclose(unit, torch.tensor([[0.6, 0.8]])))
        with self.assertRaisesRegex(semantic.SemanticError, "zero norm"):
            semantic.unit_vectors(torch.zeros(1, 4))
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad), self.assertRaisesRegex(semantic.SemanticError, "non-finite"):
                semantic.unit_vectors(torch.tensor([[1.0, bad]]))


@unittest.skipUnless(HAS_ML, "needs numpy")
class VectorAndRankingTests(unittest.TestCase):
    def setUp(self):
        self.np = semantic.import_ml().np

    def unit_rows(self, rows):
        np = self.np
        matrix = np.zeros((len(rows), semantic.DIMENSION), dtype=np.float32)
        for i, row in enumerate(rows):
            matrix[i, : len(row)] = row
        return matrix

    def test_check_vectors_rejects_bad_matrices(self):
        np = self.np
        good = self.unit_rows([[1.0], [0.0, 1.0]])
        self.assertEqual(len(semantic.check_vectors(good, np)), 2)
        bad = {
            "dimension": np.ones((2, 10), dtype=np.float32) / np.sqrt(10),
            "dtype": good.astype(np.float64),
            "nan": np.where(good == 1.0, np.nan, good).astype(np.float32),
            "inf": np.where(good == 1.0, np.inf, good).astype(np.float32),
            "zero": np.zeros_like(good),
            "not unit": good * 2,
        }
        for name, matrix in bad.items():
            with self.subTest(case=name), self.assertRaises(semantic.SemanticError):
                semantic.check_vectors(matrix, np)

    def test_ranking_descends_and_breaks_exact_ties_by_chunk_id(self):
        np = self.np
        embeddings = self.unit_rows([[0.6, 0.8], [1.0], [1.0], [0.0, 1.0]])
        ids = ["d", "c", "a", "b"]
        query = self.unit_rows([[1.0]])[0]
        ranked = semantic.rank_scores(embeddings, ids, query, 3, np)
        self.assertEqual([(rank, cid) for rank, cid, _s in ranked], [(1, "a"), (2, "c"), (3, "d")])
        self.assertEqual(ranked[0][2], ranked[1][2])
        self.assertGreater(ranked[1][2], ranked[2][2])
        for value in (0, -1, True, "3"):
            with self.subTest(top_k=value), self.assertRaises(semantic.SemanticError):
                semantic.rank_scores(embeddings, ids, query, value, np)
        with self.assertRaises(semantic.SemanticError):
            semantic.rank_scores(embeddings, ids, query[:10], 3, np)


@unittest.skipUnless(HAS_ML, "needs numpy")
class IndexTests(unittest.TestCase):
    def setUp(self):
        self.np = semantic.import_ml().np
        self.encoder = FakeEncoder(self.np)
        self.tmp = temp_dir()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, name="index", chunks=SYNTHETIC):
        return semantic.build_index(list(reversed(chunks)), self.encoder, self.root / name,
                                    {"inputs": {"chunks": {"sha256": "c" * 64}}}, lambda: {"created_at": "now"})

    def test_build_orders_by_chunk_id_and_reloads(self):
        summary = self.build()
        self.assertEqual(summary["shape"], [3, semantic.DIMENSION])
        index = semantic.load_index(self.root / "index", SYNTHETIC, self.np, {"chunks": "c" * 64})
        self.assertEqual(index.chunk_ids, ["doc:7.1", "doc:7.2", "doc:7.3"])
        self.assertEqual(index.metadata["created_at"], "now")
        self.assertEqual(index.metadata["protocol"]["passage_prefix"], "passage: ")
        with self.assertRaises(FileExistsError):
            self.build()

    def test_tampered_or_mismatched_indexes_are_rejected(self):
        self.build()
        directory = self.root / "index"
        with self.assertRaisesRegex(semantic.SemanticError, "expected"):
            semantic.load_index(directory, SYNTHETIC, self.np, {"chunks": "d" * 64})
        with self.assertRaisesRegex(semantic.SemanticError, "chunk ids"):
            semantic.load_index(directory, SYNTHETIC[:2], self.np)
        extra = [*SYNTHETIC, synthetic_chunk("doc:7.4", "stranger text")]
        with self.assertRaisesRegex(semantic.SemanticError, "chunk ids"):
            semantic.load_index(directory, extra, self.np)
        changed = [dict(SYNTHETIC[0], text="other", text_sha256=hashlib.sha256(b"other").hexdigest()), *SYNTHETIC[1:]]
        with self.assertRaisesRegex(semantic.SemanticError, "another text"):
            semantic.load_index(directory, changed, self.np)
        data = (directory / semantic.EMBEDDINGS_FILE).read_bytes()
        (directory / semantic.EMBEDDINGS_FILE).write_bytes(data[:-4] + b"\x00\x00\x80\x3f")
        with self.assertRaisesRegex(semantic.SemanticError, "do not match"):
            semantic.load_index(directory, SYNTHETIC, self.np)

    def write_matrix(self, name, matrix):
        """Index whose metadata hash matches a deliberately invalid matrix."""
        np = self.np
        self.build(name)
        directory = self.root / name
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        data = buffer.getvalue()
        (directory / semantic.EMBEDDINGS_FILE).write_bytes(data)
        metadata = json.loads((directory / semantic.METADATA_FILE).read_text(encoding="utf-8"))
        metadata["embeddings"].update({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                       "shape": list(matrix.shape)})
        (directory / semantic.METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
        return directory

    def test_invalid_vectors_with_matching_hash_are_rejected(self):
        np = self.np
        self.build("base")
        valid = semantic.load_index(self.root / "base", SYNTHETIC, np).embeddings
        cases = {"nan": valid.copy(), "norm": valid * np.float32(1.01), "shape": valid[:, :100].copy(),
                 "rows": valid[:2].copy()}
        cases["nan"][0, 0] = np.nan
        for name, matrix in cases.items():
            with self.subTest(case=name), self.assertRaises(semantic.SemanticError):
                semantic.load_index(self.write_matrix(name, matrix), SYNTHETIC, np)

    def test_reproduction_tolerance(self):
        np = self.np
        self.build("a")
        first = semantic.load_index(self.root / "a", SYNTHETIC, np)
        second = semantic.load_index(self.root / "a", SYNTHETIC, np)
        self.assertEqual(semantic.compare_indexes(first, second, np)["status"], "reproduced_within_tolerance")
        shifted = SimpleNamespace(**vars(second))
        shifted.embeddings = second.embeddings + np.float32(2e-5)
        shifted.metadata = second.metadata
        self.assertEqual(semantic.compare_indexes(first, shifted, np)["status"], "not_reproduced")

    def test_search_returns_provenance_without_labels(self):
        self.build()
        index = semantic.load_index(self.root / "index", SYNTHETIC, self.np)
        result = semantic.search("kestrel lantern", index, self.encoder, 2)
        self.assertEqual([hit["rank"] for hit in result["hits"]], [1, 2])
        for hit in result["hits"]:
            self.assertNotIn("relevant", hit)
            self.assertEqual(hit["text"], index.chunks_by_id[hit["chunk_id"]]["text"])
            self.assertIn("encoder_tokens", hit)
        self.assertIn("not a probability", result["score_note"])


# ---------------------------------------------------------------- model files and loader


class ModelFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = temp_dir()
        self.dir = Path(self.tmp.name) / "model"
        (self.dir / "1_Pooling").mkdir(parents=True)
        config = {"model_type": "bert", "architectures": ["BertModel"], "hidden_size": 384, "num_hidden_layers": 12,
                  "max_position_embeddings": 512, "torch_dtype": "float32"}
        files = {"config.json": json.dumps(config), "tokenizer_config.json": json.dumps({"model_max_length": 512}),
                 "1_Pooling/config.json": json.dumps({"pooling_mode_mean_tokens": True, "word_embedding_dimension": 384}),
                 "model.safetensors": "synthetic weights"}
        self.lock = {"files": []}
        for name, text in files.items():
            (self.dir / name).write_bytes(text.encode("utf-8"))
            self.lock["files"].append({"path": name, "size_bytes": len(text.encode("utf-8")),
                                       "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_files_pass(self):
        report = semantic.verify_model_dir(self.dir, self.lock)
        self.assertEqual([f["path"] for f in report["files"]], sorted(f["path"] for f in self.lock["files"]))

    def test_extra_code_or_weights_and_wrong_bytes_are_rejected(self):
        for extra in ("modeling_custom.py", "pytorch_model.bin"):
            with self.subTest(extra=extra):
                (self.dir / extra).write_bytes(b"x")
                with self.assertRaisesRegex(semantic.SemanticError, "differ from the locked files"):
                    semantic.verify_model_dir(self.dir, self.lock)
                (self.dir / extra).unlink()
        (self.dir / "model.safetensors").write_bytes(b"synthetic weightz")
        with self.assertRaisesRegex(semantic.SemanticError, "size or SHA-256"):
            semantic.verify_model_dir(self.dir, self.lock)

    def test_auto_map_is_rejected(self):
        text = json.dumps({"model_type": "bert", "auto_map": {"AutoModel": "custom.Model"}})
        (self.dir / "config.json").write_bytes(text.encode("utf-8"))
        entry = next(f for f in self.lock["files"] if f["path"] == "config.json")
        entry.update({"size_bytes": len(text.encode("utf-8")), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
        with self.assertRaisesRegex(semantic.SemanticError, "auto_map"):
            semantic.verify_model_dir(self.dir, self.lock)

    def test_model_lock_parameters_are_enforced(self):
        lock = semantic.load_model_lock(MODEL_LOCK)
        self.assertEqual((lock["model_id"], lock["revision"]), (semantic.MODEL_ID, semantic.MODEL_REVISION))
        original = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))
        for key, value in (("trust_remote_code", True), ("batch_size", 4), ("local_files_only", False),
                           ("revision", "main"), ("query_prefix", "")):
            with self.subTest(key=key):
                path = Path(self.tmp.name) / f"lock-{key}.json"
                path.write_text(json.dumps({**original, key: value}), encoding="utf-8")
                with self.assertRaises(semantic.SemanticError):
                    semantic.load_model_lock(path)

    def test_encoder_uses_local_safetensors_without_remote_code(self):
        tokenizer, model = mock.Mock(), mock.Mock()
        transformers = SimpleNamespace(AutoTokenizer=mock.Mock(), AutoModel=mock.Mock())
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        transformers.AutoModel.from_pretrained.return_value = model
        ml = SimpleNamespace(np=None, torch=SimpleNamespace(float32="float32"), transformers=transformers)
        with mock.patch.object(semantic, "verify_model_dir", return_value={"files": []}), \
                mock.patch.object(semantic.Encoder, "_check_loaded", return_value={}):
            semantic.Encoder(ml, self.dir, {"files": []})
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            str(self.dir), use_fast=True, local_files_only=True, trust_remote_code=False)
        transformers.AutoModel.from_pretrained.assert_called_once_with(
            str(self.dir), use_safetensors=True, local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", dtype="float32")
        model.eval.assert_called_once_with()


# ---------------------------------------------------------------- acquisition


class FakeResponse:
    def __init__(self, body: bytes, url: str, status: int = 200, headers: dict | None = None) -> None:
        self._data = io.BytesIO(body)
        self.status = status
        self._url = url
        self.headers = email.message.Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self, size=-1):
        return self._data.read(size)

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def fake_factory(actions: list):
    calls = []

    def factory(_hosts, _chain):
        class Opener:
            def open(self, request, timeout=None):
                calls.append(request.full_url)
                action = actions[len(calls) - 1]
                if isinstance(action, BaseException):
                    raise action
                return action
        return Opener()

    factory.calls = calls
    return factory


class AcquisitionTests(unittest.TestCase):
    HOSTS = ("files.pythonhosted.org", "huggingface.co")

    def setUp(self):
        self.acq = load_acquire()
        self.tmp = temp_dir()
        self.out = Path(self.tmp.name) / "out"
        self.out.mkdir()
        self.body = b"synthetic wheel bytes"
        self.url = "https://files.pythonhosted.org/packages/x/demo-1.0-py3-none-any.whl"
        self.item = {"kind": "wheel", "name": "demo-1.0-py3-none-any.whl", "relative": ["wheels", "demo-1.0-py3-none-any.whl"],
                     "url": self.url, "size": len(self.body), "sha256": hashlib.sha256(self.body).hexdigest()}

    def tearDown(self):
        self.tmp.cleanup()

    def final(self):
        return self.out / "wheels" / "demo-1.0-py3-none-any.whl"

    def test_url_rules(self):
        self.assertEqual(self.acq.check_url(self.url, self.HOSTS), "files.pythonhosted.org")
        for url in ("http://files.pythonhosted.org/x", "https://evil.example/x", "https://user:pw@huggingface.co/x",
                    "https://huggingface.co:8443/x", "ftp://huggingface.co/x"):
            with self.subTest(url=url), self.assertRaises(self.acq.AcquisitionError):
                self.acq.check_url(url, self.HOSTS)

    def test_redirect_outside_allowlist_is_rejected(self):
        chain = []
        handler = self.acq.AllowlistRedirectHandler(self.HOSTS, chain)
        request = urllib.request.Request("https://huggingface.co/a")
        with self.assertRaises(self.acq.AcquisitionError):
            handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.evil.example/a?sig=1")
        self.assertEqual(chain, [{"status": 302, "to_host": "cdn.evil.example"}])
        allowed = handler.redirect_request(request, None, 307, "Temporary", {}, "https://huggingface.co/b")
        self.assertEqual(allowed.full_url, "https://huggingface.co/b")

    def test_valid_download_is_promoted_once(self):
        factory = fake_factory([FakeResponse(self.body, self.url, headers={"Content-Length": str(len(self.body))})])
        record = self.acq.acquire_item(self.item, self.out, self.HOSTS, factory)
        self.assertEqual(record["result"], "downloaded_verified")
        self.assertEqual(self.final().read_bytes(), self.body)
        again = self.acq.acquire_item(self.item, self.out, self.HOSTS, fake_factory([]))
        self.assertEqual(again["result"], "present_verified")

    def test_existing_different_final_is_never_replaced(self):
        self.final().parent.mkdir()
        self.final().write_bytes(b"other bytes")
        with self.assertRaisesRegex(self.acq.AcquisitionError, "not replaced"):
            self.acq.acquire_item(self.item, self.out, self.HOSTS, fake_factory([]))
        self.assertEqual(self.final().read_bytes(), b"other bytes")

    def test_wrong_bytes_are_not_promoted_or_retried(self):
        factory = fake_factory([FakeResponse(b"synthetic wheel bytez", self.url)])
        with self.assertRaisesRegex(self.acq.AcquisitionError, "no retry"):
            self.acq.acquire_item(self.item, self.out, self.HOSTS, factory)
        self.assertEqual(len(factory.calls), 1)
        self.assertFalse(self.final().exists())
        self.assertEqual(len(list((self.out / "tmp").iterdir())), 1)
        too_big = fake_factory([FakeResponse(self.body + b"!", self.url)])
        with self.assertRaisesRegex(self.acq.AcquisitionError, "larger"):
            self.acq.acquire_item(self.item, self.out, self.HOSTS, too_big)
        self.assertEqual(len(too_big.calls), 1)

    def test_one_retry_only_after_transport_failure(self):
        reset = urllib.error.URLError(ConnectionResetError("reset"))
        factory = fake_factory([reset, FakeResponse(self.body, self.url)])
        record = self.acq.acquire_item(self.item, self.out, self.HOSTS, factory)
        self.assertEqual([a["outcome"] for a in record["attempts"]], ["transport_failure", "promoted"])
        twice = fake_factory([reset, reset])
        other = dict(self.item, relative=["wheels", "other.whl"], name="other.whl")
        with self.assertRaisesRegex(self.acq.AcquisitionError, "twice"):
            self.acq.acquire_item(other, self.out, self.HOSTS, twice)
        http_error = fake_factory([urllib.error.HTTPError(self.url, 503, "Unavailable", None, None)])
        third = dict(self.item, relative=["wheels", "third.whl"], name="third.whl")
        with self.assertRaisesRegex(self.acq.AcquisitionError, "HTTP 503"):
            self.acq.acquire_item(third, self.out, self.HOSTS, http_error)
        self.assertEqual(len(http_error.calls), 1)

    def test_destination_must_stay_inside_output_dir(self):
        base = Path(self.tmp.name) / "base"
        outside = Path(self.tmp.name) / "outside"
        base.mkdir()
        outside.mkdir()
        with self.assertRaises(self.acq.AcquisitionError):
            self.acq.contained(base, base / ".." / "outside" / "file")
        link = base / "link"
        if not _make_junction(link, outside):
            self.skipTest("cannot create a junction or symbolic link here")
        try:
            with self.assertRaisesRegex(self.acq.AcquisitionError, "outside"):
                self.acq.contained(base, link / "file")
        finally:
            _remove_junction(link)
        with self.assertRaises(self.acq.AcquisitionError):
            self.acq.check_relative_path("../model.safetensors")

    def test_real_locks_give_the_pinned_plan(self):
        plan = self.acq.load_plan(DOWNLOAD_LOCK, MODEL_LOCK)
        self.assertEqual(len(plan["items"]), 35)
        self.assertEqual(sum(item["size"] for item in plan["items"]), 647065645)
        self.assertEqual(plan["allowed_hosts"], ["download.pytorch.org", "files.pythonhosted.org", "huggingface.co",
                                                 "us.aws.cdn.hf.co"])
        lock = json.loads(DOWNLOAD_LOCK.read_text(encoding="utf-8"))
        lock["packages"][0]["url"] = "https://download-r2.pytorch.org/whl/" + lock["packages"][0]["filename"]
        path = Path(self.tmp.name) / "downloads.json"
        path.write_text(json.dumps(lock), encoding="utf-8")
        with self.assertRaises(self.acq.AcquisitionError):
            self.acq.load_plan(path, MODEL_LOCK)


def _make_junction(link: Path, target: Path) -> bool:
    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
        return True
    except (ImportError, AttributeError, OSError):
        return False


def _remove_junction(link: Path) -> None:
    if os.name == "nt":
        os.rmdir(link)  # removes the junction itself, not its target
    else:
        link.unlink()


# ---------------------------------------------------------------- golden set, metrics, freeze


def real_items():
    from tracerag.corpus import read_chunks

    chunks = read_chunks(CHUNKS)
    return chunks, comparison.load_m006_golden_set(GOLDEN_M006, chunks)


class GoldenSetM006Tests(unittest.TestCase):
    def test_frozen_golden_set_matches_the_packet(self):
        _chunks, items = real_items()
        summary = comparison.m006_summary(items)
        self.assertEqual((summary["cases"], summary["languages"], summary["positives"], summary["negatives"]),
                         (24, {"pt": 12, "en": 12}, 16, 8))
        self.assertEqual((summary["pairs"], summary["relevance_references"], summary["distinct_relevant_chunks"]),
                         (12, 46, 13))
        expected_r = {f"m006-pair-{n:02d}": r for n, r in ((1, 7), (2, 4), (3, 7), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1))}
        self.assertEqual(summary["r_by_pair"], expected_r)
        for name in ("all", "pt", "en"):
            self.assertEqual(summary["macro_recall_ceiling"][name]["fraction"], "185/224")
        authors = comparison._authorship(items)
        self.assertEqual(authors["user_original_questions"], [f"m006-pt-0{n}" for n in range(1, 6)])

    def write(self, directory, items, newline="\n"):
        path = Path(directory) / "golden.jsonl"
        path.write_bytes("".join(json.dumps(item, ensure_ascii=False) + newline for item in items).encode("utf-8"))
        return path

    def test_contract_violations_are_rejected(self):
        chunks, items = real_items()
        first, second = items[0], items[1]
        stranger = "doc:missing:1"
        cases = {
            "duplicate id": [first, dict(second, id=first["id"])],
            "unknown chunk": [dict(first, relevant_chunk_ids=[*first["relevant_chunk_ids"][:-1], stranger]), second],
            "answerable without chunks": [dict(first, relevant_chunk_ids=[], evidence_locators=[], relevance_basis={}),
                                          second],
            "locator mismatch": [dict(first, evidence_locators=list(reversed(first["evidence_locators"]))), second],
            "basis order": [dict(first, relevance_basis=dict(reversed(list(first["relevance_basis"].items())))), second],
            "pair labels": [first, dict(second, question_scope="another scope")],
            "policy": [dict(first, relevance_policy="other"), dict(second, relevance_policy="other")],
            "extra key": [dict(first, extra=1), second],
        }
        with temp_dir() as tmp:
            for name, members in cases.items():
                with self.subTest(case=name), self.assertRaises(comparison.ComparisonError):
                    comparison.load_m006_golden_set(self.write(tmp, members), chunks)
            with self.assertRaisesRegex(comparison.ComparisonError, "LF"):
                comparison.load_m006_golden_set(self.write(tmp, [first, second], "\r\n"), chunks)
            negative = next(item for item in items if not item["answerable"])
            with self.assertRaises(comparison.ComparisonError):
                comparison.load_m006_golden_set(self.write(tmp, [dict(negative, relevant_chunk_ids=first["relevant_chunk_ids"][:1])]), chunks)


class MetricsTests(unittest.TestCase):
    def test_multiple_relevant_chunks_by_hand(self):
        four = ["r1", "r2", "r3", "r4"]
        result = comparison.score_ranking(["x", "r1", "r2"], four)
        self.assertEqual((result["recall_at_3"], result["rr_at_3"], result["first_relevant_rank"], result["relevant_retrieved"]),
                         (Fraction(2, 4), Fraction(1, 2), 2, 2))
        seven = [f"s{n}" for n in range(1, 8)]
        self.assertEqual(comparison.score_ranking(["s1", "y", "s2"], seven)["recall_at_3"], Fraction(2, 7))
        full = comparison.score_ranking(["s3", "s1", "s2", "s4"], seven)
        self.assertEqual((full["recall_at_3"], full["rr_at_3"]), (Fraction(3, 7), Fraction(1)))
        miss = comparison.score_ranking(["a", "b", "c", "r1"], four)
        self.assertEqual((miss["recall_at_3"], miss["rr_at_3"], miss["hit"], miss["first_relevant_rank"]),
                         (Fraction(0), Fraction(0), False, None))

    def test_ceilings_and_analytic_reference(self):
        self.assertEqual([comparison.recall_ceiling(r) for r in (7, 4, 1)], [Fraction(3, 7), Fraction(3, 4), Fraction(1)])
        ceilings = [comparison.recall_ceiling(r) for r in (7, 4, 7, 1, 1, 1, 1, 1)]
        self.assertEqual(sum(ceilings, Fraction(0)) / 8, Fraction(185, 224))
        expected = {1: (Fraction(1, 5), Fraction(11, 90)), 4: (Fraction(58, 91), Fraction(1741, 4095)),
                    7: (Fraction(57, 65), Fraction(379, 585))}
        for r, (hit, rr) in expected.items():
            reference = evaluation.chance_reference(15, r)
            self.assertEqual((reference["expected_recall"], reference["hit_probability"], reference["expected_rr"]),
                             (Fraction(1, 5), hit, rr))

    def test_paired_evaluation_counts_and_slices(self):
        chunks = {cid: {"document_id": "doc", "unit_locator": cid} for cid in ("a", "b", "c", "d", "e")}
        items = [
            {"id": "p1", "language": "pt", "query": "q1", "answerable": True, "relevant_chunk_ids": ["a"]},
            {"id": "e1", "language": "en", "query": "q2", "answerable": True, "relevant_chunk_ids": ["b", "c"]},
            {"id": "e2", "language": "en", "query": "q3", "answerable": True, "relevant_chunk_ids": ["d"]},
            {"id": "n1", "language": "pt", "query": "q4", "answerable": False, "relevant_chunk_ids": []},
        ]
        lexical = {"q1": ["a", "b", "c"], "q2": ["b", "a", "d"], "q3": ["a", "b", "c"], "q4": ["a", "b", "c"]}
        vectors = {"q1": ["b", "a", "c"], "q2": ["b", "c", "a"], "q3": ["d", "a", "b"], "q4": ["e", "d", "c"]}

        def runner(table, key):
            return lambda query, k: {"hits": [{"rank": r, key: 1.0 / r, "chunk_id": c} for r, c in enumerate(table[query][:k], 1)]}

        report = comparison.evaluate_set(items, chunks, runner(lexical, "score"), runner(vectors, "cosine_score"))
        rows = {row["id"]: row for row in report["questions"]}
        self.assertEqual(rows["p1"]["outcome_rr_at_3"], "bm25_better")
        self.assertEqual(rows["e1"]["outcome_rr_at_3"], "equal")
        self.assertEqual(rows["e1"]["outcome_recall_at_3"], "semantic_better")
        self.assertEqual(rows["e2"]["outcome_rr_at_3"], "semantic_better")
        self.assertNotIn("recall_at_3", rows["n1"]["bm25"])
        self.assertTrue(rows["n1"]["metrics"].startswith("N/A"))
        aggregates = report["aggregates"]
        self.assertEqual(aggregates["all_positive"]["queries"], 3)
        self.assertEqual(aggregates["negatives"]["ids"], ["n1"])
        self.assertEqual(aggregates["all_positive"]["paired_rr_at_3"], {"semantic_better": 1, "equal": 1, "bm25_better": 1})
        self.assertEqual(aggregates["all_positive"]["bm25"]["mrr_at_3"]["fraction"], "2/3")
        self.assertEqual(aggregates["all_positive"]["semantic"]["mrr_at_3"]["fraction"], "5/6")
        self.assertEqual(aggregates["en_positive"]["semantic"]["mean_recall_at_3"]["fraction"], "1/1")
        self.assertEqual(aggregates["pt_positive"]["bm25"]["queries_with_relevant_hit"], 1)
        for block in (aggregates["all_positive"], aggregates["pt_positive"], aggregates["en_positive"]):
            for metric in ("paired_rr_at_3", "paired_recall_at_3"):
                self.assertEqual(sum(block[metric].values()), block["queries"])

    def test_report_reproduction_check(self):
        def report(cosine, order=("a", "b")):
            question = {"id": "q", "bm25": {"hits": [{"rank": 1, "chunk_id": "a", "relevant": True, "score": -2.0}],
                                            "recall_at_3": 1, "rr_at_3": 1, "first_relevant_rank": 1},
                        "semantic": {"hits": [{"rank": i + 1, "chunk_id": c, "relevant": c == "a", "cosine_score": cosine - i}
                                              for i, c in enumerate(order)], "recall_at_3": 1, "rr_at_3": 1,
                                     "first_relevant_rank": 1}}
            block = {"questions": [question], "aggregates": {"x": 1}}
            return {"sets": {"m004_legacy": block, "m006": block}, "bm25_history_reproduction": {"status": "identical"}}

        base = report(0.9)
        self.assertEqual(comparison.compare_reports(base, report(0.9 + 5e-6))["status"], "reproduced")
        self.assertEqual(comparison.compare_reports(base, report(0.9 + 2e-5))["status"], "differs")
        self.assertEqual(comparison.compare_reports(base, report(0.9, ("b", "a")))["status"], "differs")


class FreezeTests(unittest.TestCase):
    CONTEXT = {"execution_revision": "TEST-REVISION", "packet_sha256": "a" * 64, "pre_sha256": "b" * 64,
               "semantic_t0": "2026-10-07 14:51:04 -0300"}

    def test_freeze_is_exclusive_and_verifiable(self):
        with temp_dir() as tmp:
            path = Path(tmp) / "freeze.json"
            created = comparison.create_freeze(REPO_ROOT, path, self.CONTEXT)
            freeze = created["freeze"]
            self.assertEqual(freeze["execution"]["revision"], "TEST-REVISION")
            self.assertEqual(sorted(freeze["inputs"]), sorted(comparison.FROZEN_INPUTS))
            self.assertEqual(freeze["resource_provenance"]["frozen_by_revision"], "M006-SEMANTIC-v2")
            self.assertEqual(freeze["golden_set_m006"]["macro_recall_ceiling"]["all"]["fraction"], "185/224")
            self.assertEqual(comparison.verify_freeze(REPO_ROOT, path)["inputs_verified"], len(comparison.FROZEN_INPUTS))
            with self.assertRaises(FileExistsError):
                comparison.create_freeze(REPO_ROOT, path, self.CONTEXT)
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["inputs"]["chunks"]["sha256"] = "0" * 64
            other = Path(tmp) / "tampered.json"
            other.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(comparison.ComparisonError, "changed after the freeze"):
                comparison.verify_freeze(REPO_ROOT, other)
            with self.assertRaises(comparison.ComparisonError):
                comparison.create_freeze(REPO_ROOT, Path(tmp) / "bad.json", dict(self.CONTEXT, pre_sha256="xyz"))


# ---------------------------------------------------------------- process isolation and resources


def run_python(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, check=False)


class IsolationTests(unittest.TestCase):
    def test_lexical_cli_and_m006_modules_do_not_import_ml(self):
        code = ("import json, sys\n"
                "from tracerag import __main__ as cli, comparison, performance, semantic\n"
                "cli.build_parser().parse_args(['search', '--query', 'x'])\n"
                "print(json.dumps(sorted(n for n in ('numpy', 'torch', 'transformers', 'tokenizers') if n in sys.modules)))\n")
        completed = run_python(code)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        self.assertEqual(json.loads(completed.stdout.decode("utf-8")), [])

    @unittest.skipUnless(HAS_ML, "needs torch and transformers")
    def test_network_is_blocked_and_ml_imports_need_no_network(self):
        code = ("import json, socket, urllib.request\n"
                "from tracerag import semantic\n"
                "semantic.prepare_process()\n"
                "ml = semantic.import_ml()\n"
                "out = {}\n"
                "for name, call in (('socket', lambda: socket.create_connection(('127.0.0.1', 9), timeout=1)),\n"
                "                   ('urllib', lambda: urllib.request.urlopen('https://huggingface.co', timeout=1))):\n"
                "    try:\n"
                "        call(); out[name] = 'not blocked'\n"
                "    except semantic.NetworkBlockedError:\n"
                "        out[name] = 'blocked'\n"
                "    except Exception as exc:\n"
                "        out[name] = 'blocked' if isinstance(exc.__context__, semantic.NetworkBlockedError) else repr(exc)\n"
                "report = semantic.network_report()\n"
                "print(json.dumps({'out': out, 'attempts': report['attempts'], 'threads': ml.torch.get_num_threads(),\n"
                "                  'interop': ml.torch.get_num_interop_threads(), 'cuda': ml.versions['torch_cuda']}))\n")
        completed = run_python(code)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        result = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
        self.assertEqual(result["out"], {"socket": "blocked", "urllib": "blocked"})
        self.assertTrue(result["attempts"])
        self.assertEqual((result["threads"], result["interop"], result["cuda"]), (1, 1, None))

    def test_cli_paths_must_stay_inside_data_m006(self):
        for value in ("data/m004/out.json", "data/m006", "../out.json", "data/m006/../m004/x.json"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "inside data/m006"):
                cli._m006_path(value, new=True)
        with self.assertRaisesRegex(ValueError, "end with .json"):
            cli._m006_path("data/m006/tests/tmp/out.txt", new=True, suffix=".json")
        with temp_dir() as tmp:
            existing = Path(tmp) / "exists.json"
            existing.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                cli._m006_path(str(existing), new=True)
            link = Path(tmp) / "link"
            if _make_junction(link, REPO_ROOT / "data" / ".venv-m006"):
                try:
                    with self.assertRaisesRegex(ValueError, "inside data/m006"):
                        cli._m006_path(str(link / "out.json"), new=True)
                finally:
                    _remove_junction(link)


class ResourceTests(unittest.TestCase):
    def test_guard_uses_the_smallest_of_three_samples(self):
        def reader(values):
            iterator = iter(values)
            return lambda: {"available_physical_bytes": next(iterator), "timestamp": "t"}

        sleeps = []
        guard = performance.memory_guard(reader=reader([3 << 30, 2 << 30, 5 << 30]), sleep=sleeps.append)
        self.assertEqual((guard["min_available_physical_bytes"], len(guard["samples"]), sleeps), (2 << 30, 3, [1.0, 1.0]))
        with self.assertRaises(performance.ResourceGuardError):
            performance.memory_guard(reader=reader([3 << 30, (2 << 30) - 1, 5 << 30]), sleep=lambda _s: None)

    def test_summaries_use_medians_of_seven_samples(self):
        per_query = [{"id": "a", "total_ns": [5, 1, 9, 3, 7, 2, 8]}, {"id": "b", "total_ns": [1, 1, 1, 1, 1, 1, 1]}]
        summary = performance.summarize(per_query, ("total_ns",))["total_ns"]
        self.assertEqual(summary["per_query_median_ns"], {"a": 5, "b": 1})
        self.assertEqual(summary["sweep_totals_ns"], [6, 2, 10, 4, 8, 3, 9])
        self.assertEqual(summary["median_sweep_total_ns"], 6)

    @unittest.skipUnless(os.name == "nt", "Windows memory APIs")
    def test_windows_memory_apis(self):
        system = performance.system_memory()
        self.assertGreater(system["available_physical_bytes"], 0)
        self.assertGreater(system["total_physical_bytes"], system["available_physical_bytes"])
        process = performance.process_memory()
        self.assertGreaterEqual(process["peak_working_set_bytes"], process["working_set_bytes"])
        self.assertGreater(process["working_set_bytes"], 0)

    def test_directory_size_is_logical(self):
        with temp_dir() as tmp:
            (Path(tmp) / "a").write_bytes(b"12345")
            (Path(tmp) / "sub").mkdir()
            (Path(tmp) / "sub" / "b").write_bytes(b"12")
            self.assertEqual({k: performance.directory_size(Path(tmp))[k] for k in ("files", "bytes")}, {"files": 2, "bytes": 7})


# ---------------------------------------------------------------- real model (opt-in, offline)


@unittest.skipUnless(REAL and HAS_ML, "set TRACERAG_M006_REAL=1 (after the freeze) to run the real offline model tests")
class RealModelTests(unittest.TestCase):
    """Verified local model, real encodings and evaluation, network blocked in this process."""

    @classmethod
    def setUpClass(cls):
        from tracerag.corpus import read_chunks

        cls.memory = {"system_before_load": performance.system_memory(), "process_before_load": performance.process_memory()}
        semantic.prepare_process(require_fresh=False)
        cls.socket_patch = mock.patch("socket.socket.connect", side_effect=AssertionError("network attempted"))
        cls.socket_patch.start()
        cls.ml = semantic.import_ml()
        cls.lock = semantic.load_model_lock(MODEL_LOCK)
        cls.encoder = semantic.Encoder(cls.ml, MODEL_DIR, cls.lock)
        cls.memory.update({"system_after_load": performance.system_memory(), "process_after_load": performance.process_memory()})
        cls.chunks = read_chunks(CHUNKS)
        cls.run_dir = TEST_ROOT / f"real-{dt.datetime.now():%Y%m%dT%H%M%S}-{os.getpid()}"
        cls.run_dir.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        cls.socket_patch.stop()
        network = semantic.network_report()
        semantic.release_network_block()
        cls.memory["process_end"] = performance.process_memory()
        summary = {"memory": cls.memory, "network_guard": network, "threads": cls.encoder.thread_report(),
                   "versions": cls.ml.versions, "load_checks": cls.encoder.load_checks}
        (cls.run_dir / "real-test-summary.json").write_bytes(
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        del cls.encoder

    def test_loaded_weights_come_from_the_verified_safetensors(self):
        from safetensors import safe_open

        self.assertEqual([f["sha256"] for f in self.encoder.verification["files"]],
                         [f["sha256"] for f in sorted(self.lock["files"], key=lambda f: f["path"])])
        state = self.encoder.model.state_dict()
        with safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt") as handle:
            key = sorted(k for k in handle.keys() if k.endswith("LayerNorm.weight"))[-1]
            stored = handle.get_tensor(key)
        loaded = state.get(key, state.get(key.removeprefix("bert.")))
        self.assertTrue(self.ml.torch.equal(stored, loaded))
        self.assertEqual(self.encoder.load_checks["parameter_dtypes"], ["torch.float32"])
        self.assertEqual(self.encoder.thread_report()["torch_num_threads"], 1)

    def test_rebuilt_index_and_evaluation_reproduce_the_first_measurement(self):
        np = self.ml.np
        summary = semantic.build_index(self.chunks, self.encoder, self.run_dir / "index",
                                       {"inputs": comparison.input_hashes(REPO_ROOT), "note": "real offline test run"})
        rebuilt = semantic.load_index(self.run_dir / "index", self.chunks, np)
        self.assertEqual(summary["shape"], [15, 384])
        self.assertTrue(all(abs(row["vector_norm"] - 1) <= semantic.UNIT_NORM_TOL for row in summary["passages"]))
        results = {"rebuilt_embeddings_sha256": summary["embeddings_sha256"]}
        if FIRST_INDEX.exists():
            first = semantic.load_index(FIRST_INDEX, self.chunks, np)
            check = semantic.compare_indexes(first, rebuilt, np)
            results["index_check"] = check
            self.assertEqual(check["status"], "reproduced_within_tolerance")
        legacy = evaluation.load_golden_set(REPO_ROOT / "evaluation" / "golden_set.jsonl", set(c["chunk_id"] for c in self.chunks))
        items = comparison.load_m006_golden_set(GOLDEN_M006, self.chunks)
        history = comparison.reproduce_bm25_history(self.chunks, legacy, REPO_ROOT / "data" / "m004" / "evaluation.json", REPO_ROOT)
        self.assertEqual(history["status"], "identical")
        sets = comparison.comparison_sets(self.chunks, legacy, items, comparison.bm25_runner(self.chunks),
                                          comparison.semantic_runner(rebuilt, self.encoder))
        report = {"sets": sets, "bm25_history_reproduction": history}
        if FIRST_COMPARISON.exists():
            first_report = json.loads(FIRST_COMPARISON.read_text(encoding="utf-8"))
            check = comparison.compare_reports(first_report, report)
            results["comparison_check"] = check
            self.assertEqual(check["status"], "reproduced", check["differences"])
        (self.run_dir / "real-test-results.json").write_bytes(
            (json.dumps(results, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    def test_network_hook_blocks_this_process(self):
        with self.assertRaises(semantic.NetworkBlockedError):
            socket.getaddrinfo("huggingface.co", 443)

    def test_lexical_search_still_runs_without_ml_imports(self):
        code = ("import contextlib, io, json, sys\n"
                "from tracerag import __main__ as cli\n"
                "buffer = io.StringIO()\n"
                "with contextlib.redirect_stdout(buffer):\n"
                "    code = cli.main(['search', '--query', 'ANOVA', '--top-k', '3'])\n"
                "hits = json.loads(buffer.getvalue())['hits']\n"
                "print(json.dumps({'code': code, 'hits': len(hits), 'ml': sorted(n for n in ('numpy', 'torch', 'transformers') if n in sys.modules)}))\n")
        completed = run_python(code)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        result = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
        self.assertEqual(result, {"code": 0, "hits": 1, "ml": []})  # only unit 3 contains "ANOVA" (M004 probe)


if __name__ == "__main__":
    unittest.main()
