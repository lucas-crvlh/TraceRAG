"""Offline unittest suite for the M004 lexical retrieval pilot.

Synthetic fixtures use invented text. No test needs the network: the
integration test runs only when the captured snapshots, the chunks and the
frozen golden set exist locally, and it blocks network calls while it runs.
Temporary files live under data/m004/test-temp/ (ignored by Git).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from tracerag import MISSION, corpus, evaluation, retrieval

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP_ROOT = REPO_ROOT / "data" / "m004" / "test-temp"
MANIFEST = REPO_ROOT / "corpus_manifest.yaml"
SNAPSHOTS = REPO_ROOT / "data" / "m004" / "snapshots.json"
CHUNKS = REPO_ROOT / "data" / "m004" / "chunks.jsonl"
GOLDEN = REPO_ROOT / "evaluation" / "golden_set.jsonl"

TITLE = "Sample Report: 7. Synthetic Section"
REVIEW = {
    "reviewed_on": "2026-10-06",
    "evidence_urls": ["https://example.org/terms"],
    "intended_use": "synthetic test entry",
    "decision": "approved",
    "notes": "synthetic notes",
}
PAGE = """<!DOCTYPE html>
<html><head><title>Sample</title><script>var token = "harrier";</script></head>
<body>
<nav><ul><li>Menu link heron</li></ul></nav>
<h1>Sample Report:   7. Synthetic
   Section</h1>
<div class="layout"><div class="text-with-summary">
<p><strong>7.1</strong> Kestrel lantern measures x<sub>i</sub> with y<sup>2</sup> &amp; more.</p>
<dd>first marigold item,</dd>
<dd>second marigold item.</dd>
<p>Continuation paragraph cites 7.2 inside a sentence.</p>
<script>document.write("osprey");</script>
<p><a name="7.2"></a><strong>7.2</strong> Granite falcon paragraph.</p>
<figure><div><img alt="square root of five" src="/img/root5.png"></div><figcaption></figcaption></figure>
<p>Trailing remark about the falcon.</p>
</div></div>
<p class="contact">Questions? <a href="/cdn-cgi/l/email-protection">[email&#160;protected]</a> pelican</p>
<footer>Footer text osprey</footer>
</body></html>
"""

_created_temp_root = False


def setUpModule():
    global _created_temp_root
    if not TEST_TEMP_ROOT.exists():
        TEST_TEMP_ROOT.mkdir(parents=True)
        _created_temp_root = True


def tearDownModule():
    if _created_temp_root and TEST_TEMP_ROOT.is_dir() and not any(TEST_TEMP_ROOT.iterdir()):
        TEST_TEMP_ROOT.rmdir()


def temp_dir():
    return tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT)


def manifest_text(reviewed_on='"2026-10-06"', decision="approved", second_id=None, extra_key=""):
    entry = """  - id: {doc_id}
    title: "Sample Report: 7. Synthetic Section"
    source_url: https://www.nist.gov/sample
    local_path: data/m004/raw/{doc_id}.html
{extra}    usage_review:
      reviewed_on: {reviewed_on}
      evidence_urls:
        - https://example.org/terms
      intended_use: synthetic test entry
      decision: {decision}
      notes: synthetic notes
"""
    text = "schema_version: 2\ndocuments:\n"
    text += entry.format(doc_id="doc-a", reviewed_on=reviewed_on, decision=decision, extra=extra_key)
    if second_id:
        text += entry.format(doc_id=second_id, reviewed_on=reviewed_on, decision=decision, extra="")
    return text


def record_for(raw: bytes) -> dict:
    digest = hashlib.sha256(raw).hexdigest()
    return {"sha256": digest, "source_version": f"snapshot:{digest}",
            "retrieved_at": "2026-10-06T10:00:00-03:00", "attribution": "Synthetic attribution."}


def document(doc_id="doc-a") -> corpus.Document:
    return corpus.Document(doc_id, TITLE, "https://www.nist.gov/sample", f"data/m004/raw/{doc_id}.html", dict(REVIEW))


def synthetic_chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "document_id": "doc", "unit_locator": "1", "part": 1, "text": text,
            "source_sha256": "0" * 64, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


class ManifestTests(unittest.TestCase):
    def load(self, text: str):
        with temp_dir() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            path = root / "corpus_manifest.yaml"
            path.write_bytes(text.encode("utf-8"))
            return corpus.load_manifest(path, root)

    def test_valid_manifest_loads(self):
        documents = self.load(manifest_text())
        self.assertEqual([d.id for d in documents], ["doc-a"])
        self.assertEqual(documents[0].usage_review["reviewed_on"], "2026-10-06")

    def test_unquoted_yaml_date_is_rejected_with_explanation(self):
        with self.assertRaisesRegex(corpus.CorpusError, "date object.*quoted string"):
            self.load(manifest_text(reviewed_on="2026-10-06"))

    def test_invalid_calendar_date_is_rejected(self):
        with self.assertRaisesRegex(corpus.CorpusError, "valid calendar date"):
            self.load(manifest_text(reviewed_on='"2026-02-30"'))

    def test_decision_must_be_approved(self):
        with self.assertRaisesRegex(corpus.CorpusError, "approved"):
            self.load(manifest_text(decision="pending"))

    def test_duplicate_ids_and_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(corpus.CorpusError, "duplicate"):
            self.load(manifest_text(second_id="doc-a"))
        with self.assertRaisesRegex(corpus.CorpusError, "unknown keys"):
            self.load(manifest_text(extra_key="    language: en\n"))


def _make_dir_link(link: Path, target: Path) -> bool:
    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
        return True
    except (ImportError, AttributeError, OSError):
        return False


class LocalPathTests(unittest.TestCase):
    def test_forbidden_forms_are_rejected(self):
        with temp_dir() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            invalid = ["", "/data/a.html", "C:/data/a.html", "data\\a.html", "data//a.html", "data/./a.html",
                       "data/../a.html", "data/a/", "data/", "data", "docs/a.html", "data/.gitkeep", None, 7]
            for value in invalid:
                with self.subTest(value=value), self.assertRaises(corpus.CorpusError):
                    corpus.validate_local_path(value, root)

    def test_valid_path_resolves_inside_data(self):
        with temp_dir() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            resolved = corpus.validate_local_path("data/m004/raw/a.html", root)
            self.assertTrue(resolved.is_relative_to((root / "data").resolve()))

    def test_link_escaping_data_is_rejected_after_resolution(self):
        with temp_dir() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            outside = root / "outside"
            outside.mkdir()
            link = root / "data" / "link"
            if not _make_dir_link(link, outside):
                self.skipTest("cannot create a junction or symbolic link here")
            try:
                with self.assertRaisesRegex(corpus.CorpusError, "outside data"):
                    corpus.validate_local_path("data/link/page.html", root)
            finally:
                if os.name == "nt":
                    os.rmdir(link)  # removes the junction itself, not its target
                else:
                    link.unlink()


class ExtractionTests(unittest.TestCase):
    def test_page_without_technical_body_is_rejected(self):
        page = PAGE.replace('class="text-with-summary"', 'class="summary"')
        with self.assertRaisesRegex(corpus.CorpusError, "exactly one div.text-with-summary, found 0"):
            corpus.parse_page(page, TITLE)

    def test_duplicate_body_and_wrong_title_are_rejected(self):
        twice = PAGE.replace("</body>", '<div class="text-with-summary"><p>Extra</p></div></body>')
        with self.assertRaisesRegex(corpus.CorpusError, "found 2"):
            corpus.parse_page(twice, TITLE)
        with self.assertRaisesRegex(corpus.CorpusError, "unexpected H1"):
            corpus.parse_page(PAGE, "Sample Report: 8. Other Section")

    def test_navigation_contact_footer_and_scripts_are_excluded(self):
        page = corpus.parse_page(PAGE, TITLE)
        self.assertEqual(page.title, TITLE)
        text = "\n".join(text for _tag, text, _notes in page.blocks)
        for word in ("heron", "harrier", "osprey", "pelican", "Footer", "protected", "Questions"):
            self.assertNotIn(word, text)

    def test_units_keep_lists_notes_and_internal_references(self):
        page = corpus.parse_page(PAGE, TITLE)
        units = corpus.split_units(page.blocks, corpus.section_number(TITLE))
        self.assertEqual([unit.locator for unit in units], ["7.1", "7.2"])
        first, second = ("\n".join(unit.texts) for unit in units)
        self.assertIn("first marigold item,\nsecond marigold item.", first)
        self.assertIn("cites 7.2 inside a sentence", first)
        self.assertIn("x_{i} with y^{2} & more.", first)
        self.assertIn("Trailing remark about the falcon.", second)
        self.assertEqual(units[1].notes, ["non-text image omitted (alt attribute: 'square root of five')"])
        self.assertNotIn("five", second)

    def test_number_starts_unit_only_at_paragraph_start(self):
        blocks = [("p", "7.1 Alpha text.", ()), ("p", "See 7.3 and 7.4 for details.", ()),
                  ("dd", "7.5 listed item", ()), ("blockquote", "7.6 quoted note", ()), ("p", "7.2 Beta text.", ())]
        units = corpus.split_units(blocks, "7")
        self.assertEqual([unit.locator for unit in units], ["7.1", "7.2"])
        self.assertEqual(len(units[0].texts), 4)

    def test_section_without_numbered_paragraphs_uses_section_locator(self):
        units = corpus.split_units([("p", "Only one paragraph, citing 3.2 later.", ())], "3")
        self.assertEqual([unit.locator for unit in units], ["3"])

    def test_out_of_order_subsection_is_rejected(self):
        with self.assertRaisesRegex(corpus.CorpusError, "out of order"):
            corpus.split_units([("p", "7.2 Beta.", ()), ("p", "7.1 Alpha.", ())], "7")


class ChunkingTests(unittest.TestCase):
    def test_chunks_are_deterministic_and_traceable(self):
        raw = PAGE.encode("utf-8")
        record = record_for(raw)
        units = corpus.split_units(corpus.parse_page(PAGE, TITLE).blocks, "7")
        first = corpus.make_chunks(document(), record, units, "7")
        second = corpus.make_chunks(document(), record, corpus.split_units(corpus.parse_page(PAGE, TITLE).blocks, "7"), "7")
        self.assertEqual(corpus.chunks_jsonl(first), corpus.chunks_jsonl(second))
        chunk = first[0]
        self.assertEqual(chunk["chunk_id"], f"doc-a:{record['sha256']}:7.1:1")
        self.assertEqual(chunk["source_version"], f"snapshot:{record['sha256']}")
        self.assertEqual(chunk["text_sha256"], hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest())
        self.assertEqual(chunk["word_count"], len(chunk["text"].split()))
        self.assertEqual((chunk["publication_id"], chunk["section"], chunk["unit_locator"], chunk["part"]),
                         ("nist-tn-1297", "7", "7.1", 1))
        self.assertIn(corpus.NORMALIZATION_NOTE, chunk["extraction_notes"])
        self.assertIn(corpus.SUBSUP_NOTE, chunk["extraction_notes"])
        self.assertIn("non-text image omitted (alt attribute: 'square root of five')", first[1]["extraction_notes"])
        line = corpus.chunks_jsonl(first).split(b"\n")[0]
        self.assertEqual(json.loads(line), chunk)

    def test_long_unit_is_split_without_overlap(self):
        words = [f"w{number}" for number in range(1, 1001)]
        unit = corpus.Unit("7.1", [" ".join(words[:600]), " ".join(words[600:])])
        chunks = corpus.make_chunks(document(), record_for(b"x"), [unit], "7")
        self.assertEqual([c["word_count"] for c in chunks], [400, 400, 200])
        self.assertEqual([c["part"] for c in chunks], [1, 2, 3])
        self.assertEqual(" ".join(c["text"] for c in chunks).split(), words)
        self.assertEqual(len({c["chunk_id"] for c in chunks}), 3)

    def test_ingest_creates_files_once_and_then_verifies_them(self):
        with temp_dir() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            snapshots = root / "data" / "m004" / "snapshots.json"
            raw = PAGE.encode("utf-8")

            def fake_fetch(doc):
                record = record_for(raw)
                record.update({"document_id": doc.id, "title": doc.title, "requested_url": doc.source_url,
                               "local_path": doc.local_path, "usage_review": doc.usage_review,
                               "bytes": len(raw), "charset": "utf-8", "observed_title": TITLE,
                               "final_url": doc.source_url, "http_status": 200, "content_type": "text/html"})
                return raw, record

            first = corpus.ingest([document()], root, snapshots, mission=MISSION, fetch=fake_fetch)
            self.assertEqual(first["status"], "captured")
            again = corpus.ingest([document()], root, snapshots, mission=MISSION, fetch=None)
            self.assertEqual(again["status"], "verified_existing_snapshots")
            (root / "data" / "m004" / "raw" / "doc-a.html").write_bytes(raw + b"changed")
            with self.assertRaisesRegex(corpus.CorpusError, "do not match"):
                corpus.ingest([document()], root, snapshots, mission=MISSION, fetch=None)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = retrieval.LexicalIndex([
            synthetic_chunk("c-1", "kestrel lantern"),
            synthetic_chunk("c-2", "granite falcon falcon"),
            synthetic_chunk("c-3", "marigold heron"),
        ])

    def test_empty_queries_return_no_hits(self):
        for query in ("", "   ", "?!* ()", None):
            with self.subTest(query=query):
                result = self.index.search(query, 3)
                self.assertEqual((result["status"], result["hits"], result["terms"]), ("no_query_terms", [], []))

    def test_fts_syntax_characters_are_neutralised(self):
        result = self.index.search('falcon AND "NEAR(" ^col: * - OR falcon)', 3)
        self.assertEqual(result["terms"], ["falcon", "and", "near", "col", "or"])
        self.assertEqual(result["match"], '"falcon" OR "and" OR "near" OR "col" OR "or"')
        self.assertEqual([hit["chunk_id"] for hit in result["hits"]], ["c-2"])

    def test_no_lexical_match_is_explicit(self):
        result = self.index.search("zebra", 3)
        self.assertEqual((result["status"], result["hits"]), ("no_lexical_match", []))

    def test_ties_are_broken_by_chunk_id_and_scores_ascend(self):
        index = retrieval.LexicalIndex([
            synthetic_chunk("b", "same words here"), synthetic_chunk("a", "same words here"),
            synthetic_chunk("c", "other text entirely"), synthetic_chunk("d", "unrelated content"),
        ])
        hits = index.search("same", 3)["hits"]
        self.assertEqual([hit["chunk_id"] for hit in hits], ["a", "b"])
        self.assertEqual(hits[0]["score"], hits[1]["score"])
        ranked = self.index.search("falcon kestrel", 3)["hits"]
        self.assertEqual([hit["rank"] for hit in ranked], [1, 2])
        self.assertLessEqual(ranked[0]["score"], ranked[1]["score"])

    def test_invalid_top_k_and_duplicate_ids_are_rejected(self):
        for value in (0, -1, True, "3"):
            with self.subTest(top_k=value), self.assertRaises(ValueError):
                self.index.search("falcon", value)
        with self.assertRaises(ValueError):
            retrieval.LexicalIndex([synthetic_chunk("x", "a"), synthetic_chunk("x", "b")])


class MetricsTests(unittest.TestCase):
    def test_hand_judged_examples(self):
        self.assertEqual(evaluation.recall_at_k(["x", "a", "y"], ["a", "b"]), Fraction(1, 2))
        self.assertEqual(evaluation.reciprocal_rank_at_k(["x", "a", "y"], ["a", "b"]), Fraction(1, 2))
        self.assertEqual(evaluation.recall_at_k(["a", "b", "c", "d"], ["d"]), Fraction(0))
        self.assertEqual(evaluation.reciprocal_rank_at_k(["a", "b", "c", "d"], ["d"]), Fraction(0))
        self.assertEqual(evaluation.recall_at_k(["a", "z"], ["a", "a"]), Fraction(1))
        self.assertEqual(evaluation.reciprocal_rank_at_k(["q", "r", "a"], ["a"]), Fraction(1, 3))

    def test_metrics_are_undefined_without_relevant_chunks(self):
        with self.assertRaises(ValueError):
            evaluation.recall_at_k(["a"], [])
        with self.assertRaises(ValueError):
            evaluation.reciprocal_rank_at_k(["a"], [])

    def test_analytic_chance_reference(self):
        single = evaluation.chance_reference(15, 1)
        self.assertEqual((single["expected_recall"], single["hit_probability"], single["expected_rr"]),
                         (Fraction(1, 5), Fraction(1, 5), Fraction(11, 90)))
        double = evaluation.chance_reference(10, 2)
        self.assertEqual((double["expected_recall"], double["hit_probability"], double["expected_rr"]),
                         (Fraction(3, 10), Fraction(8, 15), Fraction(46, 135)))
        small = evaluation.chance_reference(2, 1)
        self.assertEqual((small["k"], small["expected_recall"], small["expected_rr"]), (2, Fraction(1), Fraction(3, 4)))


class GoldenSetTests(unittest.TestCase):
    def setUp(self):
        self._dirs = []

    def tearDown(self):
        for directory in self._dirs:
            directory.cleanup()

    def write(self, items) -> Path:
        directory = temp_dir()
        self._dirs.append(directory)
        path = Path(directory.name) / "golden.jsonl"
        path.write_bytes(b"".join(json.dumps(item).encode("utf-8") + b"\n" for item in items))
        return path

    @staticmethod
    def item(qid, answerable, relevant, query="falcon granite"):
        return {"id": qid, "query": query, "language": "en", "answerable": answerable,
                "relevant_chunk_ids": relevant, "evidence_locators": ["doc#1"] if relevant else [],
                "judgment_notes": "synthetic judgment"}

    def test_unknown_relevant_id_is_rejected(self):
        path = self.write([self.item("q1", True, ["missing"])])
        with self.assertRaisesRegex(evaluation.GoldenSetError, "unknown relevant chunk ids"):
            evaluation.load_golden_set(path, {"c-1"})

    def test_answerable_and_negative_contracts(self):
        with self.assertRaisesRegex(evaluation.GoldenSetError, "at least one relevant"):
            evaluation.load_golden_set(self.write([self.item("q1", True, [])]), {"c-1"})
        with self.assertRaisesRegex(evaluation.GoldenSetError, "no relevant chunks"):
            evaluation.load_golden_set(self.write([self.item("q1", False, ["c-1"])]), {"c-1"})

    def test_evaluation_counts_only_answerable_queries(self):
        chunks = [synthetic_chunk("c-1", "kestrel lantern"), synthetic_chunk("c-2", "granite falcon falcon"),
                  synthetic_chunk("c-3", "marigold heron")]
        path = self.write([self.item("q1", True, ["c-2"]), self.item("q2", True, ["c-3", "c-1"], "kestrel"),
                           self.item("q3", False, [], "zebra granite")])
        items = evaluation.load_golden_set(path, {c["chunk_id"] for c in chunks})
        report = evaluation.run_evaluation(chunks, items)
        rows = {row["id"]: row for row in report["results"]}
        self.assertEqual(rows["q1"]["rr_at_3"]["fraction"], "1/1")
        self.assertEqual(rows["q2"]["recall_at_3"]["fraction"], "1/2")
        self.assertEqual(rows["q2"]["rr_at_3"]["fraction"], "1/1")
        self.assertTrue(rows["q3"]["retrieved_any"])
        self.assertNotIn("recall_at_3", rows["q3"])
        self.assertEqual(report["aggregates"]["answerable_queries"], 2)
        self.assertEqual(report["aggregates"]["mean_recall_at_3"]["fraction"], "3/4")
        self.assertEqual(report["aggregates"]["mrr_at_3"]["fraction"], "1/1")
        self.assertEqual(report["chance_reference"]["n_chunks"], 3)


def _frozen_artifacts_present() -> bool:
    return SNAPSHOTS.is_file() and CHUNKS.is_file() and GOLDEN.is_file() and GOLDEN.stat().st_size > 0


@unittest.skipUnless(_frozen_artifacts_present(), "needs the captured snapshots, chunks.jsonl and the frozen golden set")
class PilotIntegrationTest(unittest.TestCase):
    """Real snapshots -> real chunks -> search -> evaluation -> offline repetition."""

    def test_offline_pipeline_reproduces_frozen_artifacts(self):
        blocked = AssertionError("network access attempted during an offline test")
        with mock.patch("urllib.request.urlopen", side_effect=blocked), \
                mock.patch("urllib.request.OpenerDirector.open", side_effect=blocked):
            documents = corpus.load_manifest(MANIFEST, REPO_ROOT)
            records = corpus.load_snapshots(SNAPSHOTS, documents, REPO_ROOT)
            rebuilt = corpus.build_corpus_chunks(documents, records, REPO_ROOT)
            self.assertEqual(corpus.chunks_jsonl(rebuilt), CHUNKS.read_bytes())
            chunks = corpus.read_chunks(CHUNKS)
            by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

            probe = retrieval.LexicalIndex(chunks).search("ANOVA", 3)
            self.assertTrue(any(hit["document_id"] == "tn1297-s3" and "ANOVA" in hit["text"] for hit in probe["hits"]))
            for hit in probe["hits"]:
                self.assertEqual(hit["text"], by_id[hit["chunk_id"]]["text"])
                self.assertEqual(hit["source_sha256"], records[hit["document_id"]]["sha256"])

            items = evaluation.load_golden_set(GOLDEN, set(by_id))
            self.assertEqual(sum(item["answerable"] for item in items), 8)
            self.assertEqual(sum(not item["answerable"] for item in items), 2)
            first = evaluation.run_evaluation(chunks, items)
            second = evaluation.run_evaluation(chunks, items)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
