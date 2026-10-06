"""Corpus manifest validation, HTTPS snapshot capture and HTML extraction.

The pilot corpus is described by ``corpus_manifest.yaml`` (schema 2). Each
captured page is stored as an ignored snapshot under ``data/m004/raw`` and is
identified by the SHA-256 of its raw bytes. Every chunk derived from it keeps
that hash, so a retrieved passage can be traced back to the exact bytes that
were captured. HTML and text from the corpus are handled as data only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import yaml

PUBLICATION_ID = "nist-tn-1297"
SCHEMA_VERSION = 2
RAW_DIR = Path("data", "m004", "raw")
ALLOWED_HOST = "www.nist.gov"
USER_AGENT = "TraceRAG-M004-pilot/0.1 (local study of NIST TN 1297 web sections; Python-urllib)"
TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
BODY_CLASS = "text-with-summary"
EXTRACTOR_VERSION = "m004-html-1"
MAX_CHUNK_WORDS = 400
CITATION = (
    "Taylor, B.N. and Kuyatt, C.E. (1994), Guidelines for Evaluating and Expressing the "
    "Uncertainty of NIST Measurement Results, NIST Technical Note 1297 (1994 Edition); "
    'web section "{title}", adapted from the Technical Note, National Institute of '
    "Standards and Technology, {url} (Accessed {accessed}). "
    "Republished courtesy of the National Institute of Standards and Technology."
)
NORMALIZATION_NOTE = (
    "HTML entities decoded; whitespace runs collapsed to one space; "
    "one LF between source blocks; text not summarized, translated or corrected"
)
SUBSUP_NOTE = "subscript and superscript kept with the delimiters _{...} and ^{...}"

_DOCUMENT_KEYS = ("id", "title", "source_url", "local_path", "usage_review")
_REVIEW_KEYS = ("reviewed_on", "evidence_urls", "intended_use", "decision", "notes")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SECTION_RE = re.compile(r"[^:]*:\s*(\d+)\.\s+\S")
_BLOCK_TAGS = frozenset(
    "address article aside blockquote dd div dl dt figcaption figure footer "
    "h2 h3 h4 h5 h6 header li ol p pre section table td th tr ul".split()
)
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
_CHUNK_KEYS = (
    "chunk_id", "document_id", "publication_id", "title", "source_url", "source_sha256",
    "source_version", "snapshot_path", "retrieved_at", "section", "unit_locator", "part",
    "text", "text_sha256", "word_count", "extraction_notes", "attribution",
)


class CorpusError(ValueError):
    """The manifest, a snapshot or an extraction violates the pilot contract."""


@dataclass(frozen=True)
class Document:
    """One manifest entry: a single captured HTML section."""

    id: str
    title: str
    source_url: str
    local_path: str
    usage_review: dict


@dataclass(frozen=True)
class Page:
    """Validated page title and the ordered blocks of its technical body."""

    title: str
    blocks: tuple


@dataclass
class Unit:
    """Text of one locator (subsection or whole section) before chunking."""

    locator: str
    texts: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(payload) -> bytes:
    """Canonical, human-readable JSON (UTF-8, sorted keys, LF ending)."""
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new_file(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data``; never replace an existing file."""
    with open(path, "xb") as handle:
        handle.write(data)


# ---------------------------------------------------------------- manifest


def load_manifest(path: Path, repo_root: Path) -> list[Document]:
    """Read the manifest with ``yaml.safe_load`` and validate every entry."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CorpusError("manifest must be a YAML mapping")
    unknown = sorted(set(data) - {"schema_version", "documents"})
    if unknown:
        raise CorpusError(f"manifest has unknown top-level keys: {unknown}")
    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise CorpusError(f"schema_version must be the integer {SCHEMA_VERSION}")
    entries = data.get("documents")
    if not isinstance(entries, list):
        raise CorpusError("documents must be a list")
    documents, seen = [], set()
    for position, entry in enumerate(entries, start=1):
        document = _validate_entry(entry, f"documents[{position}]", repo_root)
        if document.id in seen:
            raise CorpusError(f"duplicate document id {document.id!r}")
        seen.add(document.id)
        documents.append(document)
    return documents


def _validate_entry(entry, where: str, repo_root: Path) -> Document:
    if not isinstance(entry, dict):
        raise CorpusError(f"{where} must be a mapping")
    _require_exact_keys(entry, _DOCUMENT_KEYS, where)
    for key in ("id", "title", "source_url", "local_path"):
        _require_text(entry[key], f"{where}.{key}")
    _require_https_url(entry["source_url"], f"{where}.source_url")
    validate_local_path(entry["local_path"], repo_root)
    _validate_review(entry["usage_review"], f"{where}.usage_review")
    return Document(
        entry["id"], entry["title"], entry["source_url"], entry["local_path"], dict(entry["usage_review"])
    )


def _validate_review(review, where: str) -> None:
    if not isinstance(review, dict):
        raise CorpusError(f"{where} must be a mapping")
    _require_exact_keys(review, _REVIEW_KEYS, where)
    reviewed_on = review["reviewed_on"]
    if isinstance(reviewed_on, dt.date):
        raise CorpusError(
            f"{where}.reviewed_on was loaded by YAML as a date object; "
            'write it as a quoted string, for example "2026-10-06"'
        )
    if not isinstance(reviewed_on, str) or not _DATE_RE.fullmatch(reviewed_on):
        raise CorpusError(f"{where}.reviewed_on must be a quoted 'YYYY-MM-DD' string")
    try:
        dt.date.fromisoformat(reviewed_on)
    except ValueError as exc:
        raise CorpusError(f"{where}.reviewed_on is not a valid calendar date") from exc
    urls = review["evidence_urls"]
    if not isinstance(urls, list) or not urls:
        raise CorpusError(f"{where}.evidence_urls must be a non-empty list")
    for number, url in enumerate(urls, start=1):
        _require_text(url, f"{where}.evidence_urls[{number}]")
        _require_https_url(url, f"{where}.evidence_urls[{number}]")
    _require_text(review["intended_use"], f"{where}.intended_use")
    if review["decision"] != "approved":
        raise CorpusError(f"{where}.decision must be 'approved' before the entry is registered")
    _require_text(review["notes"], f"{where}.notes")


def _require_exact_keys(mapping: dict, keys: tuple, where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    unknown = sorted(str(key) for key in mapping if key not in keys)
    if missing or unknown:
        raise CorpusError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def _require_text(value, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{where} must be a non-empty string")


def _require_https_url(value: str, where: str) -> None:
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        raise CorpusError(f"{where} must be an https URL")


def validate_local_path(local_path, repo_root: Path) -> Path:
    """Apply the schema-2 ``local_path`` rules and return the resolved path.

    The path is relative to the repository root, starts with ``data/``, uses
    ``/`` and names a file. Containment is checked after resolution, so
    symbolic links and junctions cannot lead outside ``data/``.
    """
    if not isinstance(local_path, str) or not local_path:
        raise CorpusError("local_path must be a non-empty string")
    if "\\" in local_path:
        raise CorpusError(f"local_path {local_path!r} uses a backslash")
    if local_path.startswith("/") or ":" in local_path or "\x00" in local_path:
        raise CorpusError(f"local_path {local_path!r} is absolute or has a drive prefix")
    if not local_path.startswith("data/"):
        raise CorpusError(f"local_path {local_path!r} must start with 'data/'")
    segments = local_path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise CorpusError(f"local_path {local_path!r} has an empty, '.' or '..' segment or a trailing slash")
    if local_path == "data/.gitkeep":
        raise CorpusError("local_path must not be data/.gitkeep")
    data_root = (Path(repo_root) / "data").resolve()
    resolved = Path(repo_root).joinpath(*segments).resolve()
    if resolved == data_root or not resolved.is_relative_to(data_root):
        raise CorpusError(f"local_path {local_path!r} resolves outside data/")
    return resolved


# ---------------------------------------------------------------- capture


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to HTTPS URLs on the same NIST host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.hostname != ALLOWED_HOST:
            raise CorpusError(f"redirect to {newurl!r} rejected; only https://{ALLOWED_HOST} is allowed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _require_allowed_host(url: str, where: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != ALLOWED_HOST:
        raise CorpusError(f"{where}: {url!r} is not an https URL on {ALLOWED_HOST}")


def attribution(document: Document, accessed: dt.date) -> str:
    """Citation recommended by NIST, followed by its courtesy statement."""
    return CITATION.format(
        title=document.title, url=document.source_url,
        accessed=f"{accessed:%B} {accessed.day}, {accessed.year}",
    )


def capture(document: Document) -> tuple[bytes, dict]:
    """Download one section page and return its raw bytes and snapshot record.

    Nothing is written here: the page is validated (status, type, size, host,
    H1 and technical body) before the caller decides to store it.
    """
    url = document.source_url
    _require_allowed_host(url, document.id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    opener = urllib.request.build_opener(_RedirectGuard())
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        raise CorpusError(f"{document.id}: HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise CorpusError(f"{document.id}: request failed ({exc.reason})") from exc
    retrieved = dt.datetime.now().astimezone().replace(microsecond=0)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise CorpusError(f"{document.id}: response is larger than {MAX_RESPONSE_BYTES} bytes")
    if status != 200:
        raise CorpusError(f"{document.id}: unexpected HTTP status {status}")
    if content_type.split(";")[0].strip().lower() != "text/html":
        raise CorpusError(f"{document.id}: unexpected content type {content_type!r}")
    _require_allowed_host(final_url, document.id)
    try:
        page = parse_page(raw.decode(charset), document.title)
    except (LookupError, UnicodeDecodeError) as exc:
        raise CorpusError(f"{document.id}: cannot decode the page as {charset}") from exc
    digest = sha256_hex(raw)
    record = {
        "document_id": document.id,
        "publication_id": PUBLICATION_ID,
        "title": document.title,
        "observed_title": page.title,
        "section": section_number(document.title),
        "requested_url": url,
        "final_url": final_url,
        "http_status": status,
        "content_type": content_type,
        "charset": charset,
        "retrieved_at": retrieved.isoformat(),
        "bytes": len(raw),
        "sha256": digest,
        "source_version": f"snapshot:{digest}",
        "local_path": document.local_path,
        "user_agent": USER_AGENT,
        "attribution": attribution(document, retrieved.date()),
        "usage_review": document.usage_review,
    }
    return raw, record


def ingest(documents: list[Document], repo_root: Path, snapshots_path: Path, *, mission: str, fetch=capture) -> dict:
    """Capture the reviewed documents once, or verify and reuse existing snapshots.

    Existing snapshots are never downloaded again or replaced: they are
    checked against the SHA-256 recorded in ``snapshots.json``.
    """
    raw_root = (Path(repo_root) / RAW_DIR).resolve()
    targets = {}
    for document in documents:
        target = validate_local_path(document.local_path, repo_root)
        if not target.is_relative_to(raw_root):
            raise CorpusError(f"{document.id}: snapshots must be stored under {RAW_DIR.as_posix()}/")
        targets[document.id] = target
    if Path(snapshots_path).exists():
        records = load_snapshots(snapshots_path, documents, repo_root)
        return {"status": "verified_existing_snapshots", "snapshots": [_summary(records[d.id]) for d in documents]}
    present = [document.id for document in documents if targets[document.id].exists()]
    if present:
        raise CorpusError(f"raw snapshots exist without {Path(snapshots_path).name}: {present}; nothing was overwritten")
    captured = [fetch(document) for document in documents]
    raw_root.mkdir(parents=True, exist_ok=True)
    for document, (raw, _record) in zip(documents, captured):
        write_new_file(targets[document.id], raw)
    payload = {"mission": mission, "publication_id": PUBLICATION_ID, "snapshots": [record for _raw, record in captured]}
    write_new_file(Path(snapshots_path), json_bytes(payload))
    return {"status": "captured", "snapshots": [_summary(record) for _raw, record in captured]}


def _summary(record: dict) -> dict:
    keys = ("document_id", "observed_title", "final_url", "http_status", "content_type",
            "retrieved_at", "bytes", "sha256", "local_path")
    return {key: record[key] for key in keys}


def load_snapshots(path: Path, documents: list[Document], repo_root: Path) -> dict:
    """Load ``snapshots.json`` and verify every raw file against its recorded hash."""
    payload = json.loads(Path(path).read_bytes().decode("utf-8"))
    records = payload.get("snapshots") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise CorpusError(f"{Path(path).name} has no snapshot list")
    by_id = {}
    for record in records:
        document_id = record.get("document_id") if isinstance(record, dict) else None
        if not isinstance(document_id, str) or document_id in by_id:
            raise CorpusError(f"{Path(path).name} has a missing or duplicate document_id")
        by_id[document_id] = record
    if list(by_id) != [document.id for document in documents]:
        raise CorpusError("snapshots.json does not list exactly the manifest documents, in order")
    for document in documents:
        record = by_id[document.id]
        expected = (document.title, document.source_url, document.local_path, document.usage_review)
        found = (record.get("title"), record.get("requested_url"), record.get("local_path"), record.get("usage_review"))
        if found != expected:
            raise CorpusError(f"{document.id}: manifest entry changed since capture; review it before reuse")
        raw = validate_local_path(document.local_path, repo_root).read_bytes()
        if sha256_hex(raw) != record.get("sha256") or len(raw) != record.get("bytes"):
            raise CorpusError(f"{document.id}: snapshot bytes do not match the recorded SHA-256")
    return by_id


# ---------------------------------------------------------------- extraction


def _normalize(text: str) -> str:
    return " ".join(text.split())


class _PageParser(HTMLParser):
    """Collect the H1 text and the blocks of the single technical body.

    Only the content of ``div.text-with-summary`` up to its own closing tag is
    kept; navigation, contact data, footer and scripts are never read as text.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.h1_titles: list[str] = []
        self.body_count = 0
        self.blocks: list[tuple] = []
        self._h1: list[str] | None = None
        self._stack: list[str] = []
        self._skip = 0
        self._parts: list[str] = []
        self._notes: list[str] = []

    @property
    def body_open(self) -> bool:
        return bool(self._stack)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "h1" and self._h1 is None:
            self._h1 = []
        if not self._stack:
            if tag == "div" and BODY_CLASS in (attributes.get("class") or "").split():
                self.body_count += 1
                self._stack = ["div"]
            return
        if self._skip or tag in _SKIP_TAGS:
            if tag in _SKIP_TAGS:
                self._skip += 1
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            self._stack.append(tag)
        elif tag == "sub":
            self._parts.append("_{")
        elif tag == "sup":
            self._parts.append("^{")
        elif tag == "br":
            self._parts.append(" ")
        elif tag == "img":
            self._notes.append(f"non-text image omitted (alt attribute: {attributes.get('alt')!r})")

    def handle_endtag(self, tag):
        if tag == "h1" and self._h1 is not None:
            self.h1_titles.append(_normalize("".join(self._h1)))
            self._h1 = None
        if not self._stack:
            return
        if self._skip:
            if tag in _SKIP_TAGS:
                self._skip -= 1
            return
        if tag in _BLOCK_TAGS and tag in self._stack:
            self._flush()
            while self._stack.pop() != tag:
                pass
        elif tag in ("sub", "sup"):
            self._parts.append("}")

    def handle_data(self, data):
        if self._h1 is not None:
            self._h1.append(data)
        if self._stack and not self._skip:
            self._parts.append(data)

    def _flush(self) -> None:
        text = _normalize("".join(self._parts))
        if text or self._notes:
            self.blocks.append((self._stack[-1], text, tuple(self._notes)))
        self._parts, self._notes = [], []


def parse_page(html_text: str, expected_title: str) -> Page:
    """Parse one section page and validate its H1 and technical body."""
    parser = _PageParser()
    parser.feed(html_text)
    parser.close()
    if parser.body_count != 1:
        raise CorpusError(f"expected exactly one div.{BODY_CLASS}, found {parser.body_count}")
    if parser.body_open:
        raise CorpusError(f"div.{BODY_CLASS} is not closed; the page looks truncated")
    if parser.h1_titles != [_normalize(expected_title)]:
        raise CorpusError(f"unexpected H1 {parser.h1_titles!r}; expected {expected_title!r}")
    if not any(text for _tag, text, _notes in parser.blocks):
        raise CorpusError(f"div.{BODY_CLASS} has no text")
    return Page(parser.h1_titles[0], tuple(parser.blocks))


def section_number(title: str) -> str:
    """Section number written in a title such as ``NIST TN 1297: 2. Classification ...``."""
    match = _SECTION_RE.match(title)
    if not match:
        raise CorpusError(f"cannot read a section number from title {title!r}")
    return match.group(1)


def split_units(blocks, section: str) -> list[Unit]:
    """Group blocks into units by locator.

    A new unit starts only at a paragraph (``p``) whose text begins with
    ``<section>.<n>`` followed by a space. Numbers cited in the middle of a
    sentence, list items and notes stay in the current unit. Text before the
    first numbered paragraph, or a section without numbered paragraphs, uses
    the section number itself as locator.
    """
    start = re.compile(rf"{re.escape(section)}\.(\d+)(?=\s|$)")
    units: list[Unit] = []
    last = 0
    for tag, text, notes in blocks:
        match = start.match(text) if tag == "p" else None
        if match:
            number = int(match.group(1))
            if number <= last:
                raise CorpusError(f"subsection {section}.{number} appears out of order")
            last = number
            units.append(Unit(f"{section}.{number}"))
        elif not units:
            units.append(Unit(section))
        if text:
            units[-1].texts.append(text)
        units[-1].notes.extend(notes)
    return units


def make_chunks(document: Document, record: dict, units: list[Unit], section: str) -> list[dict]:
    """Turn units into chunks of at most ``MAX_CHUNK_WORDS`` words, with provenance."""
    chunks = []
    for unit in units:
        text = "\n".join(unit.texts)
        words = text.split()
        if not words:
            raise CorpusError(f"{document.id}: unit {unit.locator} has no text")
        if len(words) <= MAX_CHUNK_WORDS:
            parts = [text]
        else:
            parts = [" ".join(words[i:i + MAX_CHUNK_WORDS]) for i in range(0, len(words), MAX_CHUNK_WORDS)]
        for number, part in enumerate(parts, start=1):
            notes = [NORMALIZATION_NOTE]
            if "_{" in part or "^{" in part:
                notes.append(SUBSUP_NOTE)
            notes.extend(unit.notes)
            if len(parts) > 1:
                notes.append(
                    f"part {number} of {len(parts)}: unit above {MAX_CHUNK_WORDS} words split on "
                    "whitespace without overlap; line breaks inside the part flattened"
                )
            chunks.append({
                "chunk_id": f"{document.id}:{record['sha256']}:{unit.locator}:{number}",
                "document_id": document.id,
                "publication_id": PUBLICATION_ID,
                "title": document.title,
                "source_url": document.source_url,
                "source_sha256": record["sha256"],
                "source_version": record["source_version"],
                "snapshot_path": document.local_path,
                "retrieved_at": record["retrieved_at"],
                "section": section,
                "unit_locator": unit.locator,
                "part": number,
                "text": part,
                "text_sha256": sha256_hex(part.encode("utf-8")),
                "word_count": len(part.split()),
                "extraction_notes": notes,
                "attribution": record["attribution"],
            })
    return chunks


def build_corpus_chunks(documents: list[Document], records: dict, repo_root: Path) -> list[dict]:
    """Extract and chunk every verified snapshot, in manifest order."""
    chunks = []
    for document in documents:
        record = records[document.id]
        raw = validate_local_path(document.local_path, repo_root).read_bytes()
        if sha256_hex(raw) != record["sha256"]:
            raise CorpusError(f"{document.id}: snapshot changed after capture")
        page = parse_page(raw.decode(record["charset"]), document.title)
        section = section_number(document.title)
        chunks.extend(make_chunks(document, record, split_units(page.blocks, section), section))
    return chunks


def chunks_jsonl(chunks: list[dict]) -> bytes:
    """Canonical JSONL bytes: one compact JSON object per line, sorted keys, LF."""
    return b"".join(
        json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for chunk in chunks
    )


def read_chunks(path: Path) -> list[dict]:
    """Read frozen chunks and check their fields and text hashes."""
    chunks = []
    for number, line in enumerate(Path(path).read_bytes().decode("utf-8").splitlines(), start=1):
        chunk = json.loads(line)
        if not isinstance(chunk, dict) or any(key not in chunk for key in _CHUNK_KEYS):
            raise CorpusError(f"{Path(path).name} line {number}: missing chunk fields")
        if sha256_hex(chunk["text"].encode("utf-8")) != chunk["text_sha256"]:
            raise CorpusError(f"{Path(path).name} line {number}: text does not match text_sha256")
        chunks.append(chunk)
    return chunks
