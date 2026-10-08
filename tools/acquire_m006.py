"""Download the M006 wheels and model files pinned in the download lock.

Usage, from the repository root:

    python tools/acquire_m006.py --downloads config/m006_downloads.json \
        --model-lock config/m006_model.json --output-dir data/m006

Standard library only. Every request and every redirect must be HTTPS on a host of
the lock's allowlist, and nothing authenticated is sent. Each file is streamed to a
new temporary file under ``<output-dir>/tmp``, measured and hashed, and promoted to
its final path only when size and SHA-256 match the lock. Final files are never
overwritten. A purely transport failure (no HTTP status received) may be retried
once with the same URL; an HTTP error, a host outside the allowlist or a size or
hash mismatch is never retried and stops the acquisition. Incomplete temporaries
stay in place and are listed in the log; nothing is cleaned globally.

The log keeps the fixed origin URL from the lock, redirect hosts, status codes,
bytes and hashes; it never stores signed redirect URLs or headers. This tool runs
no Git, pip, package or model code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
M006_ROOT = REPO_ROOT / "data" / "m006"
USER_AGENT = "TraceRAG-M006-acquire/1.0 (Python-urllib; pinned public downloads only)"
TIMEOUT_SECONDS = 60
BLOCK_BYTES = 1024 * 1024
_WHEEL_RE = re.compile(r"[A-Za-z0-9._+-]+\.whl")
_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]+")


class AcquisitionError(Exception):
    """The lock, a destination, a response or a downloaded file violates the contract."""


class TransportError(AcquisitionError):
    """The transfer failed before a complete HTTP response was read."""


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def contained(base: Path, target: Path) -> Path:
    """Resolve ``target`` (following links and junctions) and require it inside ``base``."""
    root = base.resolve()
    resolved = target.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise AcquisitionError(f"{target} resolves outside {root}")
    return resolved


def check_url(url: str, allowed_hosts) -> str:
    """Return the host of an HTTPS URL on the allowlist, without credentials or odd ports."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or host not in allowed_hosts:
        raise AcquisitionError(f"{parts.scheme}://{host} is not an allowed HTTPS host")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise AcquisitionError(f"URL for {host} carries credentials or a non-default port")
    return host


def check_relative_path(path: str) -> list[str]:
    """Split a lock path such as ``1_Pooling/config.json`` into safe segments."""
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        raise AcquisitionError(f"unsafe model path {path!r}")
    segments = path.split("/")
    for segment in segments:
        if segment in (".", "..") or not _SEGMENT_RE.fullmatch(segment):
            raise AcquisitionError(f"unsafe model path {path!r}")
    return segments


def load_plan(downloads_path: Path, model_lock_path: Path) -> dict:
    """Validate both locks and return the ordered list of files to acquire."""
    downloads = json.loads(Path(downloads_path).read_bytes().decode("utf-8"))
    model_lock = json.loads(Path(model_lock_path).read_bytes().decode("utf-8"))
    if downloads.get("schema_version") != 1 or model_lock.get("schema_version") != 1:
        raise AcquisitionError("both locks must have schema_version 1")
    if downloads.get("allow_authenticated_requests") is not False:
        raise AcquisitionError("the download lock must forbid authenticated requests")
    if downloads.get("allow_redirects_only_within_allowlist") is not True:
        raise AcquisitionError("the download lock must restrict redirects to its allowlist")
    hosts = downloads.get("allowed_https_hosts")
    if not isinstance(hosts, list) or not hosts or any(not isinstance(h, str) or h != h.lower() for h in hosts):
        raise AcquisitionError("allowed_https_hosts must be a non-empty list of lowercase host names")
    allowed = frozenset(hosts)
    items = []
    for package in downloads.get("packages") or []:
        name = package.get("filename")
        if not isinstance(name, str) or not _WHEEL_RE.fullmatch(name):
            raise AcquisitionError(f"unsafe wheel file name {name!r}")
        url = package["url"]
        check_url(url, allowed)
        if unquote(urlsplit(url).path.rsplit("/", 1)[-1]) != name:
            raise AcquisitionError(f"{name}: URL does not end with the wheel file name")
        items.append({"kind": "wheel", "name": name, "relative": ["wheels", name], "url": url,
                      "size": package["size_bytes"], "sha256": package["sha256"]})
    model_id, revision = model_lock.get("model_id"), model_lock.get("revision")
    locked = {entry["path"]: entry for entry in model_lock.get("files") or []}
    listed = downloads.get("model_files") or []
    if sorted(locked) != sorted(entry.get("path") for entry in listed):
        raise AcquisitionError("model files differ between the download lock and the model lock")
    for entry in listed:
        path = entry["path"]
        segments = check_relative_path(path)
        if (entry["size_bytes"], entry["sha256"]) != (locked[path]["size_bytes"], locked[path]["sha256"]):
            raise AcquisitionError(f"{path}: size or SHA-256 differs between the locks")
        url = entry["url"]
        check_url(url, allowed)
        if urlsplit(url).path != f"/{model_id}/resolve/{revision}/{path}":
            raise AcquisitionError(f"{path}: URL is not pinned to {model_id}@{revision}")
        items.append({"kind": "model", "name": path, "relative": ["model", *segments], "url": url,
                      "size": entry["size_bytes"], "sha256": entry["sha256"]})
    if model_lock.get("local_directory") != "data/m006/model":
        raise AcquisitionError("model lock local_directory must be data/m006/model")
    expected = downloads.get("expected_download_bytes") or {}
    for kind, key in (("wheel", "packages"), ("model", "model")):
        total = sum(item["size"] for item in items if item["kind"] == kind)
        if total != expected.get(key):
            raise AcquisitionError(f"{key}: listed sizes sum to {total}, lock says {expected.get(key)}")
    for item in items:
        if not isinstance(item["size"], int) or item["size"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise AcquisitionError(f"{item['name']}: invalid size or SHA-256 in the lock")
    return {"allowed_hosts": sorted(allowed), "items": items, "model_id": model_id, "revision": revision}


class AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an HTTPS URL on the allowlist; record host and status only."""

    def __init__(self, allowed_hosts, chain: list) -> None:
        super().__init__()
        self.allowed_hosts = frozenset(allowed_hosts)
        self.chain = chain

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = (urlsplit(newurl).hostname or "").lower()
        self.chain.append({"status": code, "to_host": host})
        check_url(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def default_opener(allowed_hosts, chain: list):
    context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), AllowlistRedirectHandler(allowed_hosts, chain)
    )


def fetch_to_temp(item: dict, allowed_hosts, temp_path: Path, opener_factory=default_opener) -> dict:
    """Stream one URL into ``temp_path`` (created exclusively) and return what was observed."""
    chain: list = []
    observed = {"redirects": chain, "status": None, "final_host": None, "bytes": 0, "sha256": None}
    request = urllib.request.Request(item["url"], headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    digest = hashlib.sha256()
    with open(temp_path, "xb") as out:
        try:
            with opener_factory(allowed_hosts, chain).open(request, timeout=TIMEOUT_SECONDS) as response:
                observed["status"] = response.status
                observed["final_host"] = check_url(response.geturl(), allowed_hosts)
                encoding = (response.headers.get("Content-Encoding") or "identity").lower()
                if encoding != "identity":
                    raise AcquisitionError(f"unexpected Content-Encoding {encoding!r}")
                length = response.headers.get("Content-Length")
                observed["content_length"] = int(length) if length and length.isdigit() else None
                while True:
                    block = response.read(BLOCK_BYTES)
                    if not block:
                        break
                    observed["bytes"] += len(block)
                    if observed["bytes"] > item["size"]:
                        raise AcquisitionError("response is larger than the locked size")
                    digest.update(block)
                    out.write(block)
        except urllib.error.HTTPError as exc:
            observed["status"] = exc.code
            raise AcquisitionError(f"HTTP {exc.code}") from exc
        except AcquisitionError:
            raise
        except (urllib.error.URLError, http.client.HTTPException, OSError, socket.timeout) as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
    observed["sha256"] = digest.hexdigest()
    return observed


def verify_existing(path: Path, item: dict) -> bool:
    data_size = path.stat().st_size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
            digest.update(block)
    return data_size == item["size"] and digest.hexdigest() == item["sha256"]


def acquire_item(item: dict, output_dir: Path, allowed_hosts, opener_factory=default_opener) -> dict:
    """Acquire one locked file; at most one retry, only after a transport failure."""
    final = output_dir.joinpath(*item["relative"])
    final_resolved = contained(output_dir, final)
    record = {"kind": item["kind"], "name": item["name"], "url": item["url"], "expected_bytes": item["size"],
              "expected_sha256": item["sha256"], "final_path": final.relative_to(REPO_ROOT).as_posix()
              if final.is_relative_to(REPO_ROOT) else str(final), "attempts": []}
    if final_resolved.exists():
        if not verify_existing(final_resolved, item):
            raise AcquisitionError(f"{item['name']}: existing final file differs from the lock; not replaced")
        record["result"] = "present_verified"
        return record
    final_resolved.parent.mkdir(parents=True, exist_ok=True)
    contained(output_dir, final_resolved.parent)
    temp_dir = output_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)
    contained(output_dir, temp_dir)
    for attempt in (1, 2):
        safe = item["name"].replace("/", "__")
        temp = temp_dir / f"{safe}.part-{attempt}-{os.getpid()}-{time.time_ns()}-{os.urandom(4).hex()}"
        entry = {"attempt": attempt, "started_at": now(), "temp_path": temp.name}
        record["attempts"].append(entry)
        try:
            observed = fetch_to_temp(item, allowed_hosts, temp, opener_factory)
        except TransportError as exc:
            entry.update({"finished_at": now(), "error": str(exc), "outcome": "transport_failure"})
            if attempt == 1:
                continue
            raise AcquisitionError(f"{item['name']}: transport failed twice; last: {exc}") from exc
        except AcquisitionError as exc:
            entry.update({"finished_at": now(), "error": str(exc), "outcome": "rejected"})
            raise AcquisitionError(f"{item['name']}: {exc}") from exc
        entry.update({"finished_at": now(), **observed})
        if observed["bytes"] != item["size"] or observed["sha256"] != item["sha256"]:
            entry["outcome"] = "size_or_hash_mismatch"
            raise AcquisitionError(f"{item['name']}: size or SHA-256 differs from the lock; not promoted, no retry")
        os.rename(temp, final_resolved)  # fails if the final path already exists
        entry["outcome"] = "promoted"
        record["result"] = "downloaded_verified"
        return record
    raise AcquisitionError(f"{item['name']}: no attempt succeeded")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Acquire the pinned M006 wheels and model files")
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--model-lock", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    started = now()
    log = {"tool": "tools/acquire_m006.py", "user_agent": USER_AGENT, "started_at": started, "items": []}
    status = 1
    log_root = M006_ROOT
    try:
        output_dir = Path(args.output_dir)
        output_dir = contained(M006_ROOT, output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = contained(M006_ROOT, output_dir)
        log_root = output_dir
        plan = load_plan(Path(args.downloads), Path(args.model_lock))
        log.update({"allowed_hosts": plan["allowed_hosts"], "model": f"{plan['model_id']}@{plan['revision']}",
                    "planned_files": len(plan["items"]), "planned_bytes": sum(i["size"] for i in plan["items"])})
        for item in plan["items"]:
            log["items"].append(acquire_item(item, output_dir, plan["allowed_hosts"]))
        log["result"] = "ok"
        status = 0
    except (AcquisitionError, OSError, ValueError, KeyError) as exc:
        log["result"] = "error"
        log["error"] = f"{type(exc).__name__}: {exc}"
    log["finished_at"] = now()
    log["downloaded_bytes"] = sum(
        attempt.get("bytes", 0) for item in log["items"] for attempt in item["attempts"] if attempt.get("outcome") == "promoted"
    )
    data = (json.dumps(log, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        log_dir = contained(M006_ROOT, log_root / "acquisition")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        with open(log_dir / f"acquire-{stamp}-{os.getpid()}.json", "xb") as handle:
            handle.write(data)
    except OSError as exc:
        print(f"could not write the acquisition log: {exc}", file=sys.stderr)
        status = 1
    sys.stdout.buffer.write(data)
    sys.stdout.flush()
    return status


if __name__ == "__main__":
    sys.exit(main())
