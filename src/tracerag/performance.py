"""Local time, memory and storage observation for M006 (Windows APIs through ctypes).

``observe-costs`` keeps a light parent process and starts one child process per
backend, in sequence: the BM25 child never imports torch, and the semantic child
loads the model and the persisted embeddings once. Process start, imports, model
load and index preparation are timed apart from the warm query latency. Query
latency comes from two warm-up sweeps and seven measured sweeps over the 24 M006
queries, in file order, with ``perf_counter_ns``; printing, logging and disk access
stay outside the timed region and query embeddings are never cached.

Memory is read with GetProcessMemoryInfo (PROCESS_MEMORY_COUNTERS). The peak working
set covers the whole child process: interpreter, libraries, model, runtime and the
warm-up. It is not the memory of one query, not an exclusive allocation and not GPU
memory. Everything here is a local observation on one machine, not a benchmark.

The resource guard reads GlobalMemoryStatusEx three times, one second apart, and
passes only when the smallest available-memory sample reaches 2147483648 bytes.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

RAM_GUARD_BYTES = 2147483648
GUARD_SAMPLES = 3
GUARD_INTERVAL_SECONDS = 1.0
WARMUP_SWEEPS = 2
MEASURED_SWEEPS = 7
METHOD = (
    "one child process per backend, run in sequence by a light parent; timings with "
    "time.perf_counter_ns; 2 warm-up and 7 measured sweeps over the 24 M006 queries in "
    "file order; per-query medians of the 7 samples and the median of the 7 sweep totals; "
    "memory from GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS of each child process"
)
LIMITATIONS = (
    "Local observation on one Windows machine with other applications open; not a benchmark.",
    "PeakWorkingSetSize is the peak of the whole child process (interpreter, libraries, model, "
    "runtime and warm-up), not the memory of one query.",
    "Both backends run with one thread; the latency of a different thread setting was not measured.",
    "Storage sizes are logical file sizes (sum of file lengths), not allocated disk space.",
)


class ResourceGuardError(ValueError):
    """Available physical memory is below the guard, or the memory API failed."""


class PerformanceError(ValueError):
    """A measurement could not be taken as defined."""


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _require_windows() -> None:
    if os.name != "nt":
        raise PerformanceError("the memory observation uses Windows APIs and needs Windows")


def system_memory() -> dict:
    """Physical memory of the machine from GlobalMemoryStatusEx (bytes)."""
    _require_windows()
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ResourceGuardError(f"GlobalMemoryStatusEx failed (error {ctypes.get_last_error()})")
    return {"timestamp": _now(), "memory_load_percent": status.dwMemoryLoad,
            "total_physical_bytes": status.ullTotalPhys, "available_physical_bytes": status.ullAvailPhys}


def process_memory() -> dict:
    """Working set and its peak for the current process, from GetProcessMemoryInfo (bytes)."""
    _require_windows()
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise PerformanceError(f"GetProcessMemoryInfo failed (error {ctypes.get_last_error()})")
    return {"timestamp": _now(), "working_set_bytes": counters.WorkingSetSize,
            "peak_working_set_bytes": counters.PeakWorkingSetSize,
            "pagefile_usage_bytes": counters.PagefileUsage, "peak_pagefile_usage_bytes": counters.PeakPagefileUsage}


def memory_guard(samples: int = GUARD_SAMPLES, interval: float = GUARD_INTERVAL_SECONDS,
                 minimum: int = RAM_GUARD_BYTES, reader=system_memory, sleep=time.sleep) -> dict:
    """Sample available physical memory and pass only if the smallest sample reaches ``minimum``."""
    readings = []
    for number in range(samples):
        if number:
            sleep(interval)
        readings.append(reader())
    smallest = min(reading["available_physical_bytes"] for reading in readings)
    guard = {"rule": f"min of {samples} samples, {interval:g} s apart, >= {minimum} bytes available",
             "samples": readings, "min_available_physical_bytes": smallest, "passed": smallest >= minimum}
    if not guard["passed"]:
        raise ResourceGuardError(f"available physical memory {smallest} B is below the guard of {minimum} B")
    return guard


def median(values) -> float | int:
    values = list(values)
    if not values:
        raise PerformanceError("median of an empty sample")
    return statistics.median(values)


def summarize(per_query: list[dict], fields: tuple) -> dict:
    """Per-query medians and per-sweep totals (with their median) for each timed field."""
    summary = {}
    for field in fields:
        sweeps = len(per_query[0][field])
        totals = [sum(query[field][sweep] for query in per_query) for sweep in range(sweeps)]
        summary[field] = {
            "per_query_median_ns": {query["id"]: median(query[field]) for query in per_query},
            "sweep_totals_ns": totals,
            "median_sweep_total_ns": median(totals),
            "median_of_query_medians_ns": median(median(query[field]) for query in per_query),
        }
    return summary


def directory_size(path: Path) -> dict:
    """Logical size (sum of file lengths, links not followed) and file count of a tree."""
    path = Path(path)
    if not path.exists():
        return {"path": path.as_posix(), "exists": False, "files": 0, "bytes": 0}
    if path.is_file():
        return {"path": path.as_posix(), "exists": True, "files": 1, "bytes": path.stat().st_size}
    files = total = 0
    for root, _dirs, names in os.walk(path, followlinks=False):
        for name in names:
            files += 1
            total += os.lstat(os.path.join(root, name)).st_size
    return {"path": path.as_posix(), "exists": True, "files": files, "bytes": total}


def storage_report(repo_root: Path) -> dict:
    """Logical sizes of the M006 data and environment, by category (M004 inputs excluded)."""
    m006 = Path(repo_root) / "data" / "m006"
    categories = {
        "wheels": m006 / "wheels", "model_files": m006 / "model", "index": m006 / "index",
        "reproduction": m006 / "reproduction", "results": m006 / "results", "caches": m006 / "cache",
        "process_temp": m006 / "temp", "download_temp": m006 / "tmp", "acquisition_logs": m006 / "acquisition",
        "tests": m006 / "tests", "freeze": m006 / "freeze.json",
    }
    report = {name: directory_size(path) for name, path in categories.items()}
    for item in report.values():
        item["path"] = Path(item["path"]).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    report["data_m006_total"] = directory_size(m006)
    report["data_m006_total"]["path"] = "data/m006"
    report["venv_m006"] = directory_size(Path(repo_root) / "data" / ".venv-m006")
    report["venv_m006"]["path"] = "data/.venv-m006"
    report["note"] = ("logical sizes measured before performance.json is written; data/m004 inputs are not "
                      "counted as new downloads")
    return report


def environment() -> dict:
    import sqlite3

    memory = system_memory() if os.name == "nt" else {}
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "total_physical_bytes": memory.get("total_physical_bytes"), "sqlite": sqlite3.sqlite_version,
    }


def run_child(backend: str, extra_args: list[str], repo_root: Path, env: dict) -> dict:
    """Run one measurement child and return its JSON result with the parent's own observations."""
    command = [sys.executable, "-m", "tracerag", "observe-costs", "--child", backend, *extra_args]
    before = system_memory()
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(command, cwd=repo_root, env=env, capture_output=True, check=False)
    wall_ns = time.perf_counter_ns() - started_ns
    after = system_memory()
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise PerformanceError(f"{backend} child exited with {completed.returncode}: {stderr[-2000:]}")
    result = json.loads(completed.stdout.decode("utf-8"))
    result["parent_observation"] = {
        "command": ["<venv python>", *command[1:]], "exit_code": completed.returncode,
        "wall_time_including_process_start_ns": wall_ns, "system_memory_before": before,
        "system_memory_after": after, "stderr_bytes": len(completed.stderr), "stderr_tail": stderr[-1500:],
    }
    return result


def bm25_child(chunks: list[dict], items: list[dict], index_factory) -> dict:
    """Measure the lexical backend in this process (torch is never imported here)."""
    memory = {"start": process_memory()}
    started = time.perf_counter_ns()
    index = index_factory(chunks)
    prep_ns = time.perf_counter_ns() - started
    memory["after_index"] = process_memory()
    per_query = [{"id": item["id"], "total_ns": []} for item in items]
    for sweep in range(WARMUP_SWEEPS + MEASURED_SWEEPS):
        for item, record in zip(items, per_query):
            t0 = time.perf_counter_ns()
            index.search(item["query"], 3)
            t1 = time.perf_counter_ns()
            if sweep >= WARMUP_SWEEPS:
                record["total_ns"].append(t1 - t0)
    memory["end"] = process_memory()
    return {
        "backend": "bm25", "torch_imported": "torch" in sys.modules,
        "index_preparation_ns": prep_ns, "queries": len(items),
        "warmup_sweeps": WARMUP_SWEEPS, "measured_sweeps": MEASURED_SWEEPS,
        "samples": per_query, "summary": summarize(per_query, ("total_ns",)), "memory": memory,
        "peak_working_set_bytes": memory["end"]["peak_working_set_bytes"],
        "timed_region": "query term extraction, FTS5 MATCH and bm25 ranking of the top 3 (LexicalIndex.search)",
    }


def semantic_child(items: list[dict], load_ml, load_encoder, load_index, rank) -> dict:
    """Measure the semantic backend in this process: one model, persisted embeddings, no query cache."""
    memory = {"start": process_memory()}
    system = {"before_imports": system_memory()}
    t0 = time.perf_counter_ns()
    ml = load_ml()
    imports_ns = time.perf_counter_ns() - t0
    memory["after_imports"] = process_memory()
    system["before_model_load"] = system_memory()
    t0 = time.perf_counter_ns()
    encoder = load_encoder(ml)
    load_ns = time.perf_counter_ns() - t0
    memory["after_model_load"] = process_memory()
    system["after_model_load"] = system_memory()
    t0 = time.perf_counter_ns()
    index = load_index(ml)
    prep_ns = time.perf_counter_ns() - t0
    memory["after_index"] = process_memory()
    per_query = [{"id": item["id"], "encode_ns": [], "rank_ns": [], "total_ns": []} for item in items]
    for sweep in range(WARMUP_SWEEPS + MEASURED_SWEEPS):
        for item, record in zip(items, per_query):
            t0 = time.perf_counter_ns()
            vector, _info = encoder.encode(item["query"], "query")
            t1 = time.perf_counter_ns()
            rank(index, vector, 3)
            t2 = time.perf_counter_ns()
            if sweep >= WARMUP_SWEEPS:
                record["encode_ns"].append(t1 - t0)
                record["rank_ns"].append(t2 - t1)
                record["total_ns"].append(t2 - t0)
    memory["end"] = process_memory()
    system["end"] = system_memory()
    return {
        "backend": "semantic", "imports_ns": imports_ns, "model_load_ns": load_ns,
        "index_preparation_ns": prep_ns, "queries": len(items),
        "warmup_sweeps": WARMUP_SWEEPS, "measured_sweeps": MEASURED_SWEEPS,
        "samples": per_query, "summary": summarize(per_query, ("encode_ns", "rank_ns", "total_ns")),
        "memory": memory, "system_memory": system,
        "peak_working_set_bytes": memory["end"]["peak_working_set_bytes"],
        "threads": encoder.thread_report(), "versions": ml.versions,
        "timed_region": ("encode = tokenization (with the untruncated token count kept for provenance), "
                         "forward pass, mean pooling and L2 normalization of one query; rank = dot products "
                         "with the 15 stored unit vectors and ordering of the top 3"),
    }
