from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests


API_BASE = "https://uts-ws.nlm.nih.gov/rest"
SEARCH_TYPES = {"Anatomy", "Compound", "Disease", "Pharmacologic Class"}
SKIP_TYPES = {"Gene", "Biological Process", "Cellular Component", "Molecular Function", "Pathway"}
ENGLISH_ALIAS_SOURCES = {
    "CHV",
    "CSP",
    "DRUGBANK",
    "FMA",
    "GS",
    "HPO",
    "ICD10",
    "ICD10CM",
    "ICD9CM",
    "LNC",
    "MDR",
    "MEDCIN",
    "MEDLINEPLUS",
    "MSH",
    "MTH",
    "MTHSPL",
    "NCI",
    "NDFRT",
    "OMIM",
    "RCD",
    "RXNORM",
    "SNM",
    "SNMI",
    "SNOMEDCT_US",
    "UWDA",
    "VANDF",
}


class UmlsRequestError(RuntimeError):
    def __init__(self, path: str, error_type: str, retryable: bool) -> None:
        super().__init__(f"UMLS request failed for {path}: {error_type}")
        self.retryable = retryable


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


class UmlsClient:
    def __init__(self, api_key: str, cache_dir: Path, requests_per_second: float, max_retries: int) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max(requests_per_second, 0.1)
        self.max_retries = max_retries
        self.last_request = 0.0
        self.stats = Counter()
        self.rate_lock = threading.Lock()
        self.content_versions: set[str] = set()

    def record_content_versions(self, payload: dict) -> None:
        serialized = json.dumps(payload, ensure_ascii=False)
        self.content_versions.update(re.findall(r"/content/([^/\"?]+)", serialized))

    def discover_cached_content_versions(self) -> None:
        for cache_path in self.cache_dir.glob("*.json"):
            text = cache_path.read_text(encoding="utf-8")
            self.content_versions.update(re.findall(r"/content/([^/\"?]+)", text))

    def get(self, path: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        cache_payload = {"path": path, "params": sorted(params.items())}
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            self.stats["cache_hits"] += 1
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.record_content_versions(payload)
            return payload

        request_params = {**params, "apiKey": self.api_key}
        last_error = None
        for attempt in range(self.max_retries):
            with self.rate_lock:
                wait = self.min_interval - (time.time() - self.last_request)
                if wait > 0:
                    time.sleep(wait)
                self.last_request = time.time()
            try:
                response = requests.get(f"{API_BASE}{path}", params=request_params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                self.record_content_versions(payload)
                self.stats["network_requests"] += 1
                return payload
            except Exception as exc:
                last_error = exc
                self.stats["request_retries"] += 1
                if isinstance(exc, requests.HTTPError):
                    status = exc.response.status_code if exc.response is not None else None
                    if status in {400, 401, 403, 404}:
                        break
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        self.stats["request_failures"] += 1
        if isinstance(last_error, requests.HTTPError):
            status = last_error.response.status_code if last_error.response is not None else None
            retryable = status not in {400, 401, 403, 404}
            raise UmlsRequestError(path, f"HTTPError(status={status})", retryable)
        error_type = type(last_error).__name__ if last_error else "UnknownError"
        raise UmlsRequestError(path, error_type, True)

    def search(self, name: str, sabs: str | None = None) -> list[dict]:
        params = {"string": name, "searchType": "exact", "pageSize": 10}
        if sabs:
            params["sabs"] = sabs
        payload = self.get("/search/current", params)
        return payload.get("result", {}).get("results", [])

    def atoms(self, cui: str, max_atoms: int) -> list[dict]:
        payload = self.get(f"/content/current/CUI/{cui}/atoms", {"pageSize": max_atoms})
        return payload.get("result", [])


def node_cui(node: dict, client: UmlsClient) -> str | None:
    node_id = node["id"]
    kind = node["kind"]
    if kind == "Side Effect":
        suffix = node_id.split("::", 1)[-1]
        return suffix if re.fullmatch(r"C\d+", suffix) else None
    if kind == "Symptom":
        results = client.search(node["name"], sabs="MSH")
    elif kind in SEARCH_TYPES:
        results = client.search(node["name"])
    else:
        return None
    expected = normalize_text(node["name"])
    for result in results:
        cui = str(result.get("ui", ""))
        if cui.startswith("C") and normalize_text(str(result.get("name", ""))) == expected:
            return cui
    return None


def alias_row(alias: str, node: dict, cui: str, source: str) -> dict | None:
    normalized = normalize_text(alias)
    if not normalized:
        return None
    return {
        "alias": alias.strip(),
        "normalized_alias": normalized,
        "node_id": node["id"],
        "kind": node["kind"],
        "alias_source": source,
        "external_id": cui,
    }


def aliases_for_node(node: dict, client: UmlsClient, max_atoms: int) -> tuple[list[dict], str | None]:
    cui = node_cui(node, client)
    if not cui:
        return [], None
    rows = []
    for atom in client.atoms(cui, max_atoms):
        name = str(atom.get("name", "")).strip()
        if not name:
            continue
        row = alias_row(name, node, cui, f"umls:{atom.get('rootSource', '') or 'unknown'}")
        if row:
            rows.append(row)
    return rows, cui


def dedupe(rows: list[dict]) -> list[dict]:
    output = {}
    for row in rows:
        key = (row["node_id"], row["normalized_alias"])
        output.setdefault(key, row)
    return sorted(output.values(), key=lambda row: (row["node_id"], row["normalized_alias"], row["alias_source"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UMLS-derived aliases for an expanded Hetionet graph.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--private-dir", default=None)
    parser.add_argument("--requests-per-second", type=float, default=8.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=50)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("UMLS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing UMLS_API_KEY environment variable.")

    data_dir = Path(args.data_dir)
    graph_dir = Path(args.graph_dir) if args.graph_dir else data_dir / "processed" / "stage1_umls" / "expanded_graph"
    processed_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed" / "stage1_umls"
    private_dir = Path(args.private_dir) if args.private_dir else data_dir / "private"
    alias_path = private_dir / "stage1_umls" / "umls_aliases.jsonl"
    progress_path = private_dir / "stage1_umls" / "umls_progress.jsonl"
    cache_dir = private_dir / "umls_api_cache"
    client = UmlsClient(api_key, cache_dir, args.requests_per_second, args.max_retries)

    nodes = list(read_tsv(graph_dir / "hetionet_nodes.tsv"))
    eligible = [node for node in nodes if node["kind"] not in SKIP_TYPES]
    if args.max_nodes is not None:
        eligible = eligible[: args.max_nodes]

    completed = {row["node_id"]: row for row in read_jsonl(progress_path)} if args.resume else {}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    def process_node(node: dict) -> dict:
        try:
            rows, cui = aliases_for_node(node, client, args.max_atoms)
            return {"node_id": node["id"], "kind": node["kind"], "cui": cui, "aliases": rows, "error": None}
        except Exception as exc:
            return {
                "node_id": node["id"],
                "kind": node["kind"],
                "cui": None,
                "aliases": [],
                "error": str(exc),
                "retryable": getattr(exc, "retryable", True),
            }

    pending = [
        node
        for node in eligible
        if node["id"] not in completed
        or (
            completed[node["id"]].get("error")
            and completed[node["id"]].get("retryable", True)
        )
    ]
    with progress_path.open(mode, encoding="utf-8") as progress_file, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_node, node): node for node in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            node = futures[future]
            try:
                progress = future.result()
            except Exception as exc:
                progress = {
                    "node_id": node["id"],
                    "kind": node["kind"],
                    "cui": None,
                    "aliases": [],
                    "error": str(exc),
                    "retryable": getattr(exc, "retryable", True),
                }
            progress_file.write(json.dumps(progress, ensure_ascii=False) + "\n")
            progress_file.flush()
            completed[node["id"]] = progress
            if index % 100 == 0:
                print(json.dumps({"completed_in_run": index, "pending_at_start": len(pending), "eligible": len(eligible), "cache": dict(client.stats)}), flush=True)

    completed = {row["node_id"]: row for row in read_jsonl(progress_path)}
    raw_aliases = [alias for row in completed.values() for alias in row.get("aliases", [])]
    filtered_aliases = [
        alias
        for alias in raw_aliases
        if alias["alias_source"].split(":", 1)[-1] in ENGLISH_ALIAS_SOURCES
    ]
    aliases = dedupe(filtered_aliases)
    write_jsonl(alias_path, aliases)
    processed_dir.mkdir(parents=True, exist_ok=True)
    successful_cached_requests = sum(1 for _ in cache_dir.glob("*.json"))
    current_run_requests = client.stats["cache_hits"] + client.stats["network_requests"]
    stats = {
        "eligible_nodes": len(eligible),
        "completed_nodes": sum(node["id"] in completed for node in eligible),
        "mapped_nodes": sum(bool(completed.get(node["id"], {}).get("cui")) for node in eligible),
        "failed_nodes": sum(bool(completed.get(node["id"], {}).get("error")) for node in eligible),
        "raw_aliases": len(raw_aliases),
        "aliases": len(aliases),
        "filtered_non_english_source_aliases": len(raw_aliases) - len(filtered_aliases),
        "deduplicated_aliases": len(filtered_aliases) - len(aliases),
        "alias_source_counts": dict(Counter(row["alias_source"] for row in aliases)),
        "successful_cached_requests": successful_cached_requests,
        "progress_records": count_lines(progress_path),
        "cache_hit_rate_current_run": round(client.stats["cache_hits"] / max(current_run_requests, 1), 4),
        "cache_stats": dict(client.stats),
        "output": str(alias_path),
    }
    stats_path = processed_dir / "umls_alias_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    client.discover_cached_content_versions()
    manifest = {
        "umls_content_version": sorted(client.content_versions) or ["current"],
        "api_base": API_BASE,
        "requests_per_second": args.requests_per_second,
        "max_atoms": args.max_atoms,
        "stats": stats,
    }
    manifest_path = processed_dir / "umls_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stats": stats, "manifest": str(manifest_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
