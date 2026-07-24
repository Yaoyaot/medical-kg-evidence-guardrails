from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from repo_paths import find_repo_root


ROOT = find_repo_root()
EXCLUDED_FILES = {"CODE_MANIFEST.json", "SHA256SUMS.txt"}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "logs",
    "debug",
    "scratch",
    "tmp",
    "render",
}


def repository_files() -> list[str]:
    if (ROOT / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
        )
        paths = [
            value
            for value in completed.stdout.decode("utf-8").split("\0")
            if value
        ]
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        paths.extend(
            value
            for value in untracked.stdout.decode("utf-8").split("\0")
            if value
        )
    else:
        paths = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file()
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
        ]
    return sorted(
        {
            path.replace("\\", "/")
            for path in paths
            if path.replace("\\", "/") not in EXCLUDED_FILES
            and not any(
                part in EXCLUDED_PARTS for part in Path(path).parts
            )
            and path != "artifacts/provenance/local_export_report.json"
        }
    )


def build() -> tuple[dict, str]:
    rows = []
    for relative in repository_files():
        path = ROOT / relative
        data = path.read_bytes()
        rows.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "scope": "anonymous review code and curated reproducibility artifacts",
        "manifest_self_included": False,
        "sha256sums_self_included": False,
        "files": rows,
    }
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    return manifest, sums


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or check repository hashes.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest, sums = build()
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.check:
        current_manifest = (ROOT / "CODE_MANIFEST.json").read_text(encoding="utf-8")
        current_sums = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8")
        if current_manifest != manifest_text or current_sums != sums:
            raise SystemExit("Repository manifest is stale.")
        print(f"Repository manifest passed for {len(manifest['files'])} files.")
        return
    with (ROOT / "CODE_MANIFEST.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(manifest_text)
    with (ROOT / "SHA256SUMS.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(sums)
    print(f"Wrote repository manifest for {len(manifest['files'])} files.")


if __name__ == "__main__":
    main()
