from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from repo_paths import find_repo_root


ROOT = find_repo_root()
SCRIPTS = ROOT / "scripts"
REQUIREMENT_ALIASES = {
    "sklearn": "scikit-learn",
}
FORBIDDEN_TRACKED_PARTS = {
    "__pycache__",
    "data/raw",
    "data/interim",
    "data/processed",
    "outputs",
    "logs",
    "debug",
    "scratch",
    "tmp",
    "render",
    "manuscript",
    "figures",
}
FORBIDDEN_TRACKED_SUFFIXES = {
    ".docx",
    ".pdf",
    ".png",
    ".svg",
    ".zip",
    ".pyc",
    ".pyo",
}
MANIFEST_FILES = {"CODE_MANIFEST.json", "SHA256SUMS.txt"}
ENUMERATION_IGNORES = {
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


def requirement_names() -> set[str]:
    names: set[str] = set()
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line
        for token in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(token, 1)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def tracked_files() -> list[str]:
    if (ROOT / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        output = [
            item for item in completed.stdout.decode("utf-8").split("\0") if item
        ]
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        output.extend(
            item for item in untracked.stdout.decode("utf-8").split("\0") if item
        )
        return sorted(set(output))
    output = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in ENUMERATION_IGNORES for part in relative.parts):
            continue
        if relative.as_posix() == "artifacts/provenance/local_export_report.json":
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        output.append(relative.as_posix())
    return sorted(output)


def file_bytes(relative: str) -> bytes | None:
    path = ROOT / relative
    if not path.is_file():
        return None
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the code and released reproducibility artifacts."
    )
    parser.add_argument(
        "--skip-cli-help",
        action="store_true",
        help="Skip importing every CLI; useful for the dependency-light quick check.",
    )
    args = parser.parse_args()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    required = [
        "README.md",
        "LICENSE",
        "THIRD_PARTY_LICENSES.md",
        "requirements.txt",
        "requirements-lock.txt",
        ".env.example",
        "config/kg_resources.json",
        "config/dataset_versions.json",
        "config/experiment_config.json",
        "docs/PIPELINE.md",
        "docs/DATA_REQUIREMENTS.md",
        "docs/THIRD_PARTY_DATA.md",
        "docs/PAPER_ARTIFACT_MAP.md",
        "reproduce_quick.py",
        "reproduce_frozen_results.py",
        "reproduce_full_pipeline.py",
        "artifacts/predictions/formal600_predictions.jsonl",
        "artifacts/predictions/risk_routing_scores.csv",
        "artifacts/data_splits/formal600_membership.csv",
        "artifacts/data_splits/claim_component_map.csv",
        "artifacts/data_splits/formal600_inner_folds.csv",
        "artifacts/data_splits/scorer_exclusion_manifest.json",
        "artifacts/api_manifest/response_file_hashes.json",
        "artifacts/provenance/feature_provenance.json",
    ]
    checks["required_files_present"] = all((ROOT / item).is_file() for item in required)

    scripts = sorted(SCRIPTS.glob("*.py"))
    syntax_errors: list[str] = []
    imported_third_party: set[str] = set()
    cli_scripts: list[Path] = []
    local_modules = {path.stem for path in scripts}
    for path in scripts:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            syntax_errors.append(f"{path.name}: {exc}")
            continue
        if path.resolve() != Path(__file__).resolve() and "ArgumentParser(" in source:
            cli_scripts.append(path)
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in sys.stdlib_module_names and root not in local_modules:
                        imported_third_party.add(root)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                module = node.module.split(".", 1)[0]
                if module not in sys.stdlib_module_names and module not in local_modules:
                    imported_third_party.add(module)
    checks["all_python_files_parse"] = not syntax_errors
    details["syntax_errors"] = syntax_errors

    requirements = requirement_names()
    missing_requirements = sorted(
        module
        for module in imported_third_party
        if REQUIREMENT_ALIASES.get(module, module).lower().replace("_", "-")
        not in requirements
    )
    checks["third_party_imports_declared"] = not missing_requirements
    details["missing_requirements"] = missing_requirements

    help_failures: list[str] = []
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if not args.skip_cli_help:
        for path in cli_scripts:
            completed = subprocess.run(
                [sys.executable, str(path), "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                env=subprocess_env,
            )
            if completed.returncode != 0:
                help_failures.append(
                    f"{path.name}: {completed.stderr.strip() or completed.stdout.strip()}"
                )
    checks["all_cli_help_commands_succeed"] = not help_failures
    details["help_failures"] = help_failures
    details["cli_scripts_checked"] = len(cli_scripts)

    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    checks["deepseek_endpoint_consistent"] = (
        "OPENAI_BASE_URL=https://api.deepseek.com" in env_text
        and "OPENAI_MODEL=deepseek-v4-flash" in env_text
    )

    tracked = tracked_files()
    forbidden: list[str] = []
    for item in tracked:
        normalized = item.replace("\\", "/").lower()
        parts = normalized.split("/")
        if Path(normalized).suffix in FORBIDDEN_TRACKED_SUFFIXES:
            forbidden.append(item)
            continue
        if "__pycache__" in parts or any(
            normalized.startswith(marker + "/")
            for marker in FORBIDDEN_TRACKED_PARTS
            if marker != "__pycache__"
        ):
            forbidden.append(item)
    checks["no_generated_or_manuscript_artifacts_tracked"] = not forbidden
    details["forbidden_tracked_files"] = forbidden

    manifest = json.loads((ROOT / "CODE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_rows = {row["path"]: row for row in manifest.get("files", [])}
    manifest_mismatches: list[str] = []
    for relative, row in manifest_rows.items():
        data = file_bytes(relative)
        if data is None:
            manifest_mismatches.append(f"missing:{relative}")
            continue
        if len(data) != int(row["bytes"]):
            manifest_mismatches.append(f"bytes:{relative}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != row["sha256"]:
            manifest_mismatches.append(f"sha256:{relative}")
    expected_tracked = {item for item in tracked if item not in MANIFEST_FILES}
    checks["code_manifest_hashes_match"] = not manifest_mismatches
    checks["code_manifest_covers_tracked_files"] = set(manifest_rows) == expected_tracked
    details["manifest_mismatches"] = manifest_mismatches
    details["manifest_missing_tracked"] = sorted(expected_tracked - set(manifest_rows))
    details["manifest_untracked_entries"] = sorted(set(manifest_rows) - expected_tracked)

    sums_path = ROOT / "SHA256SUMS.txt"
    expected_sums = {
        row["path"]: row["sha256"] for row in manifest.get("files", [])
    }
    actual_sums: dict[str, str] = {}
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            actual_sums[relative] = digest
    checks["sha256sums_matches_code_manifest"] = actual_sums == expected_sums

    forbidden_identity_patterns = {
        "public_github_owner": "yao" + "yaot",
        "public_repository_url": "github.com/" + "yao" + "yaot/",
        "local_windows_user": "\\\\users\\\\" + "len" + "ovo",
        "local_workspace": "e:\\\\" + "knowledge graph\\\\code",
    }
    identity_hits: list[str] = []
    for relative in tracked:
        path = ROOT / relative
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".jsonl",
            ".csv",
            ".yml",
            ".yaml",
            ".example",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_identity_patterns.items():
            if pattern in text:
                identity_hits.append(f"{name}:{relative}")
    checks["no_known_identity_or_absolute_path_leaks"] = not identity_hits
    details["identity_or_path_hits"] = identity_hits

    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "details": details,
        "scripts_checked": len(scripts),
        "python": sys.version.split()[0],
    }
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
