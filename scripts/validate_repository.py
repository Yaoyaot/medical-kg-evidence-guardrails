from __future__ import annotations

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
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [item for item in completed.stdout.decode("utf-8").split("\0") if item]


def main() -> None:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    required = [
        "README.md",
        "requirements.txt",
        ".env.example",
        "config/kg_resources.json",
        "docs/PIPELINE.md",
        "docs/DATA_REQUIREMENTS.md",
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
        if any(
            marker in parts or normalized.startswith(marker + "/")
            for marker in FORBIDDEN_TRACKED_PARTS
        ):
            forbidden.append(item)
    checks["no_generated_or_manuscript_artifacts_tracked"] = not forbidden
    details["forbidden_tracked_files"] = forbidden

    manifest = json.loads((ROOT / "CODE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_rows = {row["path"]: row for row in manifest.get("files", [])}
    manifest_mismatches: list[str] = []
    for relative, row in manifest_rows.items():
        blob = subprocess.run(
            ["git", "show", f":{relative}"],
            cwd=ROOT,
            capture_output=True,
        )
        if blob.returncode != 0:
            manifest_mismatches.append(f"missing:{relative}")
            continue
        data = blob.stdout
        if len(data) != int(row["bytes"]):
            manifest_mismatches.append(f"bytes:{relative}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != row["sha256"]:
            manifest_mismatches.append(f"sha256:{relative}")
    expected_tracked = {item for item in tracked if item != "CODE_MANIFEST.json"}
    checks["code_manifest_hashes_match"] = not manifest_mismatches
    checks["code_manifest_covers_tracked_files"] = set(manifest_rows) == expected_tracked
    details["manifest_mismatches"] = manifest_mismatches
    details["manifest_missing_tracked"] = sorted(expected_tracked - set(manifest_rows))
    details["manifest_untracked_entries"] = sorted(set(manifest_rows) - expected_tracked)

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
