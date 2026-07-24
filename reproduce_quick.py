from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*parts: str) -> None:
    completed = subprocess.run([sys.executable, *parts], cwd=ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    run("scripts/validate_repository.py", "--skip-cli-help")
    run("scripts/run_minimal_example.py")
    print("Quick validation passed: repository, manifests, and synthetic rules.")


if __name__ == "__main__":
    main()
