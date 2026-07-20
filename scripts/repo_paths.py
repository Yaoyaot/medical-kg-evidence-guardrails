from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root without assuming a fixed script depth."""
    start = (start or Path(__file__)).resolve()
    directory = start if start.is_dir() else start.parent
    for candidate in (directory, *directory.parents):
        has_public_layout = (
            (candidate / "scripts").is_dir()
            and (candidate / "requirements.txt").exists()
        )
        has_frozen_package_layout = (candidate / "SHA256_MANIFEST.json").exists()
        if (candidate / "README.md").exists() and (
            has_public_layout or has_frozen_package_layout
        ):
            return candidate
    raise RuntimeError(f"Could not locate repository root above {directory}")
