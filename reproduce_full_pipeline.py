from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def nonempty_directory(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight a full reconstruction from legally obtained upstream "
            "datasets, graph resources, licensed terminology, and optional "
            "hosted-model access."
        )
    )
    parser.add_argument(
        "--require-umls",
        action="store_true",
        help="Fail unless data/private contains locally licensed UMLS material.",
    )
    parser.add_argument(
        "--require-api",
        action="store_true",
        help="Fail unless OPENAI_API_KEY is set for hosted-model reconstruction.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero exit code when any requested resource is missing.",
    )
    args = parser.parse_args()

    checks = {
        "public_upstream_data_present": nonempty_directory(ROOT / "data/raw"),
        "licensed_umls_material_present": nonempty_directory(
            ROOT / "data/private"
        ),
        "deepseek_api_credential_present": bool(
            os.environ.get("OPENAI_API_KEY", "").strip()
        ),
        "dataset_manifest_present": (ROOT / "config/dataset_versions.json").is_file(),
        "kg_manifest_present": (ROOT / "config/kg_resources.json").is_file(),
    }
    required = ["public_upstream_data_present"]
    if args.require_umls:
        required.append("licensed_umls_material_present")
    if args.require_api:
        required.append("deepseek_api_credential_present")
    missing = [name for name in required if not checks[name]]

    report = {
        "status": "ready_for_documented_full_reconstruction"
        if not missing
        else "missing_external_resources",
        "checks": checks,
        "required_checks": required,
        "missing": missing,
        "network_or_api_call_performed": False,
        "next_steps": [
            "Verify versions and legal access using docs/THIRD_PARTY_DATA.md.",
            "Place resources under data/raw and licensed UMLS under data/private.",
            "Follow the ordered commands and stage descriptions in docs/PIPELINE.md.",
            "Use --dry-run for hosted-model scripts before authorizing API calls.",
            "Compare rebuilt outputs with artifacts/results and manifests.",
        ],
    }
    print(json.dumps(report, indent=2))
    if missing:
        print(
            "\nFull reconstruction was not started. This preflight never "
            "downloads licensed resources or calls a hosted model.",
            file=sys.stderr,
        )
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
