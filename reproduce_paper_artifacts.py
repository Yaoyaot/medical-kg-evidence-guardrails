from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*parts: str) -> None:
    subprocess.run([sys.executable, *parts], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute frozen point estimates, rebuild manuscript-facing "
            "Tables 1–7/S1–S6 and Figures 3–7, and verify key paper values."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reproduced",
    )
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.is_absolute():
        output_path = output
    else:
        output_path = ROOT / output
    tables = output_path / "paper_tables"
    figures = output_path / "paper_figures"

    run(
        "reproduce_frozen_results.py",
        "--skip-render",
        "--output-dir",
        str(output_path),
    )
    run(
        "scripts/tables/build_paper_tables.py",
        "--output-dir",
        str(tables),
    )
    run(
        "scripts/figures/build_paper_figures.py",
        "--output-dir",
        str(figures),
        "--dpi",
        str(args.dpi),
    )
    run(
        "scripts/verify_paper_artifacts.py",
        "--tables-dir",
        str(tables),
        "--figures-dir",
        str(figures),
    )
    print(
        "Paper-artifact reproduction passed without network or API calls. "
        f"Outputs: {output_path}"
    )


if __name__ == "__main__":
    main()
