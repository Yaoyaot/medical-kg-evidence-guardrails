from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "SUPPORT": "SUPPORT",
    "CONTRADICT": "REFUTE",
    "NEI": "UNCERTAIN",
    "true": "SUPPORT",
    "false": "REFUTE",
    "mixture": "UNCERTAIN",
    "unproven": "UNCERTAIN",
}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(value: object) -> str:
    label = str(value).strip()
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    upper = label.upper()
    return LABEL_MAP.get(upper, upper)


def prepare_medfact_bench(raw_dir: Path, out_dir: Path, sample_limit: int | None) -> dict:
    path = raw_dir / "medfact_bench" / "train-00000-of-00001.parquet"
    if not path.exists():
        return {"dataset": "medfact_bench", "status": "missing", "path": str(path)}

    df = pd.read_parquet(path)
    rows: list[dict] = []
    for idx, item in df.iterrows():
        if sample_limit is not None and len(rows) >= sample_limit:
            break
        rows.append(
            {
                "id": f"medfact_bench_{idx}",
                "dataset": str(item.get("dataset", "medfact_bench")),
                "claim": str(item.get("claim", "")).strip(),
                "source": str(item.get("source", "")).strip(),
                "label": normalize_label(item.get("label", "")),
                "raw_label": str(item.get("label", "")).strip(),
            }
        )

    out_path = out_dir / "medfact_bench.sample.jsonl"
    write_jsonl(rows, out_path)
    return {
        "dataset": "medfact_bench",
        "status": "ok",
        "rows": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "output": str(out_path),
    }


def prepare_pubhealth(raw_dir: Path, out_dir: Path, sample_limit: int | None) -> dict:
    input_files = [
        ("train", raw_dir / "pubhealth" / "pubhealth-train.parquet"),
        ("validation", raw_dir / "pubhealth" / "pubhealth-validation.parquet"),
        ("test", raw_dir / "pubhealth" / "pubhealth-test.parquet"),
    ]

    summaries: list[dict] = []
    all_rows: list[dict] = []
    for split, path in input_files:
        if not path.exists():
            summaries.append({"split": split, "status": "missing", "path": str(path)})
            continue

        df = pd.read_parquet(path)
        rows: list[dict] = []
        for _, item in df.iterrows():
            if sample_limit is not None and len(rows) >= sample_limit:
                break
            rows.append(
                {
                    "id": f"pubhealth_{split}_{item.get('id', len(rows))}",
                    "dataset": "pubhealth",
                    "split": split,
                    "claim": str(item.get("text_1", "")).strip(),
                    "source": str(item.get("text_2", "")).strip(),
                    "label": normalize_label(item.get("label", "")),
                    "raw_label": str(item.get("label", "")).strip(),
                }
            )
        all_rows.extend(rows)
        summaries.append(
            {
                "split": split,
                "status": "ok",
                "rows": len(rows),
                "labels": dict(Counter(row["label"] for row in rows)),
            }
        )

    out_path = out_dir / "pubhealth.sample.jsonl"
    write_jsonl(all_rows, out_path)
    return {
        "dataset": "pubhealth",
        "status": "ok" if all_rows else "missing",
        "rows": len(all_rows),
        "labels": dict(Counter(row["label"] for row in all_rows)),
        "splits": summaries,
        "output": str(out_path),
    }


def inspect_hetionet(raw_dir: Path) -> dict:
    hetio_dir = raw_dir / "hetionet"
    nodes_path = hetio_dir / "hetionet-v1.0-nodes.tsv"
    edges_path = hetio_dir / "hetionet-v1.0-edges.sif.gz"

    summary = {
        "dataset": "hetionet",
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "nodes_status": "missing",
        "edges_status": "missing",
    }

    if nodes_path.exists():
        with nodes_path.open("r", encoding="utf-8") as f:
            summary["nodes_lines"] = max(sum(1 for _ in f) - 1, 0)
        summary["nodes_status"] = "ok"

    if edges_path.exists():
        with gzip.open(edges_path, "rt", encoding="utf-8") as f:
            summary["edges_lines"] = sum(1 for _ in f)
        summary["edges_status"] = "ok"

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-limit", type=int, default=2000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "hetionet": inspect_hetionet(raw_dir),
        "medfact_bench": prepare_medfact_bench(raw_dir, out_dir, args.sample_limit),
        "pubhealth": prepare_pubhealth(raw_dir, out_dir, args.sample_limit),
    }

    manifest_path = data_dir / "week1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
