#!/usr/bin/env python3
"""Validate the generated conversational dataset and print its manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_presales_lab.finetuning import dataset_stats, load_conversations, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/finetuning")
    args = parser.parse_args()
    rows: dict[str, object] = {}
    split_paths: dict[str, Path] = {}
    for split in ("train", "dev", "test"):
        path = args.directory / f"{split}.jsonl"
        if not path.exists():
            print(f"Missing {path}; run make build-finetune-dataset first.")
            return 2
        examples = load_conversations(path)
        rows[split] = dataset_stats(examples)
        split_paths[split] = path
    manifest = args.directory / "manifest.json"
    if not manifest.exists():
        print(f"Missing {manifest}; regenerate the dataset to keep hashes and stats traceable.")
        return 2
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    for split, path in split_paths.items():
        entry = manifest_payload.get("files", {}).get(split)
        if not isinstance(entry, dict):
            print(f"Manifest entry missing for {split}.")
            return 3
        expected_path = (args.directory / entry["path"]).resolve()
        if not expected_path.is_relative_to(args.directory.resolve()) or expected_path != path.resolve():
            print(f"Manifest path mismatch for {split}: {entry['path']}")
            return 3
        actual_hash = sha256_file(path)
        if actual_hash != entry.get("sha256"):
            print(f"Manifest hash mismatch for {split}: expected {entry.get('sha256')}, got {actual_hash}")
            return 3
        expected_examples = manifest_payload.get("metadata", {}).get("stats", {}).get(split, {}).get("examples")
        if expected_examples != rows[split]["examples"]:
            print(f"Manifest example count mismatch for {split}.")
            return 3
    dataset_info = args.directory / "llamafactory/dataset_info.json"
    if not dataset_info.exists():
        print(f"Missing {dataset_info}; LLaMA Factory import metadata is incomplete.")
        return 3
    rows["manifest"] = manifest_payload
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
