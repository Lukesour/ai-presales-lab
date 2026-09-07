#!/usr/bin/env python3
"""Build deterministic, synthetic conversational data for the QLoRA POC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ai_presales_lab.agent import PresalesAgent
from ai_presales_lab.evaluation import load_cases
from ai_presales_lab.finetuning import dataset_stats, validate_conversation, write_manifest
from ai_presales_lab.knowledge import KnowledgeBase
from ai_presales_lab.persistence import CheckpointStore

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "你是企业 AI 解决方案售前顾问。请基于客户输入和已提供证据输出严谨的结构化方案。"
    "不要编造价格、SLA、准确率、认证或容量；资料不足时提出澄清问题。"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/finetuning")
    parser.add_argument("--variants", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(ROOT / "data/evaluation/cases.jsonl")
    examples = _build_examples(cases, args.variants)
    for item in examples:
        validate_conversation(item)

    # Split by source case, so variants of one customer brief never leak across splits.
    buckets = {"train": [], "dev": [], "test": []}
    for item in examples:
        source_case = item["metadata"]["case_id"]
        stable_bucket = _stable_bucket(source_case)
        bucket = "test" if stable_bucket < 2 else "dev" if stable_bucket < 4 else "train"
        buckets[bucket].append(item)

    files: dict[str, Path] = {}
    for name, rows in buckets.items():
        path = output_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        files[name] = path

    llama_dir = output_dir / "llamafactory"
    llama_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = {
        name: {
            "file_name": str(path.relative_to(output_dir)),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        for name, path in files.items()
    }
    (llama_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(
        output_dir / "manifest.json",
        files=files,
        metadata={
            "generator": "scripts/build_finetune_dataset.py",
            "system_prompt_version": "v1",
            "variants": args.variants,
            "source_cases": len(cases),
            "split_policy": "deterministic case-level split: test/dev/train",
            "stats": {name: dataset_stats(rows) for name, rows in buckets.items()},
        },
    )
    print(json.dumps({name: dataset_stats(rows) for name, rows in buckets.items()}, ensure_ascii=False, indent=2))
    return 0


def _build_examples(cases: list[Any], variants: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    knowledge_base = KnowledgeBase(ROOT / "data/knowledge")
    for case in cases:
        with CheckpointStore(":memory:") as store:
            state = PresalesAgent(knowledge_base, store).run(
                case, thread_id=f"dataset:{case.case_id}"
            )
            if state.response is None:
                raise RuntimeError(f"Agent did not produce a response for {case.case_id}")
            answer = json.dumps(_training_target(state.response.to_dict()), ensure_ascii=False, sort_keys=True)
        prompts = [
            case.raw_request,
            f"请先提取约束，再为以下客户设计 POC：{case.raw_request}",
            f"请审慎检查风险后回答，不要做未经证据支持的承诺：{case.raw_request}",
        ][:variants]
        for index, prompt in enumerate(prompts, start=1):
            examples.append(
                {
                    "id": f"{case.case_id}-v{index}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {"case_id": case.case_id, "variant": index},
                }
            )
    return examples


def _stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 10


def _training_target(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove run-specific observability fields from the supervised target."""

    target = dict(payload)
    for field in ("run_id", "trace_id", "latency_ms", "usage", "model_name"):
        target.pop(field, None)
    return target


if __name__ == "__main__":
    raise SystemExit(main())
