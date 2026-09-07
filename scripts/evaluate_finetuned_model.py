#!/usr/bin/env python3
"""Compare a base model or LoRA adapter on the held-out presales test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_presales_lab.finetuning import load_conversations
from ai_presales_lab.schemas import validate_solution_dict
from ai_presales_lab.security import inspect_output, inspect_sensitive_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=ROOT / "data/finetuning/test.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = load_conversations(args.split)
    if args.limit > 0:
        rows = rows[: args.limit]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"Evaluation dependencies are optional; install the finetune extra: {exc}")
        return 3

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            print(f"PEFT is required to evaluate an adapter: {exc}")
            return 3
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    device = next(model.parameters()).device

    results: list[dict[str, Any]] = []
    for row in rows:
        prompt_messages = [message for message in row["messages"] if message["role"] != "assistant"]
        encoded = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
        results.append(_score_completion(row["id"], completion))

    report = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "split": str(args.split),
        "examples": len(results),
        "metrics": _summarize(results),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


def _score_completion(example_id: str, text: str) -> dict[str, Any]:
    item: dict[str, Any] = {"id": example_id, "json_parse": False, "schema_pass": False}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        item["error"] = "invalid_json"
        return item
    item["json_parse"] = True
    try:
        validate_solution_dict(payload)
        item["schema_pass"] = True
    except (TypeError, ValueError) as exc:
        item["error"] = str(exc)
    output_policy = inspect_output(text)
    sensitive_policy = inspect_sensitive_data(text)
    item["policy_pass"] = not output_policy.blocked and not sensitive_policy.blocked
    item["evidence_count"] = len(payload.get("evidence", [])) if isinstance(payload, dict) else 0
    item["conservative_no_evidence"] = (
        not item["evidence_count"]
        and isinstance(payload, dict)
        and "资料不足" in payload.get("executive_summary", "")
    )
    return item


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)

    def count(field: str) -> int:
        return sum(bool(item.get(field)) for item in results)

    return {
        "total": total,
        "json_parse": count("json_parse"),
        "json_parse_rate": round(count("json_parse") / total, 4) if total else 0.0,
        "schema_pass": count("schema_pass"),
        "schema_pass_rate": round(count("schema_pass") / total, 4) if total else 0.0,
        "policy_pass": count("policy_pass"),
        "policy_pass_rate": round(count("policy_pass") / total, 4) if total else 0.0,
        "conservative_no_evidence": count("conservative_no_evidence"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
