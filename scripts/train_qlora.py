#!/usr/bin/env python3
"""Run the optional TRL + PEFT QLoRA experiment on a compatible GPU."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from ai_presales_lab.finetuning import dataset_stats, load_conversations

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/finetune/trl_qlora.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from a Trainer checkpoint directory after a Colab runtime restart.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(config, ensure_ascii=False, indent=2))

    data_paths = {
        split: ROOT / config["data"][f"{split}_file"] for split in ("train", "dev", "test")
    }
    data_stats = {}
    for split, path in data_paths.items():
        if not path.exists():
            print(f"Missing {path}; run make build-finetune-dataset first.")
            return 2
        data_stats[split] = dataset_stats(load_conversations(path))
    print(json.dumps({"dataset_stats": data_stats}, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run: dataset and training configuration parsed successfully.")
        return 0

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            "Training dependencies are optional. Install the finetune extra in Colab before "
            f"running this command: {exc}"
        )
        return 3

    if not torch.cuda.is_available():
        print("A CUDA GPU is required for the configured 4-bit QLoRA run.")
        return 4

    dataset = load_dataset(
        "json",
        data_files={split: str(path) for split, path in data_paths.items()},
    )
    compute_dtype = _select_compute_dtype(torch, config.get("compute_dtype", "auto"))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_kwargs = {
        "quantization_config": quantization,
        "device_map": "auto",
        "torch_dtype": compute_dtype,
    }
    if config.get("model_revision"):
        model_kwargs["revision"] = config["model_revision"]
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **model_kwargs)
    model.config.use_cache = False
    tokenizer_kwargs = {}
    if config.get("model_revision"):
        tokenizer_kwargs["revision"] = config["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    training = config["training"]
    if training.get("gradient_checkpointing", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    dataset, loss_kwargs, loss_mode = _prepare_dataset_for_sft(dataset, tokenizer, training)
    print(json.dumps({"loss_mode": loss_mode}, ensure_ascii=False))

    lora = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    eval_strategy = training.get("eval_strategy", "epoch")
    save_strategy = training.get("save_strategy", "epoch")
    training_kwargs = dict(
        output_dir=str(_resolve_output_dir(config["output_dir"])),
        num_train_epochs=training["epochs"],
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training.get("optim", "paged_adamw_8bit"),
        warmup_ratio=training.get("warmup_ratio", 0.0),
        logging_steps=training["logging_steps"],
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        max_length=training["max_length"],
        packing=False,
        report_to=[],
        seed=training["seed"],
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        **loss_kwargs,
    )
    if eval_strategy == "steps":
        training_kwargs["eval_steps"] = training.get("eval_steps", 25)
    if save_strategy == "steps":
        training_kwargs["save_steps"] = training.get("save_steps", 25)
    training_args = SFTConfig(**training_kwargs)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        peft_config=lora,
        args=training_args,
    )
    resume_path = None
    if args.resume_from_checkpoint:
        resume_path = str(args.resume_from_checkpoint.resolve())
        if not args.resume_from_checkpoint.exists():
            print(f"Checkpoint does not exist: {args.resume_from_checkpoint}")
            return 5
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    train_output = trainer.train(resume_from_checkpoint=resume_path)
    train_seconds = round(time.perf_counter() - train_started, 3)
    output_dir = _resolve_output_dir(config["output_dir"])
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name_or_path": config["model_name_or_path"],
                "model_revision": config.get("model_revision"),
                "train_metrics": train_output.metrics,
                "test_metrics": test_metrics,
                "runtime": {
                    **_runtime_metadata(torch),
                    "train_seconds": train_seconds,
                    "peak_gpu_memory_allocated_gb": _peak_gpu_memory(torch, "allocated"),
                    "peak_gpu_memory_reserved_gb": _peak_gpu_memory(torch, "reserved"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Adapter saved to {output_dir}")
    print(json.dumps({"test_metrics": test_metrics}, ensure_ascii=False, indent=2))
    return 0


def _resolve_output_dir(value: str | Path) -> Path:
    output_dir = Path(value)
    return output_dir if output_dir.is_absolute() else ROOT / output_dir


def _runtime_metadata(torch) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "compute_capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(0)
                ),
            }
        )
    return metadata


def _select_compute_dtype(torch, configured: str):
    if configured == "float16":
        return torch.float16
    if configured == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("compute_dtype=bfloat16 was requested but this GPU does not support it")
        return torch.bfloat16
    if configured != "auto":
        raise ValueError("compute_dtype must be one of: auto, float16, bfloat16")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _peak_gpu_memory(torch, kind: str) -> float | None:
    if not torch.cuda.is_available():
        return None
    value = (
        torch.cuda.max_memory_allocated()
        if kind == "allocated"
        else torch.cuda.max_memory_reserved()
    )
    return round(value / 1024**3, 3)


def _prepare_dataset_for_sft(dataset, tokenizer, training: dict[str, object]):
    """Use assistant-only masks when the chat template supports them.

    Older Qwen2 chat templates may not expose TRL's ``{% generation %}``
    markers.  In that case, convert the same conversations to conversational
    prompt/completion records so ``completion_only_loss`` still excludes the
    user and system messages.
    """

    wants_assistant_only = bool(training.get("assistant_only_loss", True))
    template = tokenizer.chat_template or ""
    if wants_assistant_only and "{% generation %}" in template:
        return dataset, {"assistant_only_loss": True}, "assistant_only"

    def to_prompt_completion(row: dict[str, object]) -> dict[str, object]:
        messages = row["messages"]
        return {"prompt": messages[:-1], "completion": messages[-1:]}

    columns = dataset["train"].column_names
    converted = dataset.map(to_prompt_completion, remove_columns=columns)
    return converted, {"completion_only_loss": True}, "prompt_completion"


if __name__ == "__main__":
    raise SystemExit(main())
