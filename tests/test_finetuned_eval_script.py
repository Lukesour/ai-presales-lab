import json

import pytest

from scripts.evaluate_finetuned_model import _validate_adapter_artifact


def test_validate_adapter_artifact_requires_config_and_weights(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="adapter weights"):
        _validate_adapter_artifact(adapter_dir)

    (adapter_dir / "adapter_model.safetensors").write_bytes(b"placeholder")
    config = _validate_adapter_artifact(adapter_dir)
    assert config["base_model_name_or_path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["_weight_files"] == ["adapter_model.safetensors"]


def test_validate_adapter_artifact_rejects_missing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _validate_adapter_artifact(tmp_path / "missing")
