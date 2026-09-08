import json

import pytest

from scripts.evaluate_finetuned_model import (
    _score_completion,
    _summarize,
    _validate_adapter_artifact,
)


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


def test_score_completion_can_include_truncated_diagnostic_preview() -> None:
    result = _score_completion(
        "case-001",
        "not-json output",
        include_output_preview=True,
        preview_chars=8,
    )

    assert result["json_parse"] is False
    assert result["error"] == "invalid_json"
    assert result["output_chars"] == len("not-json output")
    assert result["output_preview"] == "not-json"
    assert result["generation_truncated"] is False


def test_score_completion_keeps_preview_disabled_by_default() -> None:
    result = _score_completion("case-001", "not-json output")

    assert "output_preview" not in result
    assert result["output_chars"] == len("not-json output")


def test_summary_surfaces_generation_truncation_and_output_length() -> None:
    summary = _summarize(
        [
            {
                "json_parse": False,
                "schema_pass": False,
                "output_chars": 100,
                "generation_truncated": True,
            },
            {
                "json_parse": True,
                "schema_pass": True,
                "policy_pass": True,
                "output_chars": 200,
                "generation_truncated": False,
            },
        ]
    )

    assert summary["generation_truncated"] == 1
    assert summary["generation_truncated_rate"] == 0.5
    assert summary["output_chars_p50"] == 100
    assert summary["output_chars_max"] == 200
