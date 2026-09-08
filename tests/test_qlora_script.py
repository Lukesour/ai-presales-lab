from pathlib import Path

from scripts.train_qlora import _audit_token_lengths, _build_test_evaluation_kwargs


def test_held_out_evaluation_disables_training_and_does_not_inherit_smoke_steps() -> None:
    kwargs = _build_test_evaluation_kwargs(
        {
            "output_dir": "ignored",
            "max_steps": 1,
            "eval_strategy": "epoch",
            "gradient_checkpointing": True,
            "load_best_model_at_end": True,
        },
        Path("/tmp/adapter"),
    )

    assert kwargs["output_dir"] == "/tmp/adapter/test-eval"
    assert kwargs["do_train"] is False
    assert kwargs["do_eval"] is True
    assert kwargs["eval_strategy"] == "no"
    assert kwargs["save_strategy"] == "no"
    assert kwargs["gradient_checkpointing"] is False
    assert kwargs["load_best_model_at_end"] is False
    assert "max_steps" not in kwargs


class _FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return list(range(sum(len(message["content"]) for message in messages)))

    def __call__(self, content, **_kwargs):
        return {"input_ids": list(range(len(content)))}


def test_token_audit_reports_full_sequence_overflow() -> None:
    rows = {
        "train": [
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
    }

    report = _audit_token_lengths(rows, _FakeTokenizer(), max_length=5)

    assert report["train"]["full_sequence"]["max"] == 7
    assert report["train"]["full_sequence"]["over_max_length"] == 1
    assert report["train"]["completion"]["max"] == 6
