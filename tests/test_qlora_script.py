from pathlib import Path

from scripts.train_qlora import _build_test_evaluation_kwargs


def test_held_out_evaluation_disables_training_and_does_not_inherit_smoke_steps() -> None:
    kwargs = _build_test_evaluation_kwargs(
        {
            "output_dir": "ignored",
            "max_steps": 1,
            "eval_strategy": "epoch",
            "gradient_checkpointing": True,
        },
        Path("/tmp/adapter"),
    )

    assert kwargs["output_dir"] == "/tmp/adapter/test-eval"
    assert kwargs["do_train"] is False
    assert kwargs["do_eval"] is True
    assert kwargs["eval_strategy"] == "no"
    assert kwargs["save_strategy"] == "no"
    assert kwargs["gradient_checkpointing"] is False
    assert "max_steps" not in kwargs
