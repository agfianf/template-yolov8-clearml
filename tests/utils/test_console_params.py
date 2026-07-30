"""Guards for the console-verbosity parameters.

`config_clearml()` folds `args_logging` and `args_augment` into `args_train`,
and `args_train` is splatted straight into `model_yolo.train(**args_train)`.
Ultralytics validates the keys and raises on unknown ones -- putting `log_level`
in either of those dicts breaks every training run with
"SyntaxError: 'log_level' is not a valid YOLO argument".
"""

from src.params import args_augment, args_console, args_logging, args_train


def test_console_params_are_not_ultralytics_arguments() -> None:
    for name, other in (("args_train", args_train), ("args_logging", args_logging)):
        overlap = set(args_console) & set(other)
        assert not overlap, f"{overlap} would reach YOLO.train() through {name}"


def test_console_params_stay_out_of_the_dicts_merged_into_train() -> None:
    merged = {**args_logging, **args_augment}
    assert "log_level" not in merged
    assert "progress" not in merged


def test_console_defaults_defer_to_the_environment() -> None:
    # Empty means "no opinion" so $LOG_LEVEL still wins; see resolve_level().
    assert args_console["log_level"] == ""
    assert args_console["progress"] == "auto"
