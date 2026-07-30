"""Tests for the logging infrastructure everything else in the pipeline builds on."""

import logging

import pytest

from src.utils.logging import (
    ROOT_LOGGER_NAME,
    get_logger,
    is_tty,
    progress,
    resolve_level,
    set_log_level,
    setup_logging,
)


# Module state (`_configured`, the handler) is global and leaks between tests.
pytestmark = pytest.mark.usefixtures("reset_src_logging")


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("debug", logging.DEBUG),
        ("DEBUG", logging.DEBUG),
        ("Warning", logging.WARNING),
        ("10", logging.DEBUG),
        ("", logging.INFO),
        ("   ", logging.INFO),
        ("nonsense", logging.INFO),
    ],
)
def test_resolve_level_from_env(
    monkeypatch: pytest.MonkeyPatch, env_value: str, expected: int
) -> None:
    # pytest.ini's `env =` block is inert (pytest-env is not installed), so the
    # env var has to be set here.
    monkeypatch.setenv("LOG_LEVEL", env_value)
    assert resolve_level() == expected


def test_resolve_level_defaults_to_info_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert resolve_level() == logging.INFO


def test_explicit_level_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert resolve_level("DEBUG") == logging.DEBUG


def test_blank_explicit_falls_through_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert resolve_level("") == logging.ERROR


def test_unreadable_level_warns_rather_than_failing_silently(
    monkeypatch: pytest.MonkeyPatch, capture_src_logs
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "louder-please")
    with capture_src_logs(logging.WARNING) as records:
        level = resolve_level()

    assert level == logging.INFO
    assert len(records) == 1
    assert "louder-please" in records[0].getMessage()


def test_unreadable_level_warns_once_not_once_per_module(
    monkeypatch: pytest.MonkeyPatch, capture_src_logs
) -> None:
    """get_logger() calls setup_logging(), and every module calls get_logger()."""
    monkeypatch.setenv("LOG_LEVEL", "louder-please")
    with capture_src_logs(logging.WARNING) as records:
        for name in ("src.a", "src.b", "src.c", "src.d"):
            get_logger(name)

    assert len(records) == 1


def test_setup_does_not_undo_a_later_set_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    set_log_level(logging.DEBUG)
    # A module imported after the level was raised must not reset it from the env.
    assert get_logger("src.late").isEnabledFor(logging.DEBUG)


def test_setup_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    setup_logging()
    setup_logging()
    setup_logging()
    assert len(logging.getLogger(ROOT_LOGGER_NAME).handlers) == 1


def test_no_double_print_when_root_logger_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: ClearML calls logging.basicConfig() on the root logger."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = get_logger("src.demo")

    root_records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            root_records.append(record)

    src_records: list[logging.LogRecord] = []

    class _SrcCollector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            src_records.append(record)

    root_handler = _Collector()
    src_handler = _SrcCollector()
    logging.getLogger().addHandler(root_handler)
    logging.getLogger(ROOT_LOGGER_NAME).addHandler(src_handler)
    try:
        logger.info("only once")
    finally:
        logging.getLogger().removeHandler(root_handler)
        logging.getLogger(ROOT_LOGGER_NAME).removeHandler(src_handler)

    assert len(src_records) == 1
    assert root_records == []


def test_names_outside_src_tree_are_normalized(
    monkeypatch: pytest.MonkeyPatch, capture_src_logs
) -> None:
    """`make run` runs src/train.py as a script, so its __name__ is __main__."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = get_logger("__main__")

    assert logger.name.startswith(f"{ROOT_LOGGER_NAME}.")
    with capture_src_logs(logging.INFO) as records:
        logger.info("reachable")
    assert len(records) == 1


def test_src_names_are_left_alone() -> None:
    assert get_logger("src.data.setup").name == "src.data.setup"


def test_set_log_level_reconfigures_after_setup(
    monkeypatch: pytest.MonkeyPatch, capture_src_logs
) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = get_logger("src.demo")
    assert not logger.isEnabledFor(logging.DEBUG)

    set_log_level(logging.DEBUG)
    assert logger.isEnabledFor(logging.DEBUG)

    with capture_src_logs(logging.DEBUG) as records:
        logger.debug("now visible")
    assert len(records) == 1


def test_debug_is_invisible_at_default_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = get_logger("src.demo")

    # Asserted on the logger rather than on captured records: capture_src_logs
    # lowers the level to whatever it is asked to capture, which would mask this.
    assert not logger.isEnabledFor(logging.DEBUG)
    assert logger.isEnabledFor(logging.INFO)


def test_is_tty_is_evaluated_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdout", type("_NoTTY", (), {})())
    assert is_tty() is False


def test_progress_emits_exactly_one_summary_line(capture_src_logs) -> None:
    with capture_src_logs(logging.INFO) as records:
        consumed = list(progress(range(500), desc="units"))

    assert len(consumed) == 500
    assert len(records) == 1
    assert "500 items" in records[0].getMessage()
