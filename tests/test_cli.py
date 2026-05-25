import pytest

from mlx_speechd.cli import build_parser, resolve_text


def test_help_contains_required_commands() -> None:
    help_text = build_parser().format_help()

    for command in ("say", "render", "hermes", "up", "down", "stop", "status", "serve"):
        assert command in help_text


def test_say_shortcut_text() -> None:
    args = build_parser().parse_args(["say", "hello"])

    assert resolve_text(args) == "hello"


def test_say_text_flag_wins() -> None:
    args = build_parser().parse_args(["say", "positional", "--text", "flag"])

    assert resolve_text(args) == "flag"


def test_text_is_required() -> None:
    args = build_parser().parse_args(["say"])

    with pytest.raises(ValueError):
        resolve_text(args)
