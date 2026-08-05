"""CLI behavior that the SDK tests cannot cover: exit codes and error text.

The commands themselves are thin wrappers whose logic is tested against the
library. What is only observable here is what a user sees when they get it
wrong: one line, a nonzero exit, and nothing written to their working tree.
"""

from __future__ import annotations

import pytest

from density.cli import main


def _run(monkeypatch, argv: list[str]) -> int:
    """Invoke the console-script entry point, returning its exit code."""
    monkeypatch.setattr("sys.argv", ["density", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    return int(exc.value.code or 0)


def test_missing_store_prints_one_line_and_exits_2(tmp_path, monkeypatch, capsys) -> None:
    """A mistyped store path must not leave a real store behind.

    Before Store.open grew create=False, this created ./typo.densty with a
    manifest and then failed with a traceback about a store the user never
    made.
    """
    monkeypatch.chdir(tmp_path)
    code = _run(monkeypatch, ["search", "./typo.densty", "refunds", "-k", "3"])
    err = capsys.readouterr().err
    assert code == 2
    assert "no store at" in err
    assert "Traceback" not in err
    assert not (tmp_path / "typo.densty").exists()
    assert list(tmp_path.iterdir()) == []


def test_missing_corpus_prints_one_line_and_exits_2(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    code = _run(monkeypatch, ["audit", "./nope"])
    err = capsys.readouterr().err
    assert code == 2
    assert "does not exist" in err
    assert "Traceback" not in err


def test_replay_on_a_missing_store_exits_2(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    code = _run(monkeypatch, ["replay", "./typo.densty", "abc"])
    assert code == 2
    assert "no store at" in capsys.readouterr().err


def test_help_still_exits_zero(monkeypatch, capsys) -> None:
    """The DensityError handler must not swallow typer's own control flow."""
    code = _run(monkeypatch, ["--help"])
    assert code == 0
    assert "synth" in capsys.readouterr().out
