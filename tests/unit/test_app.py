from __future__ import annotations

from pytest import CaptureFixture, MonkeyPatch, raises

from bragi import __version__, app


def test_application_metadata_is_stable() -> None:
    metadata = app.get_application_metadata()

    assert metadata.application_id == "dev.bragi.Bragi"
    assert metadata.display_name == "Bragi"
    assert metadata.version == __version__


def test_bragi_cli_delegates_to_web_main_with_primary_program_name(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[tuple[list[str] | None, str]] = []

    def fake_web_main(
        argv: list[str] | None,
        *,
        prog: str = "bragi-web",
        version: str | None = None,
    ) -> int:
        calls.append((argv, f"{prog}:{version}"))
        return 12

    monkeypatch.setattr(app, "web_main", fake_web_main)

    assert app.cli_main(["status"]) == 12
    assert calls == [(["status"], f"bragi:{__version__}")]


def test_app_main_uses_primary_program_name(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[list[str] | None, str]] = []

    def fake_web_main(
        argv: list[str] | None,
        *,
        prog: str = "bragi-web",
        version: str | None = None,
    ) -> int:
        calls.append((argv, f"{prog}:{version}"))
        return 0

    monkeypatch.setattr(app, "web_main", fake_web_main)

    assert app.main(["--help"]) == 0
    assert calls == [(["--help"], f"bragi:{__version__}")]


def test_bragi_cli_preserves_version_flag(
    capsys: CaptureFixture[str],
) -> None:
    with raises(SystemExit) as excinfo:
        app.cli_main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out == f"Bragi {__version__}\n"
