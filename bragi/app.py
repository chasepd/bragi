"""Compatibility entrypoint for the Bragi web app."""

from __future__ import annotations

from dataclasses import dataclass

from bragi import __version__
from bragi_web.main import main as web_main


@dataclass(frozen=True)
class ApplicationMetadata:
    """Static metadata for launchers and tests."""

    application_id: str
    display_name: str
    version: str


def get_application_metadata() -> ApplicationMetadata:
    return ApplicationMetadata(
        application_id="dev.bragi.Bragi",
        display_name="Bragi",
        version=__version__,
    )


def cli_main(argv: list[str] | None = None) -> int:
    return web_main(argv, prog="bragi", version=__version__)


def main(argv: list[str] | None = None) -> int:
    return web_main(argv, prog="bragi", version=__version__)
