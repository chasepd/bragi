from __future__ import annotations

from bragi.providers.message_names import provider_message_name


def test_provider_message_name_sanitizes_display_names() -> None:
    assert provider_message_name("  Mara Vale, Signal-Warden!  ") == (
        "Mara_Vale_Signal-Warden"
    )


def test_provider_message_name_rejects_blank_or_symbol_only_names() -> None:
    assert provider_message_name(None) is None
    assert provider_message_name("   ") is None
    assert provider_message_name("!!!") is None


def test_provider_message_name_trims_limit_without_dangling_separators() -> None:
    long_name = f"{'a' * 63}-extra"

    assert provider_message_name(long_name) == "a" * 63
