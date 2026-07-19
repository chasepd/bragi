from __future__ import annotations

from bragi.providers.errors import ProviderError, ProviderErrorCategory


def test_provider_error_keyword_construction_has_safe_non_empty_string() -> None:
    exc = ProviderError(
        category=ProviderErrorCategory.RATE_LIMITED,
        message="provider asked Bragi to slow down",
    )

    rendered = str(exc)

    assert rendered
    assert ProviderErrorCategory.RATE_LIMITED.value in rendered
    assert "provider asked Bragi to slow down" in rendered
