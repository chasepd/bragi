from __future__ import annotations

from bragi.providers.errors import (
    ProviderError,
    ProviderErrorCategory,
    provider_error_is_model_not_found,
)


def test_provider_error_keyword_construction_has_safe_non_empty_string() -> None:
    exc = ProviderError(
        category=ProviderErrorCategory.RATE_LIMITED,
        message="provider asked Bragi to slow down",
    )

    rendered = str(exc)

    assert rendered
    assert ProviderErrorCategory.RATE_LIMITED.value in rendered
    assert "provider asked Bragi to slow down" in rendered


def test_provider_error_is_model_not_found_matches_only_model_not_found() -> None:
    assert provider_error_is_model_not_found(
        ProviderError(
            category=ProviderErrorCategory.MODEL_NOT_FOUND,
            message="model not found",
            status_code=404,
        )
    )
    assert not provider_error_is_model_not_found(
        ProviderError(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="rate limited",
            status_code=429,
        )
    )
    assert not provider_error_is_model_not_found(None)
    assert not provider_error_is_model_not_found(TimeoutError("timed out"))
