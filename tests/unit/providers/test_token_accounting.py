from bragi.providers.token_accounting import estimate_text_tokens


def test_token_estimator_counts_many_short_ascii_tokens() -> None:
    text = " ".join("a" for _ in range(100))

    assert estimate_text_tokens(text) >= 100


def test_token_estimator_counts_punctuation_and_multibyte_text() -> None:
    assert estimate_text_tokens("! " * 80) >= 80
    assert estimate_text_tokens("界" * 80) >= 120


def test_token_estimator_uses_worst_case_for_entropy_and_whitespace() -> None:
    assert estimate_text_tokens("a9Z" * 100) >= 300
    assert estimate_text_tokens(" \n\t" * 100) >= 300
