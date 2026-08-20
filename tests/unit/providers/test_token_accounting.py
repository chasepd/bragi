from bragi.providers.token_accounting import estimate_text_tokens


def test_token_estimator_counts_ascii_text_at_roughly_one_token_per_byte() -> None:
    text = "a" * 400

    assert estimate_text_tokens(text) == 100


def test_token_estimator_rounds_up_for_partial_bytes() -> None:
    assert estimate_text_tokens("a") == 1
    assert estimate_text_tokens("ab") == 1
    assert estimate_text_tokens("abc") == 1
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_token_estimator_counts_punctuation_and_multibyte_text_conservatively() -> None:
    assert estimate_text_tokens("! " * 80) == 40
    assert estimate_text_tokens("界" * 80) >= 60


def test_token_estimator_is_a_conservative_upper_bound_for_arbitrary_input() -> None:
    assert estimate_text_tokens("a9Z" * 100) >= 75
    assert estimate_text_tokens(" \n\t" * 100) >= 75


def test_token_estimator_returns_zero_for_empty_text() -> None:
    assert estimate_text_tokens("") == 0
