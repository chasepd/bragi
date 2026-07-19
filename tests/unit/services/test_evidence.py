from bragi.services.evidence import quote_matches_source


def test_quote_matches_source_accepts_format_normalized_quotes() -> None:
    source = 'Captain Ilyra says "ember dawn" beside the beacon lens.'

    assert quote_matches_source("Captain\u00a0Ilyra says", source)
    assert quote_matches_source("Captain   Ilyra", source)
    assert quote_matches_source("`ember dawn`", source)
    assert quote_matches_source("\u201cember dawn\u201d", source)


def test_quote_matches_source_rejects_missing_text() -> None:
    assert not quote_matches_source("ruby library", "Mara steadies the beacon.")
    assert not quote_matches_source("   ", "Mara steadies the beacon.")
