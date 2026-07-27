from bragi.text_search import cjk_lexical_anchors


def test_cjk_lexical_anchors_split_script_runs_and_preserve_entities() -> None:
    anchors = set(cjk_lexical_anchors("月石の羅針盤はどこ 李"))

    assert {"月石", "羅針盤", "羅針", "針盤", "李"} <= anchors
    assert "の" not in anchors
    assert "は" not in anchors


def test_cjk_lexical_anchors_bound_oversized_runs() -> None:
    anchors = cjk_lexical_anchors("秘密" * 100_000)

    assert anchors
    assert max(map(len, anchors)) <= 64
