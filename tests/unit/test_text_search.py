from bragi.text_search import cjk_lexical_anchors


def test_cjk_lexical_anchors_split_script_runs_and_preserve_entities() -> None:
    anchors = set(cjk_lexical_anchors("月石の羅針盤はどこ 李"))

    assert {"月石", "羅針盤", "羅針", "針盤", "李"} <= anchors
    assert "の" not in anchors
    assert "は" not in anchors
