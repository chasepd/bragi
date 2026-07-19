_STYLES_CSS = "frontend/src/styles.css"


def test_touch_action_labels_are_hidden_until_mobile_touch_layouts() -> None:
    assert_css_rule_contains(
        ".touch-action-label",
        "display: none;",
    )
    stylesheet = _stylesheet()
    assert "@media (max-width: 760px), (pointer: coarse) {" in stylesheet
    mobile_rule_start = stylesheet.index(".touch-labeled-action .touch-action-label")
    mobile_rule_end = stylesheet.index("}", mobile_rule_start)
    mobile_rule = stylesheet[mobile_rule_start:mobile_rule_end]
    assert "display: inline;" in mobile_rule
    assert "overflow-wrap: anywhere;" in mobile_rule
    action_rule_start = stylesheet.index(".touch-labeled-action")
    action_rule_end = stylesheet.index("}", action_rule_start)
    action_rule = stylesheet[action_rule_start:action_rule_end]
    assert "min-width: 0;" in action_rule
    assert "gap: 6px;" in action_rule
    assert ".minihead-actions .touch-labeled-action" in stylesheet


def test_auth_shell_scrolls_independently_on_short_viewports() -> None:
    assert_css_rule_contains(
        ".auth-shell",
        "height: 100vh;",
        "height: 100dvh;",
        "min-height: 0;",
        "overflow-y: auto;",
        "align-items: start;",
        "justify-items: center;",
    )
    assert_css_rule_contains(
        ".auth-panel",
        "margin-block: auto;",
        "max-height: calc(100dvh - 48px);",
        "overflow: auto;",
    )


def test_character_text_phone_messages_wrap_long_viewport_breaking_content() -> None:
    assert_css_rule_contains(
        ".character-text-messages",
        "min-width: 0;",
        "min-height: 0;",
        "overflow-x: hidden;",
        "overflow-y: auto;",
    )
    assert_css_rule_contains(
        ".character-text-bubble",
        "width: fit-content;",
        "min-width: 0;",
        "max-width: min(76%, 520px);",
    )
    assert_css_rule_contains(
        ".character-text-bubble-body",
        "min-width: 0;",
        "max-width: 100%;",
        "overflow-wrap: anywhere;",
        "word-break: break-word;",
        "white-space: pre-wrap;",
    )
    assert_css_rule_contains(
        ".character-text-bubble-body p",
        "min-width: 0;",
        "max-width: 100%;",
        "overflow-wrap: anywhere;",
        "word-break: break-word;",
    )
    assert_css_rule_contains(
        ".character-text-bubble-body code",
        "overflow-wrap: anywhere;",
        "word-break: break-word;",
        "white-space: pre-wrap;",
    )
    assert_css_rule_contains(
        ".character-text-bubble-body pre",
        "overflow-wrap: anywhere;",
        "word-break: break-word;",
        "white-space: pre-wrap;",
    )
    assert_css_rule_contains(
        ".character-text-bubble-body a",
        "overflow-wrap: anywhere;",
        "word-break: break-word;",
    )


def assert_css_rule_contains(selector: str, *declarations: str) -> None:
    rules = _css_rules(selector)
    missing = [
        declaration
        for declaration in declarations
        if not any(declaration in rule for rule in rules)
    ]
    assert not missing, f"{selector} missing declarations: {missing}"


def _css_rules(selector: str) -> tuple[str, ...]:
    rules = []
    stylesheet = _stylesheet()
    for block_text in stylesheet.split("}"):
        if "{" not in block_text:
            continue
        selector_text, declaration_text = block_text.rsplit("{", 1)
        selectors = [candidate.strip() for candidate in selector_text.split(",")]
        if selector in selectors:
            rules.append(" ".join(declaration_text.split()))
    assert rules, f"Missing CSS rule for {selector}"
    return tuple(rules)


def _stylesheet() -> str:
    with open(_STYLES_CSS, encoding="utf-8") as styles:
        return styles.read()
