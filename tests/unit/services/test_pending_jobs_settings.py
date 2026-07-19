from bragi.services.pending_jobs_settings import sanitize_pending_jobs_display_mode


def test_sanitize_pending_jobs_display_mode_accepts_known_modes() -> None:
    assert sanitize_pending_jobs_display_mode("compact") == "compact"
    assert sanitize_pending_jobs_display_mode(" expanded ") == "expanded"
    assert sanitize_pending_jobs_display_mode("expanded-full") == "expanded_full"


def test_sanitize_pending_jobs_display_mode_defaults_unknown_values() -> None:
    assert sanitize_pending_jobs_display_mode("grouped") == "compact"
    assert sanitize_pending_jobs_display_mode(True) == "compact"
