from __future__ import annotations

from bragi.redaction import redact_log_value, redact_text
from bragi.services import redaction as service_redaction


def test_service_redaction_reexports_shared_helpers() -> None:
    assert service_redaction.redact_log_value is redact_log_value
    assert service_redaction.redact_text is redact_text
