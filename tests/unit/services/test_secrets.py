from __future__ import annotations

import builtins
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from bragi.services.secrets import (
    LinuxSecretStore,
    SecretStorageError,
    SystemSecretStore,
)


def test_linux_secret_store_raises_when_keyring_write_fails_without_plaintext_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, str]] = []
    keyring = ModuleType("keyring")

    def set_password(service_name: str, provider: str, api_key: str) -> None:
        calls.append(("set", service_name, provider, api_key))
        raise RuntimeError("secret service locked")

    keyring.set_password = set_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    store = LinuxSecretStore(fallback_path=fallback_path)

    with pytest.raises(RuntimeError, match="API key was not saved"):
        store.set_api_key("openrouter", "sk-openrouter-fallback")

    assert calls == [
        (
            "set",
            "dev.bragi.Bragi",
            "openrouter",
            "sk-openrouter-fallback",
        )
    ]
    assert not fallback_path.exists()


def test_linux_secret_store_get_failure_keeps_later_writes_on_keyring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str] | tuple[str, str, str, str]] = []
    keyring = ModuleType("keyring")

    def get_password(service_name: str, provider: str) -> str | None:
        calls.append(("get", service_name, provider))
        raise RuntimeError("secret service read failed")

    def set_password(service_name: str, provider: str, api_key: str) -> None:
        calls.append(("set", service_name, provider, api_key))
        raise RuntimeError("secret service write failed")

    keyring.get_password = get_password  # type: ignore[attr-defined]
    keyring.set_password = set_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    store = LinuxSecretStore(fallback_path=fallback_path)

    with pytest.raises(SecretStorageError, match="read failed|keyring"):
        store.get_api_key("venice")
    assert store.uses_fallback_storage is False

    with pytest.raises(SecretStorageError, match="API key was not saved"):
        store.set_api_key("venice", "sk-venice-keyring")

    assert calls == [
        ("get", "dev.bragi.Bragi", "venice"),
        ("set", "dev.bragi.Bragi", "venice", "sk-venice-keyring"),
    ]
    assert not fallback_path.exists()


def test_linux_secret_store_write_failure_redacts_submitted_key_and_suppresses_cause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted_key = "sk-openrouter-leaked-by-keyring"
    keyring_failure = RuntimeError(f"backend echoed submitted key {submitted_key}")
    keyring = ModuleType("keyring")

    def set_password(service_name: str, provider: str, api_key: str) -> None:
        assert service_name == "dev.bragi.Bragi"
        assert provider == "openrouter"
        assert api_key == submitted_key
        raise keyring_failure

    keyring.set_password = set_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    store = LinuxSecretStore(fallback_path=fallback_path)

    with pytest.raises(SecretStorageError) as excinfo:
        store.set_api_key("openrouter", submitted_key)

    assert submitted_key not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    assert not fallback_path.exists()


def test_linux_secret_store_get_failure_does_not_read_existing_plaintext_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str]] = []
    keyring = ModuleType("keyring")

    def get_password(service_name: str, provider: str) -> str | None:
        calls.append(("get", service_name, provider))
        raise RuntimeError("secret service read failed")

    keyring.get_password = get_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    fallback_path.parent.mkdir(parents=True)
    fallback_path.write_text(
        json.dumps({"openrouter": "sk-openrouter-plaintext"}),
        encoding="utf-8",
    )
    store = LinuxSecretStore(fallback_path=fallback_path)

    def fail_if_fallback_is_read() -> dict[str, str]:
        raise AssertionError(
            "fallback storage must not be read after keyring failure"
        )

    monkeypatch.setattr(store, "_read_fallback_store", fail_if_fallback_is_read)

    with pytest.raises(SecretStorageError, match="read failed|keyring"):
        store.get_api_key("openrouter")
    with pytest.raises(SecretStorageError, match="read failed|keyring"):
        store.has_api_key("openrouter")
    assert store.uses_fallback_storage is False
    assert calls == [
        ("get", "dev.bragi.Bragi", "openrouter"),
        ("get", "dev.bragi.Bragi", "openrouter"),
    ]


def test_linux_secret_store_persists_fallback_state_across_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"

    LinuxSecretStore(fallback_path=fallback_path).set_api_key(
        "venice",
        "sk-venice-persisted",
    )
    restarted_store = LinuxSecretStore(fallback_path=fallback_path)

    assert restarted_store.has_api_key("venice") is True
    assert restarted_store.get_api_key("venice") == "sk-venice-persisted"


def test_linux_secret_store_deletes_fallback_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    store = LinuxSecretStore(fallback_path=fallback_path)
    store.set_api_key("openrouter", "sk-openrouter")
    store.set_api_key("venice", "sk-venice")

    store.delete_api_key("openrouter")

    assert store.has_api_key("openrouter") is False
    assert store.get_api_key("openrouter") is None
    assert json.loads(fallback_path.read_text(encoding="utf-8")) == {
        "venice": "sk-venice"
    }


def test_linux_secret_store_fallback_delete_missing_key_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    store = LinuxSecretStore(fallback_path=fallback_path)

    store.delete_api_key("openrouter")

    assert store.has_api_key("openrouter") is False
    assert not fallback_path.exists()


def test_linux_secret_store_treats_empty_key_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    store = LinuxSecretStore(fallback_path=fallback_path)

    store.set_api_key("openrouter", "")

    assert store.has_api_key("openrouter") is False
    assert store.get_api_key("openrouter") is None


def test_linux_secret_store_deletes_keyring_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str]] = []
    keyring = ModuleType("keyring")

    def get_password(service_name: str, provider: str) -> str | None:
        calls.append(("get", service_name, provider))
        return "sk-openrouter"

    def delete_password(service_name: str, provider: str) -> None:
        calls.append(("delete", service_name, provider))

    keyring.get_password = get_password  # type: ignore[attr-defined]
    keyring.delete_password = delete_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    store = LinuxSecretStore(fallback_path=tmp_path / "api_keys.json")

    store.delete_api_key("openrouter")

    assert calls == [
        ("get", "dev.bragi.Bragi", "openrouter"),
        ("delete", "dev.bragi.Bragi", "openrouter"),
    ]


@pytest.mark.parametrize(
    "raw_payload",
    [
        "{not valid json",
        '["not", "a", "dict"]',
    ],
)
def test_linux_secret_store_fallback_write_fails_closed_when_existing_file_is_bad(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_payload: str,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    fallback_path.parent.mkdir(parents=True)
    fallback_path.write_text(raw_payload, encoding="utf-8")

    store = LinuxSecretStore(fallback_path=fallback_path)

    with pytest.raises(SecretStorageError):
        store.set_api_key("openrouter", "sk-new-key")

    assert fallback_path.read_text(encoding="utf-8") == raw_payload
    assert not fallback_path.with_name(f".{fallback_path.name}.tmp").exists()


def test_linux_secret_store_fallback_rejects_non_string_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    fallback_path.parent.mkdir(parents=True)
    raw_payload = '{"venice": 123}'
    fallback_path.write_text(raw_payload, encoding="utf-8")

    store = LinuxSecretStore(fallback_path=fallback_path)

    with pytest.raises(SecretStorageError):
        store.set_api_key("openrouter", "sk-new-key")

    assert fallback_path.read_text(encoding="utf-8") == raw_payload
    assert not fallback_path.with_name(f".{fallback_path.name}.tmp").exists()


def test_linux_secret_store_fallback_write_fails_closed_when_file_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"
    fallback_path.parent.mkdir(parents=True)
    fallback_path.write_text('{"openrouter": "sk-old-key"}', encoding="utf-8")
    original_payload = fallback_path.read_text(encoding="utf-8")
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == fallback_path:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    store = LinuxSecretStore(fallback_path=fallback_path)

    with pytest.raises(SecretStorageError):
        store.set_api_key("openrouter", "sk-new-key")

    monkeypatch.setattr(Path, "read_text", original_read_text)
    assert fallback_path.read_text(encoding="utf-8") == original_payload
    assert not fallback_path.with_name(f".{fallback_path.name}.tmp").exists()


def test_linux_secret_store_writes_fallback_file_with_owner_only_permissions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    fallback_path = tmp_path / "state" / "bragi" / "api_keys.json"

    LinuxSecretStore(fallback_path=fallback_path).set_api_key(
        "openrouter",
        "sk-owner-only",
    )

    if os.name != "nt":
        assert stat.S_IMODE(fallback_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(fallback_path.stat().st_mode) == 0o600


def test_linux_secret_store_default_fallback_path_uses_xdg_state_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    state_home = tmp_path / "xdg-state"
    home = tmp_path / "home"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv("HOME", str(home))

    LinuxSecretStore(platform_name="posix").set_api_key(
        "openrouter",
        "sk-xdg-state",
    )

    expected_path = state_home / "bragi" / "api_keys.json"
    assert json.loads(expected_path.read_text(encoding="utf-8")) == {
        "openrouter": "sk-xdg-state"
    }
    assert not (home / ".local" / "state" / "bragi" / "api_keys.json").exists()


def test_linux_secret_store_default_fallback_path_ignores_relative_xdg_state_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _disable_keyring_import(monkeypatch)
    home = tmp_path / "home"
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")

    monkeypatch.chdir(first_cwd)
    first_store = LinuxSecretStore(platform_name="posix")
    monkeypatch.chdir(second_cwd)
    second_store = LinuxSecretStore(platform_name="posix")

    expected_path = home / ".local" / "state" / "bragi" / "api_keys.json"
    assert first_store.fallback_path == expected_path
    assert second_store.fallback_path == expected_path
    assert first_store.fallback_path.is_absolute()

    first_store.set_api_key("openrouter", "sk-home-state")

    assert json.loads(expected_path.read_text(encoding="utf-8")) == {
        "openrouter": "sk-home-state"
    }
    assert not (first_cwd / "relative-state" / "bragi" / "api_keys.json").exists()
    assert not (second_cwd / "relative-state" / "bragi" / "api_keys.json").exists()


def test_system_secret_store_is_exported_as_linux_compatibility_alias() -> None:
    assert LinuxSecretStore is SystemSecretStore


def test_system_secret_store_default_fallback_path_uses_windows_state_dir(
    tmp_path: Path,
) -> None:
    local_app_data = tmp_path / "LocalAppData"

    store = SystemSecretStore(
        env={"LOCALAPPDATA": str(local_app_data)},
        platform_name="nt",
    )

    assert store.fallback_path == (
        local_app_data / "Bragi" / "state" / "api_keys.json"
    )


def _disable_keyring_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_keyring(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "keyring":
            raise ImportError("keyring unavailable in test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_keyring)
