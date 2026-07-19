"""Secret storage abstraction for provider API keys."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from bragi.persistence.paths import resolve_storage_paths
from bragi.private_files import ensure_private_dir


class SecretStore(Protocol):
    def set_api_key(self, provider: str, api_key: str) -> None:
        ...

    def delete_api_key(self, provider: str) -> None:
        ...

    def has_api_key(self, provider: str) -> bool:
        ...

    def get_api_key(self, provider: str) -> str | None:
        ...


class InMemorySecretStore:
    def __init__(self) -> None:
        self._api_keys: dict[str, str] = {}

    def set_api_key(self, provider: str, api_key: str) -> None:
        self._api_keys[provider] = api_key

    def delete_api_key(self, provider: str) -> None:
        self._api_keys.pop(provider, None)

    def has_api_key(self, provider: str) -> bool:
        return self.get_api_key(provider) is not None

    def get_api_key(self, provider: str) -> str | None:
        value = self._api_keys.get(provider)
        return value if value else None


class SecretStorageError(RuntimeError):
    """Raised when configured secure secret storage cannot read or persist a key."""


class SystemSecretStore:
    """Persist provider keys using the system keyring with file fallback."""

    def __init__(
        self,
        *,
        service_name: str = "dev.bragi.Bragi",
        fallback_path: Path | None = None,
        env: Mapping[str, str] | None = None,
        platform_name: str | None = None,
    ) -> None:
        self._service_name = service_name
        self._fallback_path = fallback_path or self._default_fallback_path(
            env=env,
            platform_name=platform_name,
        )
        self._keyring: Any | None = None
        self._use_keyring = False
        try:
            import keyring as keyring_module
        except ImportError:
            self._keyring = None
        else:
            self._keyring = keyring_module
            self._use_keyring = True

    def set_api_key(self, provider: str, api_key: str) -> None:
        if self._use_keyring and self._keyring is not None:
            try:
                self._keyring.set_password(self._service_name, provider, api_key)
                return
            except Exception:
                raise SecretStorageError(
                    "System keyring write failed; API key was not saved"
                ) from None

        self._set_fallback_api_key(provider=provider, api_key=api_key)

    def delete_api_key(self, provider: str) -> None:
        if self._use_keyring and self._keyring is not None:
            try:
                value = self._keyring.get_password(self._service_name, provider)
                if not value:
                    return
                self._keyring.delete_password(self._service_name, provider)
                return
            except Exception as exc:
                raise SecretStorageError(
                    "System keyring write failed; API key was not removed"
                ) from exc

        self._delete_fallback_api_key(provider=provider)

    def has_api_key(self, provider: str) -> bool:
        return self.get_api_key(provider) is not None

    def get_api_key(self, provider: str) -> str | None:
        if self._use_keyring and self._keyring is not None:
            try:
                value = self._keyring.get_password(self._service_name, provider)
            except Exception as exc:
                raise SecretStorageError(
                    "System keyring read failed; "
                    f"API key could not be read: {exc}"
                ) from exc
            else:
                value = cast(str | None, value)
                return value if value else None

        return self._get_fallback_api_key(provider)

    @property
    def uses_fallback_storage(self) -> bool:
        return not self._use_keyring

    @property
    def fallback_path(self) -> Path:
        return self._fallback_path

    def _default_fallback_path(
        self,
        *,
        env: Mapping[str, str] | None,
        platform_name: str | None,
    ) -> Path:
        return (
            resolve_storage_paths(env=env, platform_name=platform_name).state_dir
            / "api_keys.json"
        )

    def _get_fallback_api_key(self, provider: str) -> str | None:
        values = self._read_fallback_store()
        value = values.get(provider)
        return value if value else None

    def _set_fallback_api_key(self, *, provider: str, api_key: str) -> None:
        values = self._read_fallback_store()
        values[provider] = api_key
        self._write_fallback_store(values)

    def _delete_fallback_api_key(self, *, provider: str) -> None:
        values = self._read_fallback_store()
        if provider not in values:
            return
        del values[provider]
        if values:
            self._write_fallback_store(values)
            return
        try:
            self._fallback_path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise SecretStorageError(
                "Fallback API key store could not be updated; API key was not removed"
            ) from exc

    def _read_fallback_store(self) -> dict[str, str]:
        if not self._fallback_path.exists():
            return {}
        try:
            raw = self._fallback_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except OSError as exc:
            raise SecretStorageError(
                "Fallback API key store could not be read; API key was not saved"
            ) from exc
        except json.JSONDecodeError as exc:
            raise SecretStorageError(
                "Fallback API key store is malformed; API key was not saved"
            ) from exc

        if not isinstance(payload, dict):
            raise SecretStorageError(
                "Fallback API key store is malformed; API key was not saved"
            )

        values: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SecretStorageError(
                    "Fallback API key store is malformed; API key was not saved"
                )
            values[key] = value
        return values

    def _write_fallback_store(self, values: dict[str, str]) -> None:
        ensure_private_dir(self._fallback_path.parent)
        payload = json.dumps(
            values,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        temp_path = self._fallback_path.with_name(
            f".{self._fallback_path.name}.tmp"
        )
        fd = os.open(temp_path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        temp_path.chmod(0o600)
        os.replace(temp_path, self._fallback_path)
        self._fallback_path.chmod(0o600)


LinuxSecretStore = SystemSecretStore
