"""Tests for local Oura OAuth handling."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Self
from urllib.parse import parse_qs
from urllib.request import Request

import pytest

import cycle_forecast.data.oura_auth as oura_auth
from cycle_forecast.data.oura_auth import (
    OuraAuthorizationError,
    OuraScope,
    OuraToken,
    build_authorization_url,
    load_token,
    parse_authorization_redirect,
    save_token,
)


def test_authorization_url_uses_state_and_minimal_daily_scope() -> None:
    """Construct an authorization request without personal-data scopes."""
    url = build_authorization_url(
        client_id="synthetic-client",
        redirect_uri="http://localhost:8765/callback",
        state="synthetic-state",
        scopes=(OuraScope.DAILY,),
    )

    assert "scope=daily" in url
    assert "state=synthetic-state" in url
    assert "personal" not in url


def test_redirect_requires_matching_state() -> None:
    """Reject a redirect that could belong to another browser flow."""
    with pytest.raises(OuraAuthorizationError, match="state"):
        parse_authorization_redirect(
            redirected_url="http://localhost/callback?code=code&state=wrong",
            expected_state="expected",
        )


def test_token_file_round_trip_uses_owner_only_permissions(tmp_path: Path) -> None:
    """Persist OAuth secrets atomically without group or world access."""
    path = tmp_path / "private/token.json"
    token = OuraToken(
        access_token="synthetic-access",
        refresh_token="synthetic-refresh",
        expires_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    save_token(token=token, path=path)

    assert load_token(path=path) == token
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(path.parent).st_mode & 0o777 == 0o700


def test_load_oauth_application_requires_both_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before authorization when application credentials are incomplete."""
    monkeypatch.delenv("OURA_CLIENT_ID", raising=False)
    monkeypatch.delenv("OURA_CLIENT_SECRET", raising=False)

    def missing_password(_: str, __: str) -> None:
        """Simulate an absent Keychain item."""

    monkeypatch.setattr(oura_auth.keyring, "get_password", missing_password)

    with pytest.raises(OuraAuthorizationError, match="OURA_CLIENT_ID"):
        oura_auth.load_oauth_application()


def test_save_and_load_oauth_application_through_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep application credentials out of files and shell history."""
    values: dict[tuple[str, str], str] = {}

    def set_password(service: str, account: str, value: str) -> None:
        """Store an invented Keychain value."""
        values[(service, account)] = value

    def get_password(service: str, account: str) -> str | None:
        """Read an invented Keychain value."""
        return values.get((service, account))

    monkeypatch.delenv("OURA_CLIENT_ID", raising=False)
    monkeypatch.delenv("OURA_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(oura_auth.keyring, "set_password", set_password)
    monkeypatch.setattr(oura_auth.keyring, "get_password", get_password)

    oura_auth.save_oauth_application(
        client_id="synthetic-client", client_secret="synthetic-secret"
    )

    assert oura_auth.load_oauth_application() == (
        "synthetic-client",
        "synthetic-secret",
    )


def test_authorize_interactively_validates_redirect_and_saves_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete the local browser flow without exposing credentials."""
    token = OuraToken(
        access_token="synthetic-access",
        refresh_token="synthetic-refresh",
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    monkeypatch.setenv("OURA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("OURA_CLIENT_SECRET", "synthetic-secret")

    def fixed_state(_: int | None = None) -> str:
        """Return a deterministic OAuth state."""
        return "state"

    def open_browser(_: str) -> bool:
        """Simulate opening the system browser."""
        return True

    def exchange_code(**_: object) -> OuraToken:
        """Return the invented token without a network request."""
        return token

    monkeypatch.setattr(oura_auth.secrets, "token_urlsafe", fixed_state)
    monkeypatch.setattr(oura_auth.webbrowser, "open", open_browser)
    monkeypatch.setattr(
        oura_auth,
        "exchange_authorization_code",
        exchange_code,
    )
    path = tmp_path / "token.json"

    result = oura_auth.authorize_interactively(
        redirect_uri="http://localhost:8765/callback",
        input_fn=lambda _: "http://localhost:8765/callback?code=code&state=state",
        token_path=path,
    )

    assert result == token
    assert load_token(path=path) == token


def test_load_token_rejects_unsupported_schema(tmp_path: Path) -> None:
    """Reject local credential files created under unknown rules."""
    path = tmp_path / "token.json"
    path.write_text(
        '{"schema_version":"future","access_token":"a",'
        '"refresh_token":"r","expires_at":"2030-01-01T00:00:00Z"}',
        encoding="utf-8",
    )

    with pytest.raises(OuraAuthorizationError, match="unsupported"):
        load_token(path=path)


def test_exchange_and_refresh_validate_token_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate both OAuth token grants through the shared HTTP boundary."""

    class Response:
        """Provide a context-managed invented HTTP response."""

        def __enter__(self) -> Self:
            """Enter the response context."""
            return self

        def __exit__(self, *_: object) -> None:
            """Exit the response context."""

        def read(self) -> bytes:
            """Return a documented token payload."""
            return (
                b'{"access_token":"new-access","token_type":"bearer",'
                b'"expires_in":2592000,"refresh_token":"new-refresh",'
                b'"scope":"extapi:daily","id_token":null}'
            )

    requests: list[Request] = []

    def urlopen(request: Request, timeout: int) -> Response:
        """Capture token requests without contacting Oura."""
        assert timeout == 30
        requests.append(request)
        return Response()

    monkeypatch.setattr(oura_auth, "urlopen", urlopen)

    exchanged = oura_auth.exchange_authorization_code(
        code="synthetic-code",
        client_id="synthetic-client",
        client_secret="synthetic-secret",
        redirect_uri="http://localhost/callback",
    )
    refreshed = oura_auth.refresh_access_token(
        refresh_token="old-refresh",
        client_id="synthetic-client",
        client_secret="synthetic-secret",
    )

    assert exchanged.access_token == "new-access"
    assert refreshed.refresh_token == "new-refresh"
    assert len(requests) == 2
    authorization_data = requests[0].data
    refresh_data = requests[1].data
    assert isinstance(authorization_data, bytes)
    assert isinstance(refresh_data, bytes)
    assert parse_qs(authorization_data.decode())["grant_type"] == ["authorization_code"]
    assert parse_qs(refresh_data.decode())["grant_type"] == ["refresh_token"]


@pytest.mark.parametrize(
    "invalid_field",
    ("empty_access_token", "unexpected_token_type", "nonpositive_expiry"),
)
def test_token_exchange_rejects_invalid_security_fields(
    invalid_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unusable credentials and unsupported token semantics."""
    values: dict[str, object] = {
        "access_token": "synthetic-access",
        "token_type": "bearer",
        "expires_in": 3600,
        "refresh_token": "synthetic-refresh",
    }
    if invalid_field == "empty_access_token":
        values["access_token"] = ""
    elif invalid_field == "unexpected_token_type":
        values["token_type"] = "mac"
    else:
        values["expires_in"] = 0
    payload = json.dumps(values).encode()

    class Response:
        """Provide one context-managed invalid token response."""

        def __enter__(self) -> Self:
            """Enter the response context."""
            return self

        def __exit__(self, *_: object) -> None:
            """Exit the response context."""

        def read(self) -> bytes:
            """Return the invalid token payload."""
            return payload

    def urlopen(_: Request, timeout: int) -> Response:
        """Return the invented invalid response without network access."""
        assert timeout == 30
        return Response()

    monkeypatch.setattr(oura_auth, "urlopen", urlopen)

    with pytest.raises(OuraAuthorizationError, match="invalid token response"):
        oura_auth.exchange_authorization_code(
            code="synthetic-code",
            client_id="synthetic-client",
            client_secret="synthetic-secret",
            redirect_uri="http://localhost/callback",
        )
