"""Authorize a local Oura API client without exposing OAuth secrets."""

import os
import secrets
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Final, Literal
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import keyring
from keyring.errors import KeyringError
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    StringConstraints,
    ValidationError,
)

from cycle_forecast.data.private_files import ensure_private_directory

OURA_AUTHORIZE_URL: Final = "https://cloud.ouraring.com/oauth/authorize"
OURA_TOKEN_URL: Final = "https://api.ouraring.com/oauth/token"
OURA_CLIENT_ID_ENV: Final = "OURA_CLIENT_ID"
OURA_CLIENT_SECRET_ENV: Final = "OURA_CLIENT_SECRET"
DEFAULT_OURA_TOKEN_PATH = Path("data/private/oura/oauth-token.json")
OURA_KEYCHAIN_ACCOUNT: Final = "cycle-forecast"
OURA_CLIENT_ID_SERVICE: Final = "com.emmatyy19.cycle-forecast.oura-client-id"
OURA_CLIENT_SECRET_SERVICE: Final = "com.emmatyy19.cycle-forecast.oura-client-secret"
OAUTH_CODE_RESPONSE_TYPE: Final = "code"
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


class OAuthGrantType(StrEnum):
    """Identify the OAuth token grant represented by a token request."""

    AUTHORIZATION_CODE = auto()
    REFRESH_TOKEN = auto()


class OuraScope(StrEnum):
    """Identify an Oura API permission requested during authorization."""

    DAILY = auto()


class OuraAuthorizationError(RuntimeError):
    """Indicate that local Oura OAuth authorization failed."""


class _TokenResponse(BaseModel):
    """Validate the documented and live-observed Oura OAuth token response.

    Oura's authentication guide documents the four core token fields. Live V2
    authorization also returns the granted ``scope`` and a nullable ``id_token``.
    See https://cloud.ouraring.com/docs/authentication.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    access_token: NonEmptyString
    token_type: Literal["bearer"]
    expires_in: PositiveInt
    refresh_token: NonEmptyString
    scope: str | None = None
    id_token: str | None = None


class _StoredToken(BaseModel):
    """Validate the versioned local token-file schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["oura-oauth-token-v1"] = "oura-oauth-token-v1"
    access_token: str
    refresh_token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class OuraToken:
    """Represent locally stored Oura OAuth credentials.

    Parameters
    ----------
    access_token
        Short-lived bearer token.
    refresh_token
        Token used to obtain a replacement bearer token.
    expires_at
        UTC instant when the bearer token expires.
    """

    access_token: str
    refresh_token: str
    expires_at: datetime


def load_oauth_application() -> tuple[str, str]:
    """Load Oura OAuth application credentials from environment or Keychain.

    Returns
    -------
    tuple[str, str]
        Client ID and client secret.

    Raises
    ------
    OuraAuthorizationError
        If either environment variable is absent.
    """
    client_id = os.environ.get(OURA_CLIENT_ID_ENV)
    client_secret = os.environ.get(OURA_CLIENT_SECRET_ENV)
    if not client_id and not client_secret:
        try:
            client_id = keyring.get_password(
                OURA_CLIENT_ID_SERVICE, OURA_KEYCHAIN_ACCOUNT
            )
            client_secret = keyring.get_password(
                OURA_CLIENT_SECRET_SERVICE, OURA_KEYCHAIN_ACCOUNT
            )
        except KeyringError as error:
            raise OuraAuthorizationError("could not read macOS Keychain") from error
    if not client_id or not client_secret:
        raise OuraAuthorizationError(
            "store Oura application credentials with oura-setup or set "
            f"{OURA_CLIENT_ID_ENV} and {OURA_CLIENT_SECRET_ENV}"
        )
    return client_id, client_secret


def save_oauth_application(*, client_id: str, client_secret: str) -> None:
    """Store Oura application credentials in the operating-system keyring.

    Raises
    ------
    ValueError
        If either credential is empty.
    OuraAuthorizationError
        If the credential store cannot save the values.
    """
    if not client_id or not client_secret:
        raise ValueError("Oura client ID and client secret must be non-empty")
    try:
        keyring.set_password(OURA_CLIENT_ID_SERVICE, OURA_KEYCHAIN_ACCOUNT, client_id)
        keyring.set_password(
            OURA_CLIENT_SECRET_SERVICE, OURA_KEYCHAIN_ACCOUNT, client_secret
        )
    except KeyringError as error:
        raise OuraAuthorizationError("could not write macOS Keychain") from error


def build_authorization_url(
    *, client_id: str, redirect_uri: str, state: str, scopes: tuple[OuraScope, ...]
) -> str:
    """Build an Oura authorization URL for a local OAuth flow."""
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": OAUTH_CODE_RESPONSE_TYPE,
            "scope": " ".join(scopes),
            "state": state,
        }
    )
    return f"{OURA_AUTHORIZE_URL}?{query}"


def parse_authorization_redirect(*, redirected_url: str, expected_state: str) -> str:
    """Validate a pasted OAuth redirect and return its authorization code."""
    query = parse_qs(urlparse(redirected_url).query)
    if query.get("state") != [expected_state]:
        raise OuraAuthorizationError("OAuth state did not match")
    if "error" in query:
        raise OuraAuthorizationError(f"Oura authorization failed: {query['error'][0]}")
    codes = query.get("code")
    if codes is None or len(codes) != 1 or not codes[0]:
        raise OuraAuthorizationError("OAuth redirect did not contain one code")
    return codes[0]


def _post_token_form(*, values: dict[str, str]) -> _TokenResponse:
    """Post an OAuth token form and validate its response."""
    request = Request(
        OURA_TOKEN_URL,
        data=urlencode(values).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    except OSError as error:
        raise OuraAuthorizationError("Oura token request failed") from error
    try:
        return _TokenResponse.model_validate_json(payload)
    except ValueError as error:
        raise OuraAuthorizationError(
            "Oura returned an invalid token response"
        ) from error


def exchange_authorization_code(
    *, code: str, client_id: str, client_secret: str, redirect_uri: str
) -> OuraToken:
    """Exchange one Oura authorization code for local tokens."""
    response = _post_token_form(
        values={
            "grant_type": OAuthGrantType.AUTHORIZATION_CODE,
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }
    )
    return OuraToken(
        access_token=response.access_token,
        refresh_token=response.refresh_token,
        expires_at=datetime.now(tz=UTC) + timedelta(seconds=response.expires_in),
    )


def refresh_access_token(
    *, refresh_token: str, client_id: str, client_secret: str
) -> OuraToken:
    """Refresh an expired Oura bearer token."""
    response = _post_token_form(
        values={
            "grant_type": OAuthGrantType.REFRESH_TOKEN,
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }
    )
    return OuraToken(
        access_token=response.access_token,
        refresh_token=response.refresh_token,
        expires_at=datetime.now(tz=UTC) + timedelta(seconds=response.expires_in),
    )


def save_token(*, token: OuraToken, path: Path) -> None:
    """Atomically save an OAuth token with owner-only permissions."""
    try:
        ensure_private_directory(directory=path.parent)
    except OSError as error:
        raise OuraAuthorizationError(
            f"could not secure token directory {path.parent}"
        ) from error
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            stored_token = _StoredToken(
                access_token=token.access_token,
                refresh_token=token.refresh_token,
                expires_at=token.expires_at.astimezone(UTC),
            )
            handle.write(stored_token.model_dump_json() + "\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise OuraAuthorizationError(f"could not save token to {path}") from error


def load_token(*, path: Path) -> OuraToken:
    """Load a locally stored OAuth token."""
    try:
        raw = _StoredToken.model_validate_json(path.read_bytes())
        token = OuraToken(
            access_token=raw.access_token,
            refresh_token=raw.refresh_token,
            expires_at=raw.expires_at,
        )
    except ValidationError as error:
        if any(issue["loc"] == ("schema_version",) for issue in error.errors()):
            raise OuraAuthorizationError("unsupported stored token schema") from error
        raise OuraAuthorizationError(f"could not load token from {path}") from error
    except (OSError, ValueError) as error:
        raise OuraAuthorizationError(f"could not load token from {path}") from error
    if token.expires_at.tzinfo is None or token.expires_at.utcoffset() is None:
        raise OuraAuthorizationError("stored token expiry must be timezone-aware")
    return token


def authorize_interactively(
    *,
    redirect_uri: str,
    input_fn: Callable[[str], str],
    token_path: Path = DEFAULT_OURA_TOKEN_PATH,
) -> OuraToken:
    """Authorize through a browser and pasted local redirect URL."""
    client_id, client_secret = load_oauth_application()
    state = secrets.token_urlsafe(32)
    url = build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        scopes=(OuraScope.DAILY,),
    )
    webbrowser.open(url)
    redirected_url = input_fn("Paste the full redirected URL: ").strip()
    code = parse_authorization_redirect(
        redirected_url=redirected_url, expected_state=state
    )
    token = exchange_authorization_code(
        code=code,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )
    save_token(token=token, path=token_path)
    return token
