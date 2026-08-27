# Local Oura Setup and Sync

The Oura integration retrieves data directly to ignored local files. Credentials,
tokens, API payloads, snapshots, fingerprints, and derived personal data must not
be committed, pasted into issues or pull requests, or included in test output.

## Register an Oura application

Sign in to the [Oura developer portal](https://cloud.ouraring.com/) and create an
application with these values:

| Field | Value |
| --- | --- |
| Display Name | `Cycle Forecast` |
| Description | `A private, single-user research application that retrieves my Oura sleep and readiness data for local menstrual-cycle forecasting experiments. Data remains on my device. This is not a medical device.` |
| Contact Email | An email address where Oura may contact the application owner |
| Website | `https://github.com/emmatyy19/cycle-forecast` |
| Privacy Policy | `https://github.com/emmatyy19/cycle-forecast/blob/main/docs/design/003-oura-data-contract.md` |
| Terms of Service | `https://github.com/emmatyy19/cycle-forecast/blob/main/LICENSE` |
| Redirect URI | `http://localhost:8765/callback` |

Select only the **Daily** scope. Leave Email, Personal, Heartrate, Tag, Workout,
Session, SpO2, Ring Configuration, Stress, and Heart Health unselected. Read and
accept the Oura API Agreement before creating the application.

The client ID identifies this application and is not a secret. The client secret
authenticates the application during token exchange and refresh; never commit,
share, log, or put it directly in a shell command.

## Preferred guided setup

After creating the Oura application, run:

```bash
uv run cycle-forecast oura-setup --timezone America/New_York
```

Use `America/Los_Angeles` instead when setting up from California. The command:

1. prompts visibly for the non-secret client ID
2. prompts invisibly for the client secret
3. stores both values in macOS Keychain
4. opens Oura authorization and validates the redirect
5. stores the resulting local OAuth token
6. retrieves the previous two local calendar dates in check-only mode
7. reports route, page, and document counts without saving health payloads

The localhost redirect still requires copying the complete failed-page address
back into the terminal, as described under authorization below. Application
registration in the Oura portal and physically syncing the Ring cannot be
automated.

## Manual Keychain setup

The remaining setup commands are useful for troubleshooting or migrating an
existing authorization. Guided `oura-setup` users may skip to
[Check local status](#check-local-status).

Add the Oura client secret with `-w` as the final option:

```bash
security add-generic-password \
  -a cycle-forecast \
  -s com.emmatyy19.cycle-forecast.oura-client-secret \
  -l "Cycle Forecast Oura Client Secret" \
  -U \
  -w
```

When macOS asks for the password data, paste the **Oura client secret**, not the
Mac login password. The input is not displayed. In this command,
`cycle-forecast` is only the Keychain item's account label; it is not the Oura
client ID.

Verify that the entry exists without printing its secret:

```bash
security find-generic-password \
  -a cycle-forecast \
  -s com.emmatyy19.cycle-forecast.oura-client-secret
```

## Manual credential loading

Set the non-secret client ID in the current terminal, then retrieve the secret
from Keychain into an environment variable:

```bash
export OURA_CLIENT_ID="your-oura-client-id"
export OURA_CLIENT_SECRET="$(
  security find-generic-password \
    -a cycle-forecast \
    -s com.emmatyy19.cycle-forecast.oura-client-secret \
    -w
)"
```

Confirm that both variables are populated without displaying either value:

```bash
test -n "$OURA_CLIENT_ID" && echo "Client ID loaded"
test -n "$OURA_CLIENT_SECRET" && echo "Client secret loaded"
```

Environment variables are used only because the current command interface
accepts them as a Keychain fallback. Other processes running as the same user
may be able to inspect the environment while the command runs.

## Authorize the Oura account

```bash
uv run cycle-forecast oura-authorize
```

The command opens Oura's authorization page and requests only Daily access.
After approval, the browser navigates to the registered localhost redirect. It
is expected to show `localhost refused to connect` because the application does
not run a callback web server.

Do not reload the failed page. Select the complete browser address with
`Command` + `L`, copy it, return to the terminal, and paste it at this hidden
prompt:

```text
Paste the full redirected URL:
```

The URL must include one `code` and the matching `state`. Treat the complete URL
as a secret: the authorization code is short-lived and single-use. If the
command is cancelled, token exchange fails, or the code expires, rerun
`oura-authorize` to obtain a new redirect URL.

The command validates the state, exchanges the code, and writes the access and
refresh tokens to `data/private/oura/oauth-token.json`. The path is Git-ignored
and the file has owner-only `0600` permissions, but its contents are plaintext,
not encrypted or stored in Keychain.

After successful authorization, remove the temporary environment variables:

```bash
unset OURA_CLIENT_ID OURA_CLIENT_SECRET
```

## Check local status

At any time, inspect the integration without network access or health values:

```bash
uv run cycle-forecast oura-status
```

The command reports whether application credentials are available, whether the
local token is valid, its non-secret expiry time, the number of private snapshot
files, and the latest requested snapshot date. It never prints credentials,
tokens, document identifiers, or health measurements.

## Privacy-safe live check

Open the Oura mobile app and wait for the Ring to finish syncing. Validate a
small current date range without saving response payloads:

```bash
uv run cycle-forecast oura-sync \
  --start-date 2025-01-01 \
  --end-date 2025-01-02 \
  --timezone America/New_York \
  --check-only
```

Replace the synthetic example dates with the intended local range. Use the IANA
timezone active during retrieval. The output contains only route, page,
document-count, and date-bound metadata. It never prints health measurements,
document IDs, tokens, or response bodies.

## Historical import

The first saved sync requires an explicit start date, normally the date Ring use
began:

```bash
uv run cycle-forecast oura-sync \
  --start-date 2020-01-01 \
  --timezone America/New_York
```

When `--end-date` is omitted, the command uses the current local date. Every
Oura route is retrieved with bounded date parameters, all pagination is
completed, and each validated collection becomes a separate immutable snapshot
under `data/private/oura/snapshots/`.

## Morning incremental sync

After opening the Oura app and completing the Ring sync, run:

```bash
uv run cycle-forecast oura-sync --timezone America/New_York
```

The command validates existing snapshot fingerprints and starts at the most
recent requested end date. Refetching that one overlapping day captures late or
corrected Oura records in a new immutable snapshot. It never overwrites an older
snapshot.

Access tokens refresh automatically when they are within five minutes of expiry.
Oura refresh tokens are single-use, so the replacement access and refresh tokens
are written atomically. The refresh operation loads application credentials from
macOS Keychain. Environment variables remain an explicit fallback.

## Troubleshooting and revocation

- **`localhost refused to connect`:** Expected. Copy the complete address from
  the browser into the waiting terminal prompt.
- **OAuth state mismatch:** Discard the redirect URL and restart authorization.
- **Invalid or expired authorization code:** Codes are short-lived and
  single-use. Restart authorization rather than reusing the URL.
- **Redirect URI error:** The URI in Oura and the command must match exactly,
  including scheme, port, path, and trailing slash behavior.
- **401 or invalid token:** Reauthorize if refresh cannot replace the token.
- **403:** Check the Oura membership and granted Daily scope.
- **429:** Wait before retrying; do not bypass the API rate limit.
- **Pydantic validation error:** Do not weaken validation. Compare the response
  schema with Oura's current documentation, update the boundary model and
  synthetic regression tests, and never paste the private response into an
  issue or pull request.

Revoke application access from the Oura account or developer portal if a token
or client secret may have been exposed. Rotate the registered client secret,
delete the local token file, and authorize again. Remove the Keychain entry when
it is no longer needed:

```bash
security delete-generic-password \
  -a cycle-forecast \
  -s com.emmatyy19.cycle-forecast.oura-client-secret
```

The API boundary models follow Oura's versioned
[OpenAPI 1.35 specification](https://api.ouraring.com/v2/static/json/openapi-1.35.json)
and [OAuth documentation](https://cloud.ouraring.com/docs/authentication). An
undocumented response field or incompatible schema change stops ingestion so the
specification and models can be reviewed explicitly.
