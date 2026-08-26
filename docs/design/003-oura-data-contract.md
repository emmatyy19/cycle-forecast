# Design 003: Privacy-Safe Oura Data Contract

## Status

Accepted

## Context

Phase B uses Oura Ring observations to update the history-only forecast. Oura
API V2 is the sole supported wearable source. Apple Health, manual daily entry,
and multi-user ingestion are outside this contract.

The source data is personal health data. Raw responses, normalized observations,
credentials, logs containing payloads, and derived fingerprints remain local and
git-ignored. Only invented fixtures may be committed.

## Source and authentication

The primary source is Oura API V2 using OAuth 2 authorization for one account.
The application must request only the scopes required by its configured routes;
it must not request `email` or `personal`. Client secrets, access tokens, refresh
tokens, authorization codes, and webhook secrets must never appear in command
arguments, configuration committed to Git, logs, exceptions, fixtures, or model
artifacts.

Historical ingestion uses bounded API date ranges. Ongoing local ingestion runs
after the user opens the Oura app and completes the morning Ring sync. Polling on
demand is preferred to webhooks because this project has no public service. An
Oura account export may be retained as a private recovery source, but it is not
the canonical ongoing input and must pass a separately versioned loader before
being combined with API data.

Authentication metadata is separate from health data. The application may
record non-secret token expiry and granted scope names locally, but the dataset
must remain readable and testable without credentials after ingestion.

## Allowed API records

The initial contract allows only records needed to represent overnight
physiology, sleep, recovery, and data completeness:

- detailed sleep sessions
- daily sleep summaries
- daily readiness summaries
- time-series heart rate when a later feature design justifies it

The `daily` scope covers the summary and sleep records; `heartrate` is requested
only when time-series heart rate ingestion is enabled. Activity, workout, tag,
session, SpO2, stress, cardiovascular-age, resilience, and personal-information
records are excluded initially. Adding a route or scope requires a versioned
contract change that explains its forecasting value.

All fields returned by an allowed route may be preserved in the private raw
snapshot so the source can be audited. A later normalized or modeling dataset
must use an explicit field allowlist; it may not automatically expose every raw
field.

## External boundary models

Every allowed Oura response is loaded through a Pydantic model before the
application treats it as valid. These boundary models mirror Oura's documented
wire format exactly; they are not project domain models and must not rename,
flatten, derive, or discard source fields.

The initial models use Oura's versioned
[OpenAPI 1.35 specification](https://api.ouraring.com/v2/static/json/openapi-1.35.json),
whose API `info.version` is 2.0. The model module and each top-level response
model must link to that specification and the corresponding route in the
[rendered API V2 documentation](https://cloud.ouraring.com/v2/docs). The source
specification version is a named constant recorded in every snapshot.

Required versus optional fields, nullable values, nested objects, arrays,
documented enums, numeric types, date and datetime formats, and pagination must
match that specification. Pydantic models use strict validation and forbid extra
fields. This deliberately turns an undocumented or newly documented upstream
change into an explicit ingestion failure rather than silently ignoring data.
Adopting a new Oura specification requires reviewing its schema diff, updating
the models and synthetic contract tests, and changing the recorded specification
version.

The response body is parsed directly with Pydantic's JSON validation. Only after
successful boundary validation may project code construct a normalized domain
record. Domain validation may impose additional forecasting invariants, but it
must remain separate so an Oura schema mismatch cannot be confused with a
project-specific rejection.

## Raw snapshot contract

Each successful retrieval writes an immutable local snapshot containing:

- contract and snapshot schema versions
- Oura API and OpenAPI specification versions and route name
- timezone-aware retrieval start and completion instants in UTC
- the IANA timezone active at retrieval
- requested inclusive date bounds and pagination metadata
- the unmodified response documents in their source order
- a deterministic fingerprint of the canonical snapshot content

The snapshot must preserve Oura document IDs, `day` values, timestamps, UTC
offsets, nulls, and nested measurement series exactly as returned. IDs are
retained locally only for deduplication, correction detection, and provenance;
they must be removed from modeling tables and model packages.

Snapshots live under a git-ignored private-data directory. Writes must be atomic,
and an existing snapshot must not be replaced silently. A failed or incomplete
pagination sequence produces no valid snapshot.

## Validation and update semantics

The loader must reject an unsupported schema or API version, unexpected route,
malformed timestamp or date, missing document ID, duplicate document ID within a
snapshot, invalid pagination sequence, response outside the requested bounds, or
content that cannot be fingerprinted deterministically. It must not silently
sort, fill, interpolate, coerce malformed values, or convert missing values to
zero.

Oura may publish a record after its measurement period or revise a previously
published record. Repeated ingestion therefore keeps both retrieval provenance
and the newest document content. A correction never rewrites an older immutable
snapshot. Alignment code must use the version demonstrably available at each
historical prediction cutoff; when availability cannot be reconstructed, that
record is unavailable for leakage-safe backtesting.

## Dates, timezones, and travel

Timestamps retain their source offsets and are also convertible to UTC instants.
Oura's supplied `day` remains a source field and must not be recreated by first
converting timestamps to a fixed home timezone.

Every retrieval and prediction records the active IANA timezone, normally
`America/New_York` and, during west-coast travel, `America/Los_Angeles`. The
actual local timezone at the event is used rather than permanently assigning all
data to New York time. Daylight-saving transitions follow the IANA timezone
database. An offset alone is insufficient because it does not encode future or
historical daylight-saving rules.

The next alignment design will define how sleep sessions spanning midnight,
travel days, and Oura daily summaries join prediction dates. Until then, raw
timestamps, offsets, `day`, and retrieval timezone must all remain available.

## Privacy and repository boundaries

The following must never be committed or included in CI:

- OAuth credentials or tokens
- Oura exports, API responses, or normalized personal observations
- account or Oura document identifiers
- real dates, timestamps, measurements, fingerprints, or identifying paths
- fitted models, metrics, plots, notebooks, or logs derived from personal data

Committed tests use invented documents with fictional identifiers, timestamps,
and physiologically plausible but synthetic values. Examples must not be made by
shifting, rounding, aggregating, or otherwise transforming personal records.

Modeling artifacts may contain aggregated parameters derived from private data
and therefore remain git-ignored even after direct identifiers are removed.

## References

- [Oura API V2 documentation](https://cloud.ouraring.com/v2/docs)
- [Oura data export documentation](https://support.ouraring.com/hc/en-us/articles/360025441594-Export-Share-Your-Oura-Data)
