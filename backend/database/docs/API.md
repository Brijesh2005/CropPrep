# Enterprise API reference (Phase 10)

All routes are mounted under `/api/v1`. They extend the Phase 8 API without
duplicating existing paths.

Authentication is `Authorization: Bearer <access_token>`. RBAC uses five roles
(ascending): `user` < `analyst` < `dataset_manager` < `admin` < `super_admin`.
`require_role(X)` allows any role with equal or higher privilege.

## Auth (enterprise)

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/auth/password/change` | user | Verify current password, set new one, revoke other tokens/sessions |
| POST | `/auth/password/reset` | public | Issue a single-use reset token (no user enumeration) |
| POST | `/auth/password/reset/confirm` | public | Consume token and set a new password |
| POST | `/auth/verify-email/request` | user | Issue an email-verification token |
| POST | `/auth/verify-email/confirm` | public | Verify the email address |
| GET | `/auth/sessions` | user | List the user's active sessions |
| DELETE | `/auth/sessions/{session_id}` | user | Revoke a specific session |
| DELETE | `/auth/sessions` | user | Revoke all other sessions |

## Users (enterprise)

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET/PUT | `/users/preferences` | user | Read / update preferences |
| GET | `/users/locations` | user | List saved locations |
| POST | `/users/locations` | user | Save a location |
| PUT | `/users/locations/{id}/primary` | user | Mark a location primary |
| DELETE | `/users/locations/{id}` | user | Delete a location |

## Predictions (enterprise)

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/predictions/history/search` | user | Filtered history search (crop/season/year/district/taluk/village/dates/min confidence) |

## Notifications

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/notifications` | user | Inbox (paged, unread-only option) |
| GET | `/notifications/unread-count` | user | Unread count |
| POST | `/notifications/{id}/read` | user | Mark one as read |
| POST | `/notifications/read-all` | user | Mark all as read |
| POST | `/notifications` | admin | Send a notification to a user |

## Feedback

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/feedback` | optional | Submit feedback (rating 1–5, category, comment) |
| GET | `/feedback` | admin | Filtered feedback queue |
| POST | `/feedback/{id}/resolve` | admin | Resolve with a note |

## Admin (enterprise)

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/admin/enterprise/dashboard` | admin | Aggregated analytics (users, predictions, crops, regions, confidence, feedback) |
| GET | `/admin/enterprise/analytics` | admin | Analytics filtered by season/year |
| GET | `/admin/enterprise/audit` | admin | Filtered audit trail |

## Registry

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/registry/models` | dataset_manager | Register a model version |
| GET | `/registry/models` | user | List model versions |
| POST | `/registry/models/activate` | dataset_manager | Activate a model version |
| POST | `/registry/datasets` | dataset_manager | Register a dataset version |
| POST | `/registry/datasets/validate` | dataset_manager | Mark a dataset version valid/invalid |
| GET | `/registry/datasets` | user | List dataset versions |

## Catalog

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/catalog/crops` | public | Active crops (searchable) |
| POST | `/catalog/crops` | dataset_manager | Create a crop |
| GET | `/catalog/seasons` | public | Active seasons |
| POST | `/catalog/seasons` | dataset_manager | Create a season |

## Spatial

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/spatial/resolve?lon&lat` | public | Resolve coordinates → village/taluk/district (Redis-cached) |
| GET | `/spatial/boundaries?level` | public | List boundaries by level / parent |
| GET | `/spatial/boundaries/counts` | public | Boundary counts by level |
| GET | `/spatial/locations?lon&lat` | public | Nearest spatial locations |
| POST | `/spatial/locations` | dataset_manager | Create a spatial location |
| POST | `/spatial/resolve/admin` | user | Resolve a coordinate payload |

## Experiments

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/experiments` | admin | Create a research experiment |
| GET | `/experiments` | user | List experiments |
| POST | `/experiments/{id}/start` | admin | Mark running |
| POST | `/experiments/{id}/finish` | admin | Mark completed with metrics |
| POST | `/experiments/{id}/fail` | admin | Mark failed |

## Config store

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/config-store/{key}` | public | Read a config value |
| PUT | `/config-store` | admin | Upsert a versioned config key/value |
| GET | `/config-store` | admin | List config (secrets masked) |

## Error codes

Enterprise routes use the standard handlers: `400` `B-VALID-002` for
service-level `ValueError` (e.g. duplicate registry/catalog entries), `401`
`B-AUTH-001/003` for auth/token failures, `403` `B-AUTH-002` for RBAC
denials, `422` `B-VALID-001` for request validation.
