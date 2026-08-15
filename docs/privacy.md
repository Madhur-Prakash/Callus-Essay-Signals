[Home](../README.md) · [Docs index](README.md) · [Architecture](architecture.md) ·
[Methodology](detection-methodology.md) · [Dataset](dataset.md) ·
[Evaluation](evaluation.md) · [API](api.md) — **Privacy** ·
[Limitations](limitations.md)

---

# Privacy

Admissions essays are sensitive personal documents. They describe illness,
immigration status, family finances, grief, disability and religion — the
categories a person is least likely to want retained by a service they used once.
This system is built on the assumption that the safe default is to keep nothing.

## What happens to a submitted essay

By default (`SAVE_ESSAYS=false`):

| Data                                      | Stored | Logged |
| ----------------------------------------- | ------ | ------ |
| Essay text                                | **No** | **No** |
| Text of individual sentences              | **No** | **No** |
| Sentence character offsets                | Yes    | No     |
| Sentence scores and classifications       | Yes    | No     |
| Aggregate statistics (perplexity, counts)  | Yes    | Yes    |
| Verdict, probabilities, confidence         | Yes    | Yes    |
| Generated evidence for flagged sentences   | Yes    | No     |
| Timestamps, detector and model versions    | Yes    | Yes    |
| SHA-256 content hash                       | Yes    | Truncated |

The content hash is a one-way digest of `essay + detector_version + model_version`.
It cannot be reversed into the essay; it exists so that re-analysing the same draft
hits the cache.

## Why offsets are kept but text is not

Because it makes the privacy default *usable* rather than a downgrade. The
sentence-level highlighting needs to know where each sentence is, not what it says.
When a stored analysis is reloaded, the frontend slices the essay the user still
has in their editor using the stored offsets — so the highlighting works fully
without the server ever having held the text.

## Structural guarantees, not conventions

**Text cannot reach a log.** `app/core/logging.safe_text_meta()` is the only
sanctioned way to describe user text in a log record, and it emits lengths and a
truncated digest — never characters. `scrub()` defensively removes email
addresses, phone numbers and SSNs from any string that might echo user input, and
is applied to exception messages before they are logged. `logifyx` is configured
with `mask=True`.

There is a test for this that does not take the code's word for it: it submits an
essay containing a distinctive phrase, then greps every `.log` file on disk for
that phrase (`test_essay_text_never_reaches_the_log_file`).

**Text is never echoed back.** The analysis response has no top-level `text`
field, and `GET /essays/{id}` pops `text` before returning.

**Request logging is path-only.** The HTTP middleware logs method, path, status and
duration — never query strings or bodies.

## Per-request opt-out

`POST /api/v1/analysis` accepts `save: false`. The server setting is the
**ceiling**: a request can opt out of storage, never into it. If the operator set
`SAVE_ESSAYS=false`, no request can cause an essay to be stored.

## Deletion

```
DELETE /api/v1/analysis/{analysis_id}
```

Removes the analysis and its per-sentence results. Provided so a submission can be
withdrawn after the fact.

Analyses older than `ANALYSIS_RETENTION_DAYS` (default 30) are purged at startup.
Set it to `0` to disable automatic purging.

## What the frontend tells the user

The editor shows the storage state, read live from `GET /api/v1/essays/privacy`
rather than hard-coded. A privacy notice that can drift out of step with the
server's actual configuration is worse than none, because it is trusted.

When storage is enabled, a "Do not save my essay" checkbox appears. When it is
disabled, the UI states plainly that the essay is not stored.

## Input limits

| Limit                | Default   | Why                                        |
| -------------------- | --------- | ------------------------------------------ |
| `MAX_ESSAY_CHARS`    | 60,000    | ~10,000 words; beyond this, analysis is slow |
| `MIN_ESSAY_CHARS`    | 200       | below this there are too few sentences to measure |
| `MAX_REQUEST_BYTES`  | 1,000,000 | rejected by middleware before the body is parsed |
| Rate limit           | 30/min    | per client IP (`X-Forwarded-For` aware)     |

## Third-party data flow

**No hosted model is called during analysis.** The Groq client exists only in
`ml/generation/` for offline dataset construction, is never imported by `app/`, and
there is no code path from a request handler to it. A submitted essay does not
leave the server.

## Evaluation reports

The failure report shows excerpts of misclassified documents so a reviewer can see
what went wrong. Excerpts are **withheld** for any document whose source is
`ingested_real` — operator-supplied real essays are never reproduced in reports,
and the report says so in `excerpt_withheld_reason`.

`.gitignore` excludes `backend/data/raw/human/*.txt`, so real essays cannot be
committed by accident.

## Security posture

- CORS restricted to configured origins; no credentials, narrow method/header allowlists.
- Body-size rejection before parsing.
- Fixed-window rate limiting (Redis-backed when available, per-process otherwise).
- Pydantic validation on every request.
- Unicode normalisation strips zero-width, bidi and control characters.
- No stack traces in responses — every error returns `{error: {code, message}}`.
- No hardcoded credentials anywhere; `.env` is gitignored, `.env.example` has no secrets.
- nginx sets `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and hides its version.
- The MongoDB dev container has no credentials on purpose — the port is bound to
  localhost and a bootstrap corpus holds nothing sensitive. **Any deployment must
  set `MONGO_INITDB_ROOT_*` and supply credentials through `MONGODB_URL` as a
  secret**, never in the compose file.

## Not implemented

Deliberately out of scope, and named so nobody assumes otherwise:

- **No authentication.** There are no user accounts, so there is no per-user
  isolation. Do not expose this to the internet as-is.
- **No encryption at rest** beyond whatever MongoDB is configured to do.
- **No audit log** of who read which analysis.
- **No PII detection inside essays.** Essays are treated as opaque sensitive text;
  names and places inside them are neither extracted nor redacted.

## If you deploy this

1. Keep `SAVE_ESSAYS=false` unless you have a specific reason and a consent flow.
2. Put authentication in front of it.
3. Set real MongoDB credentials via secrets.
4. Set `CORS_ORIGINS` to your exact frontend origin.
5. Terminate TLS in front of the API.
6. Decide a retention period and set `ANALYSIS_RETENTION_DAYS`.
7. Read [limitations.md](limitations.md) before showing a verdict to anyone who
   might act on it.
