[Home](../README.md) · [Docs index](README.md) · [Architecture](architecture.md) ·
[Methodology](detection-methodology.md) · [Dataset](dataset.md) ·
[Evaluation](evaluation.md) - **API** · [Privacy](privacy.md) ·
[Limitations](limitations.md)

---

# API reference

Base path `/api/v1`. Interactive docs at `/docs` (Swagger) and `/redoc`; the
OpenAPI schema is generated from the Pydantic models at `/openapi.json`.

## Error envelope

Every error - validation, application, unhandled - returns the same shape. Users
never see a Python traceback.

```json
{
  "error": {
    "code": "essay_too_short",
    "message": "The essay is 84 characters; at least 200 are needed for a meaningful analysis.",
    "details": { "length": 84, "minimum": 200 }
  }
}
```

| Code                      | HTTP | Meaning                                     |
| ------------------------- | ---- | ------------------------------------------- |
| `essay_empty`             | 422  | No text supplied                            |
| `essay_too_short`         | 422  | Below `MIN_ESSAY_CHARS`                     |
| `essay_too_long`          | 413  | Above `MAX_ESSAY_CHARS`                     |
| `request_too_large`       | 413  | Body above `MAX_REQUEST_BYTES`              |
| `validation_error`        | 422  | Request body failed schema validation       |
| `rate_limit_exceeded`     | 429  | Includes `retry_after_seconds`              |
| `analysis_not_found`      | 404  | No such analysis id                         |
| `analysis_timeout`        | 504  | Exceeded the 120 s inline limit             |
| `model_not_trained`       | 503  | Artifacts missing - run the training pipeline |
| `model_unavailable`       | 503  | The language model could not be loaded      |
| `persistence_unavailable` | 503  | MongoDB is down; analysis still works       |
| `internal_error`          | 500  | Logged server-side; no internals returned   |

Every response carries `X-Response-Time-Ms`. Requests may supply `X-Request-ID`,
which is propagated into log records.

---

## `POST /analysis`

Analyse an essay. This is the only endpoint that does work.

**Request**

```json
{
  "text": "The robot never worked. That is the honest summary of my sophomore year...",
  "save": false,
  "async_mode": false
}
```

| Field        | Type          | Notes                                                                 |
| ------------ | ------------- | --------------------------------------------------------------------- |
| `text`       | string        | Required. Bounded by `MIN/MAX_ESSAY_CHARS`.                            |
| `save`       | bool \| null  | Optional opt-out. The server's `SAVE_ESSAYS` is the ceiling - this can decline storage, never request it. |
| `async_mode` | bool          | Route through Kafka. Ignored when Kafka is disabled (the default).      |

**Response `200`** - abridged; see `/docs` for the full schema.

```json
{
  "analysis_id": "9f2c1b...",
  "status": "completed",

  "classification": "ai_polished",
  "label": "Potentially AI-polished",
  "description": "The patterns are most consistent with human writing that has been edited...",
  "confidence": "moderate",
  "confidence_score": 0.71,
  "probabilities": { "human": 0.18, "ai_generated": 0.11, "ai_polished": 0.71 },
  "margin": 0.53,
  "abstained": false,
  "abstain_reason": null,

  "summary": {
    "n_words": 1243, "n_sentences": 62, "n_paragraphs": 7,
    "sentences_scored": 62, "flagged_sentences": 9,
    "uncertain_sentences": 7, "human_like_sentences": 46,
    "flagged_paragraphs": 2, "flagged_share": 0.145,
    "statistics": {
      "perplexity": 31.8, "fraction_top1_tokens": 0.34,
      "mean_token_logprob": -3.42, "mean_token_entropy": 4.11,
      "sentence_length_cv": 0.53, "burstiness_index": -0.31,
      "root_type_token_ratio": 9.6, "trigram_repeat_ratio": 0.012,
      "pos_template_repeat_ratio": 0.08, "max_style_shift": 1.15,
      "style_changepoints": 1, "flesch_reading_ease": 61.2
    },
    "lm_tokens_scored": 1680, "lm_windows": 4,
    "segmentation_backend": "spacy"
  },

  "paragraphs": [
    {
      "paragraph_id": 2, "start": 812, "end": 1104,
      "score": 0.78, "max_sentence_score": 0.91, "human_likeness": 0.22,
      "classification": "likely_ai_assisted",
      "flagged_sentence_ids": [18, 19, 21],
      "sentence_ids": [17, 18, 19, 20, 21]
    }
  ],

  "sentences": [
    {
      "sentence_id": 18, "paragraph_id": 2,
      "start": 838, "end": 921,
      "text": "Through this transformative journey, I discovered...",
      "score": 0.88, "classification": "likely_ai_assisted",
      "confidence": "high", "n_words": 24,
      "features": { "lm_perplexity": 14.2, "ctx_z_mean_logprob": 2.41 },
      "evidence": {
        "meters": [
          {
            "key": "lm_predictability", "label": "Language-model predictability",
            "strength": 0.91, "level": "high",
            "value": 0.55, "display": "55.0%",
            "reference": "human training median 32.0%",
            "percentile_vs_human": 91,
            "detail": "Most words here were the reference model's own first choice.",
            "available": true
          }
        ],
        "statements": ["Perplexity is 14.2, substantially lower than the essay's own 31.8."],
        "measurements": [
          { "name": "Perplexity", "value": 14.2, "unit": "14.2", "reference": "essay median 31.8" }
        ],
        "model_contributions": [
          { "feature": "lm_frac_top1", "contribution": 0.42, "direction": "machine-like", "method": "linear" }
        ],
        "engine_version": "1.0.0"
      }
    }
  ],

  "evidence":  { "...": "same shape, for the whole essay" },
  "rhythm":    [{ "index": 0, "words": 18, "perplexity": 42.1, "mean_logprob": -3.9 }],
  "repetition": {
    "repeated_phrases": [{ "phrase": "transformative journey", "count": 2, "length": 2 }],
    "repeated_syntactic_templates": [{ "template": "PRON VERB DET ADJ NOUN", "sentence_count": 4 }]
  },

  "model": {
    "detector_version": "1.0.0", "model_version": "1.0.0",
    "features_version": "1.0.0", "data_regime": "bootstrap",
    "language_model": "distilgpt2",
    "language_model_role": "instrument for token probabilities; it does not classify the essay",
    "classifier": "hybrid::random_forest"
  },

  "timings": { "segmentation_ms": 41.2, "lm_scoring_ms": 812.4, "total_ms": 1104.8 },
  "content_hash": "3f9a...", "created_at": "2026-08-15T00:12:03Z",
  "persisted": false, "cached": false,
  "warnings": ["This detector is trained on the offline bootstrap corpus..."],
  "disclaimer": "AI detection is probabilistic. A flag is evidence for human review, not proof of authorship..."
}
```

**Response `202`** - only when Kafka is enabled and the essay is large:

```json
{
  "analysis_id": "9f2c1b...",
  "status": "queued",
  "poll_url": "/api/v1/analysis/9f2c1b...",
  "message": "This essay was queued for background analysis. Poll the returned URL for its status."
}
```

### Field notes

- `classification` ∈ `human` | `ai_generated` | `ai_polished` | `insufficient_evidence`.
- `abstained: true` means the system declined to name a class; `abstain_reason`
  says why in plain language.
- `sentences[].score` is `null` when the sentence model is unavailable - the field
  is never faked.
- `sentences[].evidence` is present only for flagged and uncertain sentences
  (score ≥ 0.40). Attaching it to every sentence would triple the response size
  for content the UI never shows.
- `sentences[].text` is `""` for a *stored* analysis fetched when
  `SAVE_ESSAYS=false`. Offsets are always present, so the client re-slices its own
  copy of the essay.
- `meters[].available: false` means no training reference exists for that
  measurement; `percentile_vs_human` is then `null` rather than invented.
- `paragraphs[].score` is the length-weighted mean of its sentence scores;
  `max_sentence_score` is reported separately because one strongly flagged
  sentence in an otherwise ordinary paragraph is the localised-edit signal.

---

## `GET /analysis/{id}`

Returns the full analysis when `status == "completed"`, otherwise a compact status
document (`queued` / `processing` / `failed`). `503 persistence_unavailable` when
MongoDB is down.

## `GET /analysis/{id}/sentences`

Per-sentence rows only. Useful for re-rendering highlighting without re-fetching
evidence.

## `GET /analysis?limit=20&skip=0`

Recent analyses, newest first. Metadata only; no essay text.

## `DELETE /analysis/{id}`

Deletes the analysis and its per-sentence results.

---

## `GET /health`

Per-component health, because this system degrades rather than failing whole.

```json
{
  "status": "degraded",
  "version": "1.0.0",
  "components": [
    { "name": "mongodb", "enabled": true, "available": true, "detail": "database=ai_essay_detector" },
    { "name": "redis", "enabled": false, "available": false, "detail": "disabled by configuration" },
    { "name": "kafka", "enabled": false, "available": false, "detail": "disabled by configuration; analyses run synchronously" },
    { "name": "language_model", "enabled": true, "available": true, "detail": "distilgpt2 on cpu (instrument only, does not classify)" },
    { "name": "spacy", "enabled": true, "available": true, "detail": "en_core_web_sm (spacy)" },
    { "name": "detector_model", "enabled": true, "available": true, "detail": "model_version=1.0.0 regime=bootstrap" },
    { "name": "corpus_reference", "enabled": true, "available": true, "detail": "cross-corpus similarity features" }
  ],
  "uptime_seconds": 412.8
}
```

`status` is `ok` when every enabled component is available, `degraded` when a
non-essential one is not, and `unavailable` only when the trained model is missing.

- `GET /health/live` - liveness, always `200`.
- `GET /health/ready` - `200` only when the detector can answer; `503` otherwise.

---

## `GET /model/info`

Active versions, metrics, feature importance, the model comparison table, and the
methodology block the frontend's "How it works" page renders. Serving the
methodology from the backend keeps the user-facing explanation from drifting out of
step with the code.

## `POST /model/reload`

Reloads artifacts from disk without restarting, and invalidates the Redis cache so
results are not served from a previous model version.

---

## `GET /evaluation`

The full evaluation report, failure analysis and dataset card - everything the
Research page renders. When the pipeline has not been run it returns
`{"available": false, "message": "..."}` rather than fabricating numbers.

Includes: overall and per-class metrics, confusion matrix, ROC/PR curves,
calibration curve and ECE, topic/generator/length generalisation slices, the bias
analysis with Wilson intervals, abstention behaviour, the four-way feature-set
comparison, permutation feature importance, and generated interpretation lines.

- `GET /evaluation/runs` - stored runs from MongoDB.
- `GET /evaluation/model-versions` - the model registry.

---

## `GET /essays/privacy`

What this server stores, read from live configuration. The frontend fetches this
rather than hard-coding a privacy claim.

- `GET /essays?limit=20` - stored essay metadata (never text).
- `GET /essays/{id}` - one metadata record (`text` is stripped).

---

## Example

```powershell
$body = @{ text = (Get-Content essay.txt -Raw); save = $false } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/analysis `
                  -Method Post -ContentType 'application/json' -Body $body |
  Select-Object classification, confidence, confidence_score
```

```bash
curl -s localhost:8000/api/v1/analysis \
  -H 'Content-Type: application/json' \
  -d "{\"text\": $(jq -Rs . < essay.txt)}" |
  jq '{classification, confidence, flagged: .summary.flagged_sentences}'
```
