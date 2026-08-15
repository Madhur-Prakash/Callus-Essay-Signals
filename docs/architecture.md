# Architecture

## System diagram

```
                        ┌─────────────────────────┐
                        │   React + Vite + TS     │
                        │   (nginx in Docker)     │
                        └────────────┬────────────┘
                                     │  /api/v1/*  (same origin: Vite proxy in
                                     │              dev, nginx proxy in Docker)
                                     ▼
                        ┌─────────────────────────┐
                        │        FastAPI          │
                        │  CORS · body limit ·    │
                        │  rate limit · error     │
                        │  envelope · timing      │
                        └────────────┬────────────┘
                                     │
        ┌────────────────────────────┼──────────────────────────────┐
        │                            │                              │
        ▼                            ▼                              ▼
┌───────────────┐         ┌────────────────────┐         ┌────────────────────┐
│ Redis         │         │  Detector pipeline │         │ MongoDB            │
│ (optional)    │◄────────┤  (in-process)      ├────────►│ essays             │
│ result cache  │  hit /  │                    │  write  │ analyses           │
│ rate limits   │  store  └─────────┬──────────┘         │ analysis_results   │
└───────────────┘                   │                     │ model_versions     │
                                    │                     │ evaluation_runs    │
                                    │                     └────────────────────┘
        ┌───────────────────────────┼───────────────────────────┐
        ▼               ▼           ▼            ▼              ▼
   spaCy parse   distilgpt2    burstiness   repetition   corpus reference
   (POS, deps)   token probs   (rhythm)     (n-grams)    (fitted centroids)
        │               │           │            │              │
        └───────────────┴─────┬─────┴────────────┴──────────────┘
                              ▼
                   style shift (within-essay)
                              ▼
              ┌───────────────────────────────┐
              │ Feature vectors               │
              │  sentence: 189 · document: 411│
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │ Our trained classifiers       │
              │  document: 3-class (verdict)  │
              │  sentence: binary (highlight) │
              └───────────────┬───────────────┘
                              ▼
                   Platt calibration → confidence band → abstention
                              ▼
                   Explanation engine (deterministic)
                              ▼
                         Final result


  Optional async path (KAFKA_ENABLED=true, essays > 25,000 chars)
  ─────────────────────────────────────────────────────────────
  FastAPI ──► essay.analysis.requests ──► analysis_worker ──► MongoDB
     │                                          │
     └──► 202 + poll_url                        └──► essay.analysis.results
                    ▲                                      │
                    └──── frontend polls GET /analysis/{id}┘
```

## Layering

Each layer depends only on the ones above it. `app/services/` has no imports from
`app/api/`, and `ml/` imports from `app/services/` — never the reverse. That
one-way dependency is what makes train/serve consistency structural rather than
aspirational: the training pipeline and the request handler run the *same*
feature-extraction code.

```
app/config.py            typed settings from environment
app/core/                logging · errors · input validation · rate limiting
app/db/                  mongodb · redis · kafka  (all degrade independently)
app/services/            the detection pipeline (see detection-methodology.md)
app/models/              MongoDB document builders (privacy-shaped)
app/schemas/             Pydantic request/response contract → OpenAPI
app/api/                 routes and dependencies
app/workers/             Kafka consumer (only used when enabled)

ml/generation/           dataset generation (Groq + offline bootstrap)
ml/training/             prepare_dataset → extract_features → train
ml/evaluation/           evaluate → find_failures
ml/artifacts/            trained models + metadata + reference stats
```

## Model lifecycle

Model loading dominates the cost of a short analysis: distilgpt2 takes ~10–20 s
and spaCy ~5 s on CPU, against ~0.3 s for the analysis itself. So both are
process-wide singletons loaded once during the FastAPI lifespan, followed by a
warm-up pass that compiles the lazy code paths.

- `LanguageModelService` — thread-safe double-checked lock, loads on first use.
- `NlpPipeline` — same pattern, records load time and any error.
- `DetectorModels` — loads the joblib artifacts and the reference stats.
- `POST /api/v1/model/reload` picks up a newly trained model without a restart,
  and invalidates the Redis cache so results are not served from a previous model
  version.

CPU-bound analysis runs in `asyncio.to_thread`, so the event loop keeps serving
health checks while an essay is being scored.

## Graceful degradation

The system is built so that infrastructure failures reduce capability rather than
causing outages. `GET /api/v1/health` reports per component, because a flat
200/500 would hide all of this:

| Component        | Missing → consequence                                       |
| ---------------- | ----------------------------------------------------------- |
| MongoDB          | analysis works, results cannot be retrieved later           |
| Redis            | no caching, rate limiting falls back to a per-process counter |
| Kafka            | everything runs synchronously (the default anyway)          |
| spaCy model      | regex segmentation, no POS/dependency features, warning set  |
| Corpus reference | `cor_*` features are zero, warning set                      |
| **Trained model**| **analysis unavailable — 503 with instructions**             |

Only the trained model is load-bearing. Everything else has a documented fallback,
and each fallback sets a warning that reaches the response and the UI.

## Why Redis is optional

Two justified uses, no others:

1. **Analysis result cache.** The pipeline is deterministic for a fixed
   `(essay, detector_version, model_version)` triple, so results are exactly
   cacheable under `SHA256` of that triple. Users re-analyse the same draft
   constantly while editing; with the cache that costs one round trip instead of
   1–2 s of CPU. Normalisation happens before hashing, so re-pasting with
   different line wrapping still hits.
2. **Distributed rate limiting**, which needs shared state to be correct across
   workers.

Redis is disabled by default and nothing degrades without it beyond losing the
cache.

## Why Kafka is optional and off by default

A 250-word essay analyses in ~0.3 s; a 1,200-word essay in ~1.7 s. Putting that
behind a broker would add a round trip, a polling loop in the frontend, and two
new failure modes in exchange for nothing.

Kafka earns its place only for work that genuinely does not fit a request: essays
above `ASYNC_THRESHOLD_CHARS` (25,000 chars ≈ 4,000 words, where analysis moves
into tens of seconds), batch dataset generation, and evaluation runs. When it is
disabled, `should_queue()` returns `False` unconditionally — even for a huge essay
with `async_mode: true` — because queueing to a broker that is not there would
hang the request forever.

The worker dispatches its model work through `asyncio.to_thread` so the consumer
keeps heartbeating; without that, a long analysis looks like a dead consumer and
the broker rebalances the partition mid-job. Offsets are committed even for
messages that fail, so a poison message cannot block a partition — the failure is
recorded in MongoDB instead.

## Data flow for one analysis

```
POST /api/v1/analysis {text}
  │
  ├─ rate limit check                      → 429 with retry hint
  ├─ body size middleware                  → 413 before the body is parsed
  ├─ validate + normalise                  → 422 empty/short, 413 long
  ├─ content hash (essay + versions)
  ├─ Redis cache lookup                    → return immediately on hit
  ├─ should_queue()?                       → 202 + poll_url (Kafka only)
  │
  ├─ asyncio.to_thread(detector.analyse)
  │     ├─ segment
  │     ├─ stylometry per sentence
  │     ├─ LM scoring (sliding window) → attribute tokens to sentences
  │     ├─ style shift  (needs stylometry + LM)
  │     ├─ burstiness + repetition + rhythm series
  │     ├─ corpus similarity (document and per sentence)
  │     ├─ document + sentence feature vectors
  │     ├─ classifier → calibrated probabilities
  │     ├─ verdict banding / abstention
  │     ├─ sentence banding + per-sentence evidence (flagged/uncertain only)
  │     ├─ paragraph rollup (length-weighted)
  │     └─ document evidence
  │
  ├─ persist (best-effort; a failure here never fails the analysis)
  ├─ cache store
  └─ 200 AnalysisResponse
```

## Privacy shaping

Two storage documents, both built to be safe by construction:

- `analyses` — verdict, summary, evidence, timings. No essay text.
- `analysis_results` — per-sentence rows as `(start, end, score, classification)`.
  Sentence *text* is included only when `SAVE_ESSAYS=true`.

Because offsets are always kept, a reloaded analysis can be re-rendered against
the copy the user still has in their editor — without the server ever having
stored it. The frontend does exactly that.

Logging has two structural guarantees: `safe_text_meta()` is the only sanctioned
way to describe user text in a log (it emits lengths and a truncated SHA-256
digest), and `scrub()` defensively removes emails, phone numbers and SSNs from any
string that might echo user input. A test greps the actual log files for a phrase
from the test essay.

## Frontend architecture

Hash-based routing across three views (Analyse, Research, How it works) — no
router dependency for three routes. Charts are hand-rolled SVG: a charting library
would cost more in bundle size and styling friction than it saves, and the axes
need to say what these particular measurements mean.

State is local to `AnalysePage`; there is no store, because there is one document
in flight at a time. The API client turns every failure mode into a typed
`ApiError` with copy written for a person, so no component has to invent error
text.

The frontend fetches `/essays/privacy` rather than hard-coding the storage notice.
A privacy claim that can drift out of step with the server's actual setting is the
one thing a privacy notice must never do.

## Reproducibility

Every trained model records `model_version`, `dataset_version`, `features_version`,
`lexicon_version`, training configuration, seed, CV protocol, platform, Python
version, and metrics. `GET /api/v1/model/info` exposes the active versions, and
the content hash used for caching includes them — so a model change cannot serve
stale results.
