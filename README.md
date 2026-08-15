# Essay Signals — explainable AI-writing detection for admissions essays

A working detector for AI-generated and AI-polished writing in college admissions
essays. React + Vite frontend, FastAPI backend, and a hybrid ML/NLP pipeline that
produces **measured evidence for every passage it flags**.

**It is not an LLM wrapper.** A small local causal language model (`distilgpt2`) is
used as a *measuring instrument* — it reports how probable each token was given the
tokens before it. Those numbers, plus stylometric, syntactic, burstiness,
repetition, within-essay style-shift and reference-corpus features (411 document
features in all), go into a classifier **we train ourselves**. No hosted chat model
is called during analysis, and there is no code path from a request handler to one.

```
Essay → tokeniser → local LM → token log-probs / entropy / rank
                                        ↓
     stylometry · syntax · burstiness · repetition · style shift · corpus similarity
                                        ↓
                         411-feature document vector
                                        ↓
                    our trained, calibrated classifier
                                        ↓
              confidence band · abstention · deterministic evidence
```

---

## ⚠️ Read this before quoting any number

Out of the box the detector is trained on a **synthetic bootstrap corpus**: the
human class is 36 hand-authored seed essays, and the machine classes come from an
offline template generator and a rule-based editor. The reported metrics measure
*how separable those three generators are* — **not** how well the detector
identifies real AI writing.

This is surfaced everywhere: `data_regime: "bootstrap"` in the API response and
model metadata, a `REGIME WARNING` at the top of the evaluation report, and a
banner in the UI. To get meaningful numbers, supply a `GROQ_API_KEY` to generate
real machine text and add real essays to `backend/data/raw/human/`.

**Also honest:** on the current corpus the detector **over-flags human writing** —
human recall sits well below the machine classes, and most errors are human essays
called machine-polished. Overall accuracy hides this, which is why the evaluation
report leads with it. See [docs/limitations.md](docs/limitations.md).

---

## Quick start (Windows PowerShell)

Prerequisites: **Python 3.11 or 3.12**, **Node 20+**, **[uv](https://docs.astral.sh/uv/)**,
and MongoDB (local install or Docker). MongoDB is optional — without it analysis
works but results are not saved.

```powershell
# ---------- 1. backend ----------
cd backend
uv venv
uv sync                                             # includes the dev group
Copy-Item .env.example .env

# ---------- 2. build the corpus and train (≈20 min, one time) ----------
uv run python -m ml.generation.bootstrap_corpus     # offline synthetic corpus
uv run python -m ml.training.prepare_dataset        # split by group, audit leakage
uv run python -m ml.training.extract_features       # ≈6 min  (411 doc features)
uv run python -m ml.training.train                  # ≈10 min (ablation + calibration)
uv run python -m ml.evaluation.evaluate             # held-out metrics
uv run python -m ml.evaluation.find_failures        # confidently-wrong cases

# ---------- 3. run the API ----------
uv run uvicorn app.main:app --reload --port 8000
```

```powershell
# ---------- 4. frontend (new terminal) ----------
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**, click *Load an example*, and press *Analyse essay*.

> The first backend start downloads `distilgpt2` (~350 MB) and loads spaCy, so
> startup takes 20–30 seconds. Subsequent starts use the cache.

### Docker

```powershell
docker compose up --build                    # mongodb + backend + frontend
docker compose --profile cache up --build    # ... plus redis
docker compose --profile queue up --build    # ... plus kafka + async worker
```

Frontend on **http://localhost:5173**, API on **http://localhost:8000**. Model
artifacts are bind-mounted from `backend/ml/artifacts`, so run the training
pipeline once on the host first (or exec into the container and run it there).

---

## Measured results (bootstrap corpus, held-out test split, n = 128)

| Metric | Value |
| ------ | ----: |
| Accuracy | 0.891 |
| Balanced accuracy | 0.851 |
| Macro F1 | 0.852 |
| ROC-AUC (OvR macro) | 0.976 |
| Expected calibration error | 0.086 |
| **Human recall** | **0.742** |

| Class | P | R | F1 | n |
| ----- | --: | --: | --: | -: |
| human | 0.793 | 0.742 | 0.767 | 31 |
| ai_generated | 1.000 | 1.000 | 1.000 | 65 |
| ai_polished | 0.765 | 0.813 | 0.788 | 32 |

Feature-set ablation, identical estimator, only the features vary:

| Feature set | Features | Macro F1 |
| ----------- | -------: | -------: |
| **hybrid** | 411 | **0.852** |
| hybrid_no_shift | 382 | 0.841 |
| baseline_stylometric | 246 | 0.830 |
| lm_only | 118 | 0.767 |

So the language-model features do add value over stylometry alone (+0.022) and are
insufficient on their own (−0.085). The `ai_generated` row being perfect is a **red
flag, not a triumph** — the offline template generator is trivially separable. Full
analysis, generalisation slices, bias study and the three confidently wrong cases:
[docs/evaluation.md](docs/evaluation.md).

---

## What you get

### Theme and motion

- **Light / dark / system** theme toggle. "System" is a genuine third state — it
  keeps following the OS rather than freezing whatever it said at first paint.
  The choice persists to `localStorage` and sets `data-theme` on `<html>`.
- **Framer Motion** for React state transitions (page changes, results stagger,
  the sliding nav indicator), **GSAP** for imperative timeline work (SVG path
  drawing, number count-ups, scroll-triggered reveals), **Lenis** for page scroll.
  Each library has one job; none of them overlap.
- All three obey `prefers-reduced-motion` — Framer via `MotionConfig
  reducedMotion="user"`, GSAP and Lenis via an explicit gate in
  [useMotion.ts](frontend/src/hooks/useMotion.ts) — and all three no-op under test
  so the suite stays deterministic. Lenis is kept deliberately gentle and is
  disabled outright under reduced motion, because heavy scroll smoothing makes
  dense evidence tables harder to read, not nicer.

### The analysis view

- Large editor with live word / character / paragraph / sentence counts, three
  example essays, and a clear statement of whether your essay is stored.
- **Honest input gating**: hard limits (rejected by the server) are separated from
  the soft floors below which the detector abstains. Both are read from
  `GET /model/info`, so the UI cannot advertise a limit the server does not
  enforce — paste 200 characters of one long sentence and it tells you up front
  that the result will be "insufficient evidence".
- **Overall assessment** — a named class, a confidence *band* (not a fake
  two-decimal percentage), calibrated per-class probabilities, and an explicit
  "insufficient evidence" verdict when the measurements do not support a call.
- **The essay, marked up** — sentence-level marks styled as an editor's underline
  rather than a heat map. Hover or click any sentence for its evidence.
- **Evidence panel** — discrete ten-block meters with the measured value, its
  percentile within the human training distribution, within-essay comparisons
  ("perplexity 14.2 against an essay median of 31.8"), and the classifier's own
  signed per-feature contributions.
- **Sentence rhythm chart** — sentence lengths against the essay mean, each bar
  coloured by that sentence's own score, so shape and verdict can be read together.
- **Paragraph breakdown**, **repetition findings**, and a full table of every
  measured statistic.

### The research view

Read from the evaluation artifacts — nothing is computed in the browser:

Overall and per-class metrics · confusion matrix · ROC and precision-recall curves ·
calibration reliability diagram with ECE · four-way feature-set comparison ·
permutation feature importance · topic / generator / length generalisation slices ·
**bias analysis with Wilson intervals** · **three confidently wrong cases** with
measured explanations and proposed fixes · the full dataset card.

---

## Project layout

```
backend/
├── app/
│   ├── main.py              FastAPI app, lifespan, middleware
│   ├── config.py            typed settings from environment
│   ├── core/                logging (logifyx) · errors · validation · rate limit
│   ├── db/                  mongodb · redis · kafka  (each degrades alone)
│   ├── services/            THE PIPELINE
│   │   ├── nlp.py                 segmentation (spaCy, regex fallback)
│   │   ├── stylometry.py          surface · lexical · syntactic
│   │   ├── probability_analyzer.py the LM instrument (sliding window)
│   │   ├── burstiness.py          sentence rhythm
│   │   ├── repetition.py          n-gram · syntactic template · discourse
│   │   ├── style_shift.py         within-essay deviation + change points
│   │   ├── corpus_analyzer.py     topic-free reference-corpus similarity
│   │   ├── feature_extractor.py   single source of truth for the feature space
│   │   ├── classifier.py          our trained models — makes the decision
│   │   ├── calibration.py         Platt scaling · bands · abstention
│   │   ├── explanation_engine.py  deterministic evidence, no LLM
│   │   └── detector.py            orchestration
│   ├── models/ schemas/ api/ workers/
├── ml/
│   ├── generation/          Groq generators + offline bootstrap corpus
│   ├── training/            prepare_dataset → extract_features → train
│   ├── evaluation/          evaluate → find_failures (+ reports)
│   └── artifacts/           trained models, metadata, reference stats
├── data/                    seeds · raw · processed  (see data/README.md)
├── scripts/smoke_e2e.py     end-to-end check against a running API
├── tests/                   163 tests
└── pyproject.toml / uv.lock

frontend/
├── src/
│   ├── api/client.ts        typed client; every failure mode → readable message
│   ├── components/          editor · verdict · essay viewer · evidence · charts
│   │   └── ui/              reusable kit: Surface · Card/Section · Button · Badge
│   │                        Gauge · Meter · Stat · Sparkline · Progress · Tabs
│   │                        Reveal/SplitHeading · Atmosphere  (one barrel export)
│   ├── pages/               Analyse · Results · Research · How it works · Limitations
│   ├── hooks/               useTheme (light/dark/system) · useMotion (gsap/lenis)
│   ├── lib/ types/
│   └── styles/              index (entry) → theme · tokens · components · app
├── tests/                   45 tests
└── package.json

docs/                        architecture · methodology · dataset · evaluation
                             api · privacy · limitations
docker-compose.yml
```

---

## Commands

### ML pipeline

```powershell
cd backend

# dataset
uv run python -m ml.generation.bootstrap_corpus              # offline, no API key
$env:GROQ_API_KEY = "gsk_..."                                # optional, recommended
uv run python -m ml.generation.generate_ai_essays --per-model 30 --replace
uv run python -m ml.generation.polish_essays --replace
uv run python -m ml.training.prepare_dataset

# features and training
uv run python -m ml.training.extract_features
uv run python -m ml.training.extract_features --limit 40     # quick smoke run
uv run python -m ml.training.train
uv run python -m ml.training.train --calibration isotonic --seed 7

# evaluation
uv run python -m ml.evaluation.evaluate
uv run python -m ml.evaluation.evaluate --store-in-mongo
uv run python -m ml.evaluation.find_failures --top 5
```

### Running

```powershell
cd backend
uv run uvicorn app.main:app --reload --port 8000       # API
uv run python -m app.workers.analysis_worker           # Kafka worker (if enabled)

cd frontend
npm run dev                                            # dev server
npm run build                                          # typecheck + production build
npm run preview                                        # serve the build
```

### Tests

```powershell
cd backend
uv run pytest                          # all 163
uv run pytest -m "not slow"            # skip model-loading tests
uv run pytest -m integration           # MongoDB tests (skip cleanly if absent)
uv run ruff check .

cd frontend
npm test                               # 36 tests
npm run typecheck
```

With the API running, an end-to-end check across the real HTTP surface — health,
model info, privacy, three contrasting essays, persistence round-trip, every error
path, the evaluation report and the OpenAPI schema:

```powershell
cd backend
uv run python -m scripts.smoke_e2e     # 52 checks
```

---

## Configuration

Full annotated list in [`backend/.env.example`](backend/.env.example). The ones
that matter:

| Variable          | Default          | Notes                                                     |
| ----------------- | ---------------- | --------------------------------------------------------- |
| `SAVE_ESSAYS`     | `false`          | When false the essay text is **never** stored. A request can opt out, never in. |
| `MONGODB_URL`     | `localhost:27017`| Optional; without it analysis works but is not persisted.  |
| `REDIS_ENABLED`   | `false`          | Caches deterministic results; nothing breaks without it.   |
| `KAFKA_ENABLED`   | `false`          | Off deliberately — see below.                              |
| `LM_MODEL_NAME`   | `distilgpt2`     | The instrument. `gpt2-medium` measures better at ~3× cost. |
| `CORS_ORIGINS`    | `localhost:5173` | Comma-separated exact origins.                             |
| `MIN/MAX_ESSAY_CHARS` | `200` / `60000` | Input bounds.                                          |

### Why Redis is optional

Two justified uses and no others: caching deterministic analysis results under
`SHA256(essay + detector_version + model_version)` — users re-analyse the same
draft constantly while editing — and distributed rate limiting, which needs shared
state to be correct. Disabled by default; nothing degrades without it beyond losing
the cache.

### Why Kafka is off by default

A 250-word essay analyses in ~0.3 s and a 1,200-word essay in ~1.7 s. Putting that
behind a broker would add a round trip, a polling loop in the frontend and two new
failure modes for no benefit. Kafka is wired up and tested for work that genuinely
does not fit a request — essays above 25,000 characters, batch generation,
evaluation runs — and when disabled the code refuses to queue even if asked, because
queueing to a broker that is not there would hang the request forever.

---

## Performance

Measured on CPU (no GPU), 250-word and 1,200-word essays:

| Stage                    | 250 words | 1,200 words |
| ------------------------ | --------- | ----------- |
| Segmentation + parse     | ~17 ms    | ~115 ms     |
| Stylometry               | ~5 ms     | ~67 ms      |
| LM scoring               | ~113 ms   | ~1,385 ms   |
| Style shift              | ~2 ms     | ~40 ms      |
| Rhythm + repetition      | ~1 ms     | ~20 ms      |
| **Total**                | **~150 ms** | **~1.7 s** |

One-time startup: spaCy ~5 s, distilgpt2 ~10–20 s. Both load once as process
singletons; per-stage timings are returned in the response when
`DEBUG_TIMINGS=true`.

---

## Documentation

| Document | Contents |
| -------- | -------- |
| [docs/architecture.md](docs/architecture.md) | System diagram, layering, model lifecycle, degradation, data flow |
| [docs/detection-methodology.md](docs/detection-methodology.md) | Every feature layer, why the LM is an instrument, calibration and abstention |
| [docs/dataset.md](docs/dataset.md) | Corpus design decisions, leakage control, what each choice costs |
| [docs/evaluation.md](docs/evaluation.md) | Measured results, generalisation, bias, failure analysis |
| [docs/api.md](docs/api.md) | Endpoints, schemas, error codes, examples |
| [docs/privacy.md](docs/privacy.md) | What is stored and logged, and the structural guarantees |
| [docs/limitations.md](docs/limitations.md) | **Ranked limitations and known failure cases** |
| [backend/data/README.md](backend/data/README.md) | Dataset layout and how to add real essays |

---

## Scientific honesty

This system will never tell you an essay "was written by AI". The strongest honest
claim available from measuring text is:

> "This passage has statistical properties in common with the machine-written
> examples in our evaluation data."

Specifically:

- **Low perplexity is not evidence of machine authorship.** Clear, conventional
  human prose scores low too.
- **Uniform sentence length is not evidence either.** Plenty of people write
  evenly, and a model can be asked to vary.
- **A style shift means the register changed**, not that someone else wrote it.
  Quoting, or moving from narrative to reflection, does the same thing.
- **AI detectors are known to over-flag writers who learned English as an
  additional language.** This one has not been shown safe on that axis and should
  be assumed to carry the same risk.
- **The system abstains** when the top class is under 45%, the top two are within
  10 points, or the document is under 5 sentences / 120 words.

Detection ≠ proof of authorship. Do not use this as evidence in an
academic-integrity process, and do not reject an applicant based on it.

## License

MIT for the code. Generated model output is subject to the terms of whichever
provider produced it. No admissions essays were scraped.
