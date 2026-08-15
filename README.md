# Essay Signals - explainable AI-writing detection for admissions essays

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-en__core__web__sm-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white)](https://vite.dev/)
[![Tailwind](https://img.shields.io/badge/Tailwind-4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-optional-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Tests](https://img.shields.io/badge/tests-163%20backend%20%7C%2045%20frontend-brightgreen)](#tests)

<p align="center">
  <b>Home</b>
  &middot;
  <a href="docs/architecture.md">Architecture</a>
  &middot;
  <a href="docs/detection-methodology.md">Methodology</a>
  &middot;
  <a href="docs/dataset.md">Dataset</a>
  &middot;
  <a href="docs/evaluation.md">Evaluation</a>
  &middot;
  <a href="docs/api.md">API</a>
  &middot;
  <a href="docs/privacy.md">Privacy</a>
  &middot;
  <a href="docs/limitations.md">Limitations</a>
  &middot;
  <a href="docs/README.md">All docs</a>
</p>

A working detector for AI-generated and AI-polished writing in college admissions
essays. React + Vite frontend, FastAPI backend, and a hybrid ML/NLP pipeline that
produces **measured evidence for every passage it flags**.

**It is not an LLM wrapper.** A small local causal language model (`distilgpt2`) is
used as a *measuring instrument* - it reports how probable each token was given the
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

https://github.com/user-attachments/assets/9b4d353e-a9bd-4cad-a234-8aac2734e38a

---

## How this meets the brief

| The brief asks for | Where it is |
| --- | --- |
| A working application with a real interface, not a script or a notebook | React + Vite SPA over a FastAPI service. [Quick start](#quick-start-windows-powershell) · [Architecture](docs/architecture.md) |
| Show **where** and **why**, not "73% AI" | Sentence marks in the reader, plus an evidence panel per sentence: measured value, percentile against the human corpus, within-essay comparison, and the classifier's signed per-feature contributions. [How evidence is generated](docs/detection-methodology.md#explanation) |
| Not a wrapper; a chat model must not make the judgement | `distilgpt2` runs locally and returns token log-probabilities only. The decision is made by our own trained LightGBM/logistic classifier in [`classifier.py`](backend/app/services/classifier.py). No hosted model is reachable from a request handler. [Why this is not a wrapper](docs/detection-methodology.md#the-core-commitment) |
| Using an LM as an instrument is fine | That is exactly the split: [`probability_analyzer.py`](backend/app/services/probability_analyzer.py) measures, [`feature_extractor.py`](backend/app/services/feature_extractor.py) derives, [`classifier.py`](backend/app/services/classifier.py) judges |
| Detection at the level of sentences and passages | Per-sentence and per-paragraph scores, plus a document verdict. The realistic case - a human paragraph a model later polished - is the `ai_polished` class |
| Every flag backed by visible evidence | [`explanation_engine.py`](backend/app/services/explanation_engine.py) generates evidence deterministically from measured values. No language model writes the explanations |
| Build the dataset; document source, size and coverage gaps | [Dataset methodology](docs/dataset.md) · [operational layout](backend/data/README.md) · the live dataset card is served at `GET /api/v1/evaluation` and rendered under Research → Dataset |
| Honest accuracy on your own test set | [Measured results](#measured-results-bootstrap-corpus-held-out-test-split-n--128) - reported with the regime warning attached, not as a headline claim |
| **Three essays it gets confidently wrong**, with your theory why | [Confidently wrong examples](docs/evaluation.md#confidently-wrong-examples) - three cases with the measurements that misled the model and a proposed fix for each. Also in the app under Research → Failures |
| Flagging of second-language English writers, if present | An explicit bias study with Wilson intervals over an L2 subset. [Bias analysis](docs/evaluation.md#bias-analysis) - the honest result is *not measurable at this sample size*, which is reported as a gap rather than as fairness |

---

## Read this before quoting any number

> [!CAUTION]
> Out of the box the detector is trained on a **synthetic bootstrap corpus**: the
> human class is 36 hand-authored seed essays, and the machine classes come from an
> offline template generator and a rule-based editor. The reported metrics measure
> *how separable those three generators are* - **not** how well the detector
> identifies real AI writing.

This is surfaced everywhere: `data_regime: "bootstrap"` in the API response and
model metadata, a `REGIME WARNING` at the top of the evaluation report, and a
banner in the UI. To get meaningful numbers, supply a `GROQ_API_KEY` to generate
real machine text and add real essays to `backend/data/raw/human/` - see
[backend/data/README.md](backend/data/README.md).

> [!IMPORTANT]
> **Also honest:** on the current corpus the detector **over-flags human writing** -
> human recall is 0.742 against 1.000 for `ai_generated`, and most errors are human
> essays called machine-polished. Overall accuracy hides this, which is why the
> evaluation report leads with it. See [docs/limitations.md](docs/limitations.md).

---

## Quick start (Windows PowerShell)

Prerequisites: **Python 3.11 or 3.12**, **Node 20+**, **[uv](https://docs.astral.sh/uv/)**,
and MongoDB (local install or Docker). MongoDB is optional - without it analysis
works but results are not saved.

```powershell
# ---------- 1. backend ----------
cd backend
uv venv
uv sync                                             # includes the dev group
cp .env.example .env

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
flag, not a triumph** - the offline template generator is trivially separable. Full
analysis, generalisation slices, bias study and the three confidently wrong cases:
[docs/evaluation.md](docs/evaluation.md).

---

## What you get

### Theme and motion

- **Light / dark / system** theme toggle. "System" is a genuine third state - it
  keeps following the OS rather than freezing whatever it said at first paint.
  The choice persists to `localStorage` and sets `data-theme` on `<html>`.
- **Framer Motion** for React state transitions (page changes, results stagger,
  the sliding nav indicator), **GSAP** for imperative timeline work (SVG path
  drawing, number count-ups, scroll-triggered reveals), **Lenis** for page scroll.
  Each library has one job; none of them overlap.
- All three obey `prefers-reduced-motion` - Framer via `MotionConfig
  reducedMotion="user"`, GSAP and Lenis via an explicit gate in
  [useMotion.ts](frontend/src/hooks/useMotion.ts) - and all three no-op under test
  so the suite stays deterministic. Lenis is kept deliberately gentle and is
  disabled outright under reduced motion, because heavy scroll smoothing makes
  dense evidence tables harder to read, not nicer.

### The analysis view

- Large editor with live word / character / paragraph / sentence counts, three
  example essays, and a clear statement of whether your essay is stored.
- **Honest input gating**: hard limits (rejected by the server) are separated from
  the soft floors below which the detector abstains. Both are read from
  `GET /model/info`, so the UI cannot advertise a limit the server does not
  enforce - paste 200 characters of one long sentence and it tells you up front
  that the result will be "insufficient evidence".
- **Overall assessment** - a named class, a confidence *band* (not a fake
  two-decimal percentage), calibrated per-class probabilities, and an explicit
  "insufficient evidence" verdict when the measurements do not support a call.
- **The essay, marked up** - sentence-level marks styled as an editor's underline
  rather than a heat map. Hover or click any sentence for its evidence.
- **Evidence panel** - discrete ten-block meters with the measured value, its
  percentile within the human training distribution, within-essay comparisons
  ("perplexity 14.2 against an essay median of 31.8"), and the classifier's own
  signed per-feature contributions.
- **Sentence rhythm chart** - sentence lengths against the essay mean, each bar
  coloured by that sentence's own score, so shape and verdict can be read together.
- **Paragraph breakdown**, **repetition findings**, and a full table of every
  measured statistic.

### The research view

Read from the evaluation artifacts - nothing is computed in the browser:

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
│   │   ├── classifier.py          our trained models - makes the decision
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
npm test                               # 45 tests
npm run typecheck
npm run lint
```

With the API running, an end-to-end check across the real HTTP surface - health,
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
| `KAFKA_ENABLED`   | `false`          | Off deliberately - see below.                              |
| `LM_MODEL_NAME`   | `distilgpt2`     | The instrument. `gpt2-medium` measures better at ~3× cost. |
| `CORS_ORIGINS`    | `localhost:5173` | Comma-separated exact origins.                             |
| `MIN/MAX_ESSAY_CHARS` | `200` / `60000` | Input bounds.                                          |

### Why Redis is optional

Two justified uses and no others: caching deterministic analysis results under
`SHA256(essay + detector_version + model_version)` - users re-analyse the same
draft constantly while editing - and distributed rate limiting, which needs shared
state to be correct. Disabled by default; nothing degrades without it beyond losing
the cache.

### Why Kafka is off by default

A 250-word essay analyses in ~0.3 s and a 1,200-word essay in ~1.7 s. Putting that
behind a broker would add a round trip, a polling loop in the frontend and two new
failure modes for no benefit. Kafka is wired up and tested for work that genuinely
does not fit a request - essays above 25,000 characters, batch generation,
evaluation runs - and when disabled the code refuses to queue even if asked, because
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

Start at the [documentation index](docs/README.md), or go straight to a page.
Every document carries a navigation bar linking to all the others.

| Document | Contents |
| -------- | -------- |
| [docs/README.md](docs/README.md) | Index, and what to read first if you only read two things |
| [docs/architecture.md](docs/architecture.md) | System diagram, layering, model lifecycle, degradation, data flow, frontend structure |
| [docs/detection-methodology.md](docs/detection-methodology.md) | Every feature layer, why the LM is an instrument, calibration and abstention |
| [docs/dataset.md](docs/dataset.md) | Corpus design decisions, leakage control, what each choice costs |
| [docs/evaluation.md](docs/evaluation.md) | Measured results, generalisation, bias, failure analysis |
| [docs/api.md](docs/api.md) | Endpoints, schemas, error codes, examples |
| [docs/privacy.md](docs/privacy.md) | What is stored and logged, and the structural guarantees |
| [docs/limitations.md](docs/limitations.md) | **Ranked limitations and known failure cases** |
| [backend/README.md](backend/README.md) | Backend-only quick start and module map |
| [backend/data/README.md](backend/data/README.md) | Dataset layout and how to add real essays |
| [backend/data/raw/README.md](backend/data/raw/README.md) | Sourcing rules for real human essays |

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

> [!WARNING]
> Detection is not proof of authorship. Do not use this as evidence in an
> academic-integrity process, and do not reject an applicant based on it.

Full ranked list, with severity and the measurement behind each one:
[docs/limitations.md](docs/limitations.md).

---

<div align="center">

Made with 💖 for Callus by **Madhur Prakash**

</div>
