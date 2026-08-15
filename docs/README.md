<p align="center">
  <a href="../README.md">Home</a>
  &middot;
  <a href="architecture.md">Architecture</a>
  &middot;
  <a href="detection-methodology.md">Methodology</a>
  &middot;
  <a href="dataset.md">Dataset</a>
  &middot;
  <a href="evaluation.md">Evaluation</a>
  &middot;
  <a href="api.md">API</a>
  &middot;
  <a href="privacy.md">Privacy</a>
  &middot;
  <a href="limitations.md">Limitations</a>
  &middot;
  <b>All docs</b>
</p>

---

# Documentation

Every page below carries this same navigation bar, so you can move between them
without coming back here.

| Document | Read it for |
| -------- | ----------- |
| [architecture.md](architecture.md) | System diagram, layering, model lifecycle, graceful degradation, data flow for one analysis, why Redis and Kafka are optional, the frontend's component kit and motion rules |
| [detection-methodology.md](detection-methodology.md) | Every feature layer in detail, why the language model is an instrument rather than a judge, calibration and abstention policy, what the system cannot do |
| [dataset.md](dataset.md) | Corpus design decisions and what each one costs - three classes, generator diversity, length matching, leakage control, the L2 subset |
| [evaluation.md](evaluation.md) | **Measured results** - overall and per-class metrics, confusion matrix, ablation, generalisation, bias analysis, the three confidently wrong examples |
| [api.md](api.md) | Endpoints, request/response schemas, error codes, worked examples |
| [privacy.md](privacy.md) | Exactly what is stored and logged, the structural guarantees behind that, and what is deliberately not implemented |
| [limitations.md](limitations.md) | **Ranked limitations and known failure cases.** Read this before acting on any output |

Operational instructions live with the code rather than here:

| Document | Read it for |
| -------- | ----------- |
| [`../backend/README.md`](../backend/README.md) | Backend-only quick start and module map |
| [`../backend/data/README.md`](../backend/data/README.md) | Dataset layout, regenerating the corpus, adding real essays |
| [`../backend/data/raw/README.md`](../backend/data/raw/README.md) | Sourcing rules for real human essays, and what may not be collected |

## If you only read two things

1. [limitations.md](limitations.md) - the detector over-flags human writing, and
   the training corpus is synthetic by default.
2. [detection-methodology.md](detection-methodology.md) - how the classification is
   actually produced, and why it is not an LLM wrapper.

## Reading order

For a first pass at the whole system:

1. [architecture.md](architecture.md) - what the pieces are
2. [detection-methodology.md](detection-methodology.md) - what they measure
3. [dataset.md](dataset.md) - what they learned from
4. [evaluation.md](evaluation.md) - how well it works
5. [limitations.md](limitations.md) - where it does not

[api.md](api.md) and [privacy.md](privacy.md) are references; reach for them when
you need them rather than reading them through.
