# Documentation

| Document | Read it for |
| -------- | ----------- |
| [architecture.md](architecture.md) | System diagram, layering, model lifecycle, graceful degradation, data flow for one analysis, why Redis and Kafka are optional |
| [detection-methodology.md](detection-methodology.md) | Every feature layer in detail, why the language model is an instrument rather than a judge, calibration and abstention policy, what the system cannot do |
| [dataset.md](dataset.md) | Corpus design decisions and what each one costs — three classes, generator diversity, length matching, leakage control, the L2 subset |
| [evaluation.md](evaluation.md) | **Measured results** — overall and per-class metrics, confusion matrix, ablation, generalisation, bias analysis, confidently wrong examples |
| [api.md](api.md) | Endpoints, request/response schemas, error codes, worked examples |
| [privacy.md](privacy.md) | Exactly what is stored and logged, the structural guarantees behind that, and what is deliberately not implemented |
| [limitations.md](limitations.md) | **Ranked limitations and known failure cases.** Read this before acting on any output |

Operational dataset instructions (layout, regenerating, adding real essays) are in
[`../backend/data/README.md`](../backend/data/README.md).

## If you only read two things

1. [limitations.md](limitations.md) — the detector over-flags human writing, and
   the training corpus is synthetic by default.
2. [detection-methodology.md](detection-methodology.md) — how the classification is
   actually produced, and why it is not an LLM wrapper.
