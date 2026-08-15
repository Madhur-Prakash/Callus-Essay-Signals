"""Dataset generation.

Two independent paths, both producing the same on-disk record schema:

``groq_client`` + ``generate_ai_essays`` + ``polish_essays``
    The **preferred** path. Calls real instruction-tuned models through the Groq
    API across several model families, prompt strategies and temperatures.
    Requires ``GROQ_API_KEY``.

``bootstrap_corpus``
    The **offline fallback**. Produces a synthetic corpus with no network access
    so the whole pipeline is runnable and reproducible out of the box. Its
    limitations are severe and are documented in ``data/README.md`` and
    ``docs/dataset.md``: a classifier trained only on bootstrap data is learning
    to separate two *generators*, not humans from machines.
"""
