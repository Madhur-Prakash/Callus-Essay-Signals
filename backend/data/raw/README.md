# `data/raw/` — where real essays go

These directories are **inputs you supply**, not generated output. They ship empty
on purpose: this repository contains no real admissions essays, because collecting
them without consent would be both a licensing and an ethics failure.

Adding real human essays here is the single highest-value improvement available to
this project — see [docs/evaluation.md](../../../docs/evaluation.md#what-would-improve-these-numbers-most).

```
data/raw/
├── human/           ← drop real human-written essays here (.txt, one per file)
│   └── manifest.json   optional metadata, see below
├── ai_generated/    ← externally sourced machine-written essays
└── ai_polished/     ← externally sourced machine-edited human essays
```

## human/

Read by `ingest_raw_human()` in
[`ml/training/prepare_dataset.py`](../../ml/training/prepare_dataset.py) on every
`prepare_dataset` run. One essay per `.txt` file, UTF-8. Files under 80 words are
skipped with a warning.

Add `manifest.json` alongside them to record provenance. Files without an entry are
still ingested, but their licence is recorded as `unspecified` so the gap shows up
in the dataset card rather than being silently assumed fine:

```json
{
  "essay-001.txt": {
    "topic": "community volunteering",
    "l2_english": false,
    "voice": "plain narrative",
    "license": "author permission on file, 2026-08-01",
    "notes": "Collected with written consent; author reviewed this use."
  }
}
```

`manifest.example.json` in `human/` is a working template.

Ingested essays get `source: "ingested_real"`, which also means the failure report
**withholds their text** — real essays are never reproduced in generated reports.

## ai_generated/ and ai_polished/

Reserved for machine text obtained outside this repo's generators (a different
provider, a shared benchmark, a colleague's collection). The offline and Groq
generators write to `data/ai_generated/` and `data/ai_polished/` instead, so these
stay empty unless you are importing something.

To wire an import in, follow the shape of `ingest_raw_human()` — build `Sample`
records with `source`, `group_id` and a licence, then let `prepare_dataset` handle
splitting and the leakage audit.

## After adding files

```powershell
cd backend
uv run python -m ml.training.prepare_dataset     # picks up raw/human automatically
uv run python -m ml.training.extract_features
uv run python -m ml.training.train
uv run python -m ml.evaluation.evaluate
```

Watch for `data_regime` in `data/processed/manifest.json` moving from `bootstrap`
to `mixed` or `real` — that is the signal the metrics have started to mean
something.

## Do not commit real essays

`.gitignore` excludes `data/raw/human/*.txt` and everything under
`data/raw/ai_generated/` and `data/raw/ai_polished/`. The `.gitkeep` files exist so
the directories survive a clone while their contents never leave your machine.

Only use text you have the right to use. Get consent.
