# Dataset

This directory holds the corpus the detector is trained and evaluated on.

> **The single most important thing on this page.** Out of the box this corpus is
> **synthetic**. The human class is hand-authored seed essays written for this
> repository; the machine classes come from an offline template generator and a
> rule-based editor. Metrics computed on it measure *how separable those three
> generators are* — not how well the detector identifies real AI writing. Every
> report labels this the `bootstrap` regime, the API reports
> `data_regime: "bootstrap"`, and the UI shows a warning banner. Replace the
> machine classes with real model output (Groq) and the human class with real
> essays before quoting any number as performance.

## Layout

```
data/
├── seeds/                     hand-authored human seed essays (source of truth)
│   ├── human_seeds_part1.json 18 seeds
│   └── human_seeds_part2.json 18 seeds
├── raw/
│   ├── human/                 DROP REAL ESSAYS HERE (see "Adding real essays")
│   ├── ai_generated/          reserved for externally sourced machine text
│   └── ai_polished/           reserved for externally sourced edited text
├── human/human.jsonl          generated: human class
├── ai_generated/…jsonl        generated: fully machine-written class
├── ai_polished/…jsonl         generated: human text edited by machine
├── processed/
│   ├── corpus.jsonl           all three classes, with split assignments
│   ├── splits.json            group → split mapping and the split report
│   ├── manifest.json          the dataset card (counts, provenance, limitations)
│   ├── polish_pairs.jsonl     original ↔ polished side-by-side, auditable
│   ├── features.npz           extracted feature matrices
│   └── features_manifest.json per-document metadata for the matrices
└── README.md
```

`processed/`, `human/`, `ai_generated/` and `ai_polished/` are generated and are
**not** committed. Regenerate them with the commands below.

## The three classes

| Class          | What it is                                            | How it is produced offline                    |
| -------------- | ----------------------------------------------------- | --------------------------------------------- |
| `human`        | Student-style personal essays                          | 36 hand-authored seeds + human editing noise   |
| `ai_generated` | Fully machine-written essays                           | Procedural template generator, 5 personas      |
| `ai_polished`  | **Human text that a machine edited** — the real threat | Rule-based editor over the real seed text      |

`ai_polished` is the class that matters most, because it is the realistic case:

```
human writes essay  →  AI edits it  →  final essay
```

Six transforms are applied, ranging from a light touch to a heavy rewrite:

| Transform           | What it changes                                                     |
| ------------------- | ------------------------------------------------------------------- |
| `clarity`           | expands contractions, removes hedges                                |
| `vocabulary`        | upgrades word choice                                                |
| `restructure`       | merges short sentences, inserts connectives                         |
| `formalize`         | all of the above plus dropped sentence-initial conjunctions         |
| `shorten`           | removes hedges and a trailing sentence                              |
| `partial_paragraph` | **rewrites exactly one paragraph, leaving the rest byte-identical** |

`partial_paragraph` is the localised-edit case the within-essay style-shift
analysis exists to catch. `polish_pairs.jsonl` records every original/edited pair
with its word delta and overlap, so any transform can be inspected directly.

`grammar_only` is defined in `ml/generation/prompts.py` but **excluded offline**:
a rule-based spacing fixer applied to already-clean seed text returns the input
unchanged, which would put identical text under two different labels. It is
generated on the Groq path, where the model also rephrases slightly.

## Provenance recorded per sample

Every record carries the full block below, because the evaluation slices by all
of it. A sample without provenance is a sample we cannot say anything honest
about.

`record_id`, `label`, `group_id`, `source`, `topic`, `length_band`, `voice`,
`l2_english`, `model`, `strategy`, `temperature`, `top_p`, `parent_id`,
`license`, `created_at`, `notes`, `split`

`source` is always one of: `seed_authored`, `seed_variant`, `ingested_real`,
`groq`, `bootstrap_procedural`, `bootstrap_rule_polish`.

## Leakage controls

Enforced in code (`ml/training/prepare_dataset.py`), not by convention:

1. **Splits are made over groups, never samples.** `group_id` ties together every
   document derived from the same underlying essay — the original, its editing
   variants, and **every AI-polished version of it**. A polished essay therefore
   shares a group with the human original it came from and can never land on the
   opposite side of a split boundary.
2. **Held-out topics are test-only.** Six topics appear in no training document.
3. **A held-out generator is test-only.** One persona (`proxy-heldout-qwen`)
   produces no training document, which is what makes the cross-model number
   meaningful.
4. **Exact duplicates are removed** (case- and whitespace-insensitive).
5. **Near-duplicates are audited and dropped.** Any validation/test document
   whose 5-gram containment against a training document exceeds 0.55 is removed
   by default. On the bootstrap corpus this catches ~20 template-generated essays
   that share most of their n-grams; leaving them in would inflate AI-class recall
   for reasons unrelated to detection. Pass `--keep-near-duplicates` to retain
   them.
6. **Splits are stratified by label signature and by the L2-English flag**, so
   the bias analysis has L2 documents in the test split. Stratifying on labels
   alone put every L2 seed in train/validation and made the bias measurement
   impossible.

Target proportions are 70/15/15 by group. The realised split is test-heavy
because rules 2 and 3 force whole groups into test; the exact counts are in
`processed/manifest.json`.

## Preprocessing

Deliberately minimal — the detector measures *style*, so "fixing" the author's
writing would destroy the signal:

- Unicode NFKC normalisation, CRLF → LF
- zero-width, bidi and control characters removed
- runs of 3+ newlines collapsed to one paragraph break
- exact duplicate documents removed

No lowercasing, no stopword removal, no stemming, no punctuation repair.

## Licensing and ethics

- **No admissions essays were scraped.** Doing so without consent would be both
  a licensing and an ethics failure.
- Seed essays: written for this repository, MIT-licensed with the code
  (`license: synthetic-mit`).
- Groq output: `license: model-output-see-provider-terms` — check your provider's
  terms before redistributing generated text.
- Operator-supplied essays: whatever you record in `raw/human/manifest.json`.
  Files with no manifest entry are ingested with `license: unspecified` so the
  gap is visible in the dataset card rather than silently assumed to be fine.

## Known limitations

1. The human class rests on **36 hand-authored seeds**. Grouped splitting means
   the effective number of independent human documents is 36, not the sample
   count. Confidence intervals are correspondingly wide, and every reported human
   metric should be read with that in mind.
2. Those seeds were written **for this repository**, not collected from real
   applicants, so they may under-represent real variation in student writing.
3. The **L2-English subset simulates a register** rather than sampling real
   second-language writers. It contains 4 seed groups, which is enough to detect
   a large disparity and nowhere near enough to rule one out.
4. The offline `ai_generated` generator is **template-based**. It reproduces the
   measurable register of instruction-tuned prose (regular sentence lengths,
   dense formal connectives, tricolons, nominalised abstractions, no typos) but
   not its semantics, and successive essays can be somewhat incoherent.
5. Template reuse makes the offline machine class **trivially separable at the
   sentence level** (validation ROC-AUC ≈ 1.0). That is a property of the corpus,
   not a capability of the detector.
6. Essay length is matched across classes on purpose, so the classifier cannot
   learn "long = machine". Without that, the classes differed by ~200 words.

## Regenerating

```powershell
# Offline, no API key, fully reproducible (fixed seeds)
uv run python -m ml.generation.bootstrap_corpus

# With a real model (recommended): replaces the two machine classes
$env:GROQ_API_KEY = "gsk_..."
uv run python -m ml.generation.generate_ai_essays --per-model 30 --replace
uv run python -m ml.generation.polish_essays --replace

# Then always
uv run python -m ml.training.prepare_dataset
```

## Adding real essays

This is the single highest-value improvement available.

1. Put one essay per `.txt` file in `data/raw/human/`.
2. Add `data/raw/human/manifest.json`:

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

3. Re-run `prepare_dataset`, `extract_features`, `train`, `evaluate`.

Files shorter than 80 words are skipped. `.gitignore` excludes
`data/raw/human/*.txt` so real essays cannot be committed by accident, and the
failure report withholds excerpts from any document whose source is
`ingested_real`.

Only use text you have the right to use. Get consent. Anything else is not worth
the model improvement.
