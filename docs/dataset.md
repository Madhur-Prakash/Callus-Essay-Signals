[Home](../README.md) · [Docs index](README.md) · [Architecture](architecture.md) ·
[Methodology](detection-methodology.md) — **Dataset** · [Evaluation](evaluation.md) ·
[API](api.md) · [Privacy](privacy.md) · [Limitations](limitations.md)

---

# Dataset methodology

Why the corpus is shaped the way it is, and what each decision costs. The measured
consequences are in [evaluation.md](evaluation.md); the honest summary of what this
corpus cannot support is in [limitations.md](limitations.md).

Operational instructions — layout, regenerating, adding real essays — live in
[`backend/data/README.md`](../backend/data/README.md), with sourcing rules in
[`backend/data/raw/README.md`](../backend/data/raw/README.md).
This document is about the *decisions* — why the corpus is shaped the way it is,
and what each choice costs.

## The honest starting position

A detector needs three things that are hard to get legally and ethically:

1. **Real admissions essays.** They are private documents. Scraping them without
   consent is a licensing violation and an ethics failure, so this project does
   not do it.
2. **Real machine-written essays.** Obtainable, with an API key.
3. **Real machine-edited human essays.** Requires (1) as input.

So the project ships a **bootstrap corpus** that makes the whole pipeline runnable
and reproducible with no network access and no API key, and is labelled everywhere
as what it is. `data_regime` appears in the dataset manifest, the model metadata,
the API response, the evaluation report and a UI banner. Three values:

| Regime      | Meaning                                                    |
| ----------- | ---------------------------------------------------------- |
| `bootstrap` | Everything synthetic. Metrics measure generator separability. |
| `mixed`     | Real model output; human class still a proxy.               |
| `real`      | Operator-supplied real essays plus real model output.       |

## Why three classes

```
HUMAN            wrote it themselves
AI_GENERATED     a model wrote it
AI_POLISHED      a human wrote it, a model edited it   ← the realistic case
```

A two-class detector answers the wrong question. Almost nobody submits raw model
output; what actually happens is a student writes a draft and asks a model to
"improve" it. That produces text which is *mostly human*, and a binary detector
must either call it human (missing the assistance) or machine (accusing someone of
work they largely did). The third class exists so the system can say what actually
happened.

The fourth output, `insufficient_evidence`, is not a data class — it is the
system declining to answer. See [detection-methodology.md](detection-methodology.md).

## Producing AI_POLISHED

Six transforms spanning light to heavy, applied to the *real seed text*:
`clarity`, `vocabulary`, `restructure`, `formalize`, `shorten`,
`partial_paragraph`.

Two design points worth naming:

**`partial_paragraph` rewrites exactly one paragraph and leaves the rest
byte-identical.** This is the case the within-essay style-shift analysis exists
for, and it is the only transform where the ground truth is genuinely localised.

**`merge_short` is modelled explicitly** in the rule-based polisher, because
joining adjacent short sentences is the single most consequential thing an
"improve the flow" edit does to the burstiness signature. Leaving it out would
have made the offline polished class artificially easy.

Every pair is written to `polish_pairs.jsonl` with word delta and overlap, so any
transform can be audited rather than trusted.

**`grammar_only` is excluded offline.** A rule-based spacing fixer applied to clean
seed text returns the input unchanged, which would put identical text under two
labels. It is generated on the Groq path, where the model also rephrases slightly.
The realistic consequence is worth stating plainly: a genuine grammar-only edit is
*nearly undetectable*, and a detector that claimed otherwise would be lying.

## Generator diversity

A dataset built from one prompt teaches a detector to recognise one prompt. The
generation layer varies five axes, and every sample records all five:

- **model family** — 4 Groq models, or 5 offline personas
- **prompt strategy** — `plain`, `coached`, `persona`, `anti_detection`,
  `structured`, `sensory`
- **topic** — 16 training + 6 held-out
- **length band** — short / medium / long
- **sampling** — 4 temperatures × 2 top-p values

`anti_detection` explicitly instructs the model to vary sentence length, use
contractions, and avoid the usual giveaway vocabulary. Without adversarial samples
the reported accuracy would be optimistic in a way that matters.

The five offline personas differ along the axes that actually separate
instruction-tuned prose from student drafts: connective density, target sentence
length, tricolon rate, nominalisation preference, and paragraph symmetry. One
persona is reserved for the cross-generator test and its parameters sit
deliberately *between* the four training personas — the test should ask "does this
transfer to an unseen generator?", not "does it transfer to an extreme?".

## Length matching, and why it mattered

The first version of the offline corpus had `ai_generated` averaging 456 words
against 241 for `human`. A classifier trained on that learns **"long = machine"** —
a shortcut that says nothing about writing and would collapse on real essays,
where length is set by the application's word limit for everyone.

Generation now samples a target word count from the empirical human distribution
and trims whole sentences to reach it. Current means: human 241, `ai_generated`
221, `ai_polished` 238.

This is the general lesson: **the first thing to check in a synthetic corpus is
whether the classes differ in something trivial.**

## Leakage control

Five mechanisms, all enforced in code:

**1. Split by group, never by sample.** `group_id` ties together every document
derived from the same underlying essay. Critically, an AI-polished essay is
assigned **the same group as the human original it came from** — otherwise the
model would be evaluated on paraphrases of essays it had memorised.

**2. Held-out topics are test-only.** Six topics appear in no training document,
which is what makes the topic-generalisation number mean something.

**3. A held-out generator is test-only.**

**4. Near-duplicate audit.** Group ids cannot catch everything: two independently
generated essays on different topics from the same template bank can share 60% of
their 5-grams. Any validation/test document whose 5-gram containment against a
training document exceeds 0.55 is reported and, by default, dropped. On the
bootstrap corpus this removes ~20 documents, all from the template generator.
Keeping them would inflate AI-class recall for reasons unrelated to detection.

**5. Stratify by label signature *and* the L2-English flag.** Stratifying on labels
alone put every L2 seed into train/validation, which left the test split with zero
L2 human documents — and made the mandatory bias measurement impossible to run.
Finding that required actually reading the bias output rather than assuming it
worked.

## Split proportions

Target 70/15/15 by group. The realised split is test-heavy because rules 2 and 3
force whole groups into test. That is a justified deviation, not an accident: a
smaller, honest test set that actually contains unseen topics and an unseen
generator is worth more than a nominally correct ratio.

## The reference corpus is topic-free on purpose

`cor_*` features compare a document against human and machine centroids built from
character n-grams, POS n-grams and function-word profiles — **not** topical word
n-grams. A TF-IDF model over content words would learn "essays about robotics are
human" from whatever topics dominate the training split, then collapse on the
held-out-topic evaluation. Keeping the representation topic-free is what makes the
topic-generalisation number interpretable.

Centroids are fit on the **training split only**, which is why feature extraction
runs in two passes (see [detection-methodology.md](detection-methodology.md)).

## The L2-English subset

Four seed groups written in a simulated second-language register: article
omission, preposition variation, simpler connectives, calque phrasing. Flagged with
`l2_english: true`.

This is a **proxy, and a weak one**. It is enough to detect a large disparity and
nowhere near enough to rule one out. The evaluation reports the rate with a Wilson
interval and an explicit `underpowered` flag; overlapping intervals mean "cannot
tell", never "fair". Published work consistently finds AI detectors over-flag L2
English writing, and nothing in this evaluation contradicts that.

Simulating a register is not the same as sampling a population. The right fix is
real L2 writing collected with consent.

## Known limitations

Ranked, with the most severe first:

1. The human class rests on **36 hand-authored seeds**, so the effective number of
   independent human documents is 36 regardless of sample count. Every human
   metric carries wide uncertainty.
2. Those seeds were written **for this repository**, not collected from applicants.
3. The L2 subset simulates a register rather than sampling real writers.
4. The offline `ai_generated` generator is template-based: it reproduces machine
   *register* but not machine *semantics*, and successive essays can be somewhat
   incoherent.
5. Template reuse makes the offline machine class trivially separable at the
   sentence level (validation ROC-AUC ≈ 1.0) — a corpus property, not a capability.
6. Class balance is close but not exact; metrics are reported per class and
   balanced accuracy is reported alongside accuracy for this reason.

## Reproducibility

Fixed seeds throughout: `GLOBAL_SEED = 20260814` for generation, `SPLIT_SEED = 42`
for splitting, `RANDOM_SEED = 42` for training. Re-running the pipeline on the same
inputs produces the same corpus, the same splits and the same model. Versions
(`dataset_version`, `features_version`, `lexicon_version`, `model_version`) are
recorded in the artifacts and exposed by `GET /api/v1/model/info`.
