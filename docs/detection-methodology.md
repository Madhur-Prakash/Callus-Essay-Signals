[Home](../README.md) · [Docs index](README.md) · [Architecture](architecture.md) —
**Methodology** · [Dataset](dataset.md) · [Evaluation](evaluation.md) ·
[API](api.md) · [Privacy](privacy.md) · [Limitations](limitations.md)

---

# Detection methodology

What gets measured, how those measurements become a decision, and where the line
between instrument and judge is drawn. The corpus these features are learned from
is described in [dataset.md](dataset.md); how well the result performs is in
[evaluation.md](evaluation.md).

## The core commitment

The classification is produced by **our own trained classifier**. A local causal
language model is used as a *measuring instrument* — it reports how probable each
token was given the tokens before it — and nothing else. No hosted chat model is
called at any point during analysis, and there is no code path from the request
handler to one.

```
NOT this                            This
────────                            ────
Essay                               Essay
  ↓                                   ↓
Chat model                          Tokeniser
  ↓                                   ↓
"Is this AI?"                       Local causal LM (distilgpt2)
  ↓                                   ↓
Verdict                             per-token log prob / entropy / rank
                                      ↓
                                    Feature extraction (~411 document features)
                                      ↓
                                    Our trained, calibrated classifier
                                      ↓
                                    Confidence banding + abstention
                                      ↓
                                    Deterministic evidence engine
                                      ↓
                                    Result
```

The one place a hosted model appears is **offline dataset generation**
(`ml/generation/`), which builds training data and is never imported by `app/`.

## Why a language model at all

Because "how surprising is this word here?" is a real, measurable property of
text, and it is not available any other way. What the model is *not* asked is
anything about authorship. The distinction matters: a thermometer measures
temperature, it does not diagnose the patient.

## Layer 1 — segmentation

`app/services/nlp.py`. spaCy `en_core_web_sm` with NER and the lemmatiser
excluded (they cost time and contribute nothing here). Paragraphs split on blank
lines, falling back to single newlines when the text has none. Every sentence
keeps character offsets into the normalised essay, so the frontend highlighter,
the LM token alignment and the persisted rows all agree on where a sentence is.

If spaCy is unavailable the module degrades to a regex splitter and records
`segmentation_backend="regex"`, which surfaces in the response as a warning. The
loss of POS/dependency features is visible rather than silent.

## Layer 2 — stylometry

`app/services/stylometry.py`. Pure measurements over text and spaCy spans.

**Surface / lexical** (`sty_`, 37 features): word and character counts, mean and
std word length, long/short word ratios, syllables per word, type-token ratio,
**root type-token ratio** (Guiraud — far less length-dependent than raw TTR,
which matters because sentences differ in length by an order of magnitude), hapax
ratio, stopword and content-word ratios, punctuation density, per-mark rates
(comma, semicolon, colon, dash, parenthesis, quote, exclamation, question,
ellipsis), digit and uppercase ratios, contraction rate, colloquial-marker rate,
LLM-register phrase rate, transition word and phrase rates, sentence-initial
transition flag, hedge rate, intensifier rate, first-person rate, nominalisation
rate, commas per clause.

**Syntactic** (`syn_`, 70 features): normalised distributions over 17 universal
POS tags and 32 dependency labels, clause counts, mean and max dependency depth,
mean and max dependency distance, passive ratio, subordination and coordination
ratios, noun/verb, adjective/noun and adverb/verb ratios, function-word ratio, POS
and dependency entropy, and sentence-opener flags.

Word lists live in `app/services/lexicons.py` under `LEXICON_VERSION`, so a
feature value can always be traced back to a concrete rule.

> A bug worth recording: dependency depth was initially computed by walking
> `token.head` until `node.head is node`. spaCy builds a fresh `Token` object on
> every `.head` access, so that identity check is never true — every depth was
> silently pinned at the loop guard, and the whole depth feature block was
> constant and useless. It is now `node.head.i != node.i`. A test asserts the
> depths vary (`test_dependency_depth_is_measured_not_pinned`).

## Layer 3 — language-model probabilities

`app/services/probability_analyzer.py`. For every token: log probability,
probability, full-distribution predictive entropy, rank of the observed token, and
the gap to the model's own top choice.

**Sliding window.** One forward pass per window over the whole essay, not one pass
per sentence. With `LM_MAX_WINDOW=512` and `LM_STRIDE=384`, every token is scored
exactly once with up to 128 tokens of carried-over left context. Essays longer
than the model's context window are therefore handled correctly rather than
truncated. A BOS token is prepended so the document's first token also has a
conditioning context.

**Aggregation** (`lm_`, 32 features): mean/median/std/min/max log probability,
p10/p90, IQR, perplexity and log perplexity (clamped so one very surprising token
in a two-token sentence cannot produce an `inf` that poisons standardisation),
mean/std probability, probability variance, entropy statistics, mean/median/std
log rank, fraction of tokens in the top 1 / 10 / 100 and beyond rank 1000,
fraction above 50%/90% and below 5%/1% probability, mean and max top-1 gap, and
mean surprisal normalised by the distribution's own entropy.

Token→sentence attribution uses character offsets, anchored on the first
non-whitespace character of each token — GPT-2 tokens carry their leading space,
so a naive offset comparison drops the opening token of every sentence.

**Low perplexity is not evidence of machine authorship.** Clear, conventional
human prose scores low too. It is one feature among ~411.

## Layer 4 — burstiness

`app/services/burstiness.py` (`bur_`, 30 features). Human prose tends to be
*bursty*: long sentences next to short ones. Measured with mean/median/std
sentence length, coefficient of variation, IQR, range, the Goh & Barabási
burstiness index `(σ−μ)/(σ+μ)`, binned length entropy, adjacent-difference
statistics, lag-1 autocorrelation, the fraction of sentences within 20% of the
mean, direction-change rate, and coefficients of variation for word length,
punctuation, TTR, clause count, dependency depth and log probability.

**Uniformity alone is not evidence either.** Plenty of people write evenly, and a
model can be prompted to vary its sentence length — one of the six generation
strategies does exactly that.

## Layer 5 — repetition

`app/services/repetition.py` (`rep_`, 24 features). Three kinds, measured
separately because they mean different things:

- **Lexical n-gram repetition** — same words, same order. Ambiguous: appears in
  machine text drawing on a phrase bank *and* in human drafts written quickly.
- **Syntactic template repetition** — the same POS skeleton across sentences.
  The most machine-specific of the three, and it survives paraphrasing.
- **Discourse repetition** — repeated sentence openers and connectives.

Concrete repeated spans and templates are returned for display, filtered through
`UNINFORMATIVE_NGRAM_WORDS` so "is one of the" recurring three times is not
presented as evidence.

## Layer 6 — within-essay style shift

`app/services/style_shift.py` (`ctx_` per sentence, `shift_` per document).

The most useful question for the realistic case is not *does this look like AI?*
but *does this look like the rest of this essay?* Every author has a baseline; a
passage that departs from it sharply is worth surfacing.

Per sentence: clipped z-scores against the document for 16 tracked quantities,
adjacent-sentence differences, Jensen-Shannon divergence of the POS distribution
against the document / previous sentence / own paragraph, function-word profile
cosine, and a scaled style distance over a curated register-bearing feature
subset. Per document: dispersion of all of the above, a transparent two-window
change-point count, and per-paragraph statistics.

**A style shift is evidence of editing or register change, not of authorship.**
Quoting a source, moving from narrative to reflection, or simply writing a
stronger conclusion all cause genuine shifts in human writing. The UI says so.

## Layer 7 — reference corpus similarity

`app/services/corpus_analyzer.py` (`cor_`, 18 features). How close is this
writing to the human reference corpus, and how close to the machine one?

Representations are **style-bearing but deliberately topic-free**: character 3–5
grams (`char_wb`), POS 1–3 grams, function-word frequency profile, and POS
distribution compared by Jensen-Shannon divergence. Topical word n-grams are
excluded on purpose — a TF-IDF model over content words would learn "essays about
robotics are human" from whatever topics dominate the training split, and would
then collapse on the held-out-topic evaluation.

Centroids are fit on the **training split only** and stored in the artifacts.
`ml/training/extract_features.py` runs in two passes to guarantee this: pass 1
extracts everything else and collects the training views, then the reference is
fit, then pass 2 recomputes only the `cor_*` block from cached views — no second
language-model pass.

## Feature assembly

`app/services/feature_extractor.py` is the single source of truth for the feature
space. Both the offline training pipeline and the live API call it, which is what
guarantees train/serve consistency — there is no second implementation to drift.

- **Sentence vector, 189 features**: `lm_ + sty_ + syn_ + ctx_ + cor_`
- **Document vector, 411 features**: curated aggregates of the sentence blocks
  (mean/std/min/max/p25/p75 for 20 features, mean/std for 74 more) + whole-document
  LM features + `bur_ + rep_ + shift_ + doc_ + cor_`

The aggregate list is curated rather than exhaustive: the full cross-product would
be ~1,200 columns against a few hundred training documents.

## Classification

`app/services/classifier.py` + `ml/training/train.py`.

**Document model** — three-class (human / ai_generated / ai_polished) over the 411
document features. Candidates: logistic regression at two regularisation
strengths (with `SelectKBest` pre-selection), random forest, LightGBM. Selected by
macro F1 under `GroupKFold` over `group_id` on the training split only.

**Sentence model** — binary human vs machine over the 189 sentence features, used
for highlighting. Logistic regression with strong regularisation (`C=0.05`),
deliberately: on the bootstrap corpus the two sentence classes are nearly linearly
separable, and a weakly regularised model responds by driving coefficients toward
infinity — slow convergence, saturated probabilities, worse transfer.

Trained **only** on sentences from `human` and `ai_generated` documents. Rows from
`ai_polished` documents are excluded because their sentence-level authorship is
genuinely mixed: some sentences in a polished essay are untouched human text, and
labelling them all "machine" would teach the model that ordinary human sentences
are machine-like — precisely the failure mode that produces false positives on
real students.

## Calibration and abstention

`app/services/calibration.py`. Platt scaling fit on the validation split.
Isotonic regression is available but downgrades to sigmoid automatically when a
class has fewer than 10 calibration examples — it is non-parametric and would
memorise a split this size.

> scikit-learn 1.9 removed `CalibratedClassifierCV(cv="prefit")`; the code uses
> `sklearn.frozen.FrozenEstimator` with a fallback for older versions. The first
> implementation swallowed the resulting exception and silently shipped an
> uncalibrated model. Calibration failures are now logged at ERROR with their
> cause, because a silently uncalibrated model is worse than a loudly broken one:
> the UI would go on presenting raw scores as calibrated probabilities.

**The system declines to answer** (`insufficient_evidence`) when:

- the top class is below 45%, or
- the top two classes are within 10 percentage points, or
- the document has fewer than 5 sentences or 120 words.

Sentences under 5 words or 5 scored tokens are forced to `uncertain` regardless of
score. A four-word sentence gives the model three predictions to work from, which
cannot support a claim in either direction — this is the most common source of
misleading sentence-level highlighting in detectors that skip the check.

## Explanation

`app/services/explanation_engine.py`. **No language model is asked to explain
anything.** Three kinds of evidence, kept separate because they support different
claims:

1. **Reference-relative** — the measured value's percentile within the human
   training distribution, interpolated from stored p5–p95 tables
   (`reference_stats.json`, computed on the training split only). Where no
   reference exists the meter is marked `available: false` and says so rather
   than implying a comparison.
2. **Within-essay** — "perplexity here is 14.2 against an essay median of 31.8".
   Needs nothing but the essay, works before training, and is the most defensible
   kind: the comparison group is the author.
3. **Model-derived** — signed per-feature contributions straight out of the
   classifier (`coef × standardised_value`, exact for the linear models). The
   model's own arithmetic, not a narrative about it.

Every statement is phrased as a measurement, never as a conclusion about who
wrote the text. A test asserts no statement contains authorship claims.

## What this cannot do

- It cannot establish authorship. Text does not carry a signature.
- It cannot reliably detect a light copy-edit. A grammar-only pass leaves almost
  every measurement intact, and the model says so by abstaining or by calling it
  human.
- It has no defence against a writer who deliberately targets these features.
  Everything measured here is public and imitable.
- It over-flags human writing on the current corpus (see
  [evaluation.md](evaluation.md)) — human recall is well below the machine
  classes, and that gap is the honest headline.
