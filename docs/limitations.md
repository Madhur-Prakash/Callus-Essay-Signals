[Home](../README.md) · [Docs index](README.md) · [Architecture](architecture.md) ·
[Methodology](detection-methodology.md) · [Dataset](dataset.md) ·
[Evaluation](evaluation.md) · [API](api.md) · [Privacy](privacy.md) —
**Limitations**

---

# Limitations and known failure cases

> [!CAUTION]
> Read this before acting on anything this system outputs.

The measurements behind each limitation are in
[detection-methodology.md](detection-methodology.md); the numbers that demonstrate
them are in [evaluation.md](evaluation.md).

## The one-sentence version

This detector over-flags human writing. On the held-out split it recovers the
machine-written class almost perfectly and misclassifies a large share of genuine
human essays as machine-polished. Overall accuracy hides that, which is why the
evaluation report leads with human recall.

## Detection is not proof of authorship

The system measures properties of text. Text does not carry a signature. The
strongest honest claim available is:

> "This passage has statistical properties in common with the machine-written
> examples in our evaluation data."

Not: "this was written by AI." The measurements cannot distinguish a machine from a
human who writes formally, evenly, and without contractions — because *there is no
measurable difference between those two things in the text alone*.

## Ranked limitations

### 1. The training corpus is synthetic (most severe)

Out of the box the human class is 36 hand-authored seed essays and the machine
classes come from a template generator and a rule-based editor. Every metric
measures separability of those three generators. It is not an estimate of
real-world performance and must not be quoted as one.

Mitigation: run the Groq generators for real machine text, and add real essays to
`data/raw/human/`. See [dataset.md](dataset.md).

### 2. False positives on human writing

Human recall is far below the machine classes. Most errors are human essays called
`ai_polished` — the two classes overlap heavily by construction, because a lightly
polished essay *is* mostly human text.

Consequence: if this were used on real applicants, a substantial fraction of
honest writers would be flagged. That is disqualifying for any high-stakes use.

### 3. Fairness for second-language English writers is not established

Published work consistently finds AI detectors over-flag writing by people who
learned English as an additional language. Nothing here contradicts that.

The L2 subset in this corpus is a **simulated register** across 4 seed groups, not
writing collected from real L2 authors. That is enough to detect a large disparity
and nowhere near enough to rule one out. The evaluation reports the rate with a
Wilson interval and an `underpowered` flag; overlapping intervals mean "cannot
tell", not "fair".

### 4. The `ai_polished` class is inherently hard

A grammar-only pass leaves nearly every measurement intact. The system usually
calls those essays human — correctly, in the sense that most of the text *is*
human, and unhelpfully, in the sense that the edit happened. Where the edit is
heavy enough to be detectable, it is often indistinguishable from fully generated
text. The distinction between "heavily rewritten" and "generated" may not be
recoverable from text alone.

### 5. Sentence-level scores look far better than they are

Validation ROC-AUC for the sentence model is ≈1.0 on the bootstrap corpus. That is
a property of a finite template bank, not a capability. Treat sentence
highlighting as *where to look*, never as per-sentence verdicts. The training run
emits a `separability_warning` when this happens.

### 6. Small effective sample size

Grouped splitting means the effective number of independent human documents is 36.
Confidence intervals are wide; differences of a few points between feature sets are
inside fold-to-fold noise. The ablation table reports CV standard deviations for
this reason.

### 7. The instrument model is small and old

`distilgpt2` is an 82M-parameter model from 2019. Text whose vocabulary or topic
sits outside its training distribution looks "surprising" regardless of who wrote
it — which means unusual-but-human writing is penalised. A larger instrument
(`LM_MODEL_NAME=gpt2-medium`) measures better at ~3× the cost.

### 8. Trivially evadable by anyone who reads this page

Every feature is public and imitable. Vary your sentence lengths, use
contractions, avoid "moreover", add a typo. There is no adversarial robustness
here and none is claimed.

### 9. Style shift has legitimate causes

Quoting a source, moving from narrative to reflection, or writing a stronger
conclusion all produce genuine style shifts in human writing. A shift means the
register changed, not that someone else wrote it.

### 10. English only

Lexicons, the spaCy model and the instrument model are all English. Behaviour on
other languages is undefined, not merely degraded.

### 11. No abstention on the current corpus

The abstention thresholds fire on genuinely ambiguous documents, but on the
bootstrap corpus the classifier is confident nearly everywhere, so the abstention
rate is ~0. The mechanism is tested directly in
`tests/test_classifier.py::TestVerdictConstruction`; it is the data that is
unrealistically clean, not the policy that is inert.

## Where the detector confidently fails

`ml/evaluation/find_failures.py` finds the highest-confidence errors and explains
each one from the model's own feature contributions, then names an improvement
keyed to the feature group that drove it. The full write-ups are in
`ml/evaluation/failure_report.json` and rendered on the Research page.

The recurring patterns:

**A human essay called `ai_polished`, driven by burstiness features.** The
measurements that trigger it describe register, not authorship. A human writing
evenly produces the same numbers as a machine.

**An `ai_polished` essay called `human`, when the edit was light or localised.**
Most sentences retain the original author's statistics and the document-level
aggregate is dominated by them. This is the honest limit of the method.

**Corpus-similarity features dominating.** With only a few dozen independent human
documents, the human centroid is a poor summary of human writing in general. This
is the first thing that improves when real essays are added.

## Research questions, answered

| Question                                                     | Answer from the current evaluation                                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Which features distinguish human and machine writing?        | On this corpus: reference-corpus similarity, sentence-initial transitions, transition-phrase counts, LM log-probability dispersion. See feature importance. |
| Does combining LM + stylometric features help?               | Yes, modestly — the hybrid set leads the stylometry-only baseline on the held-out split, by a margin comparable to CV noise. |
| Are LM features sufficient alone?                            | No. `lm_only` is the weakest feature set.                                                                    |
| Can it detect AI-polished writing?                           | Partially. Recall is reasonable; precision is poor because human essays are pulled into this class.          |
| Does within-document style shift improve detection?          | Small positive effect on the held-out split, inside CV noise during training. Retained because the per-paragraph shift analysis is a user-facing output in its own right. |
| Does it generalise across topics?                            | For the machine class, yes — recall holds on topics absent from training. Unknown for human writing: the held-out-topic slice contains no human documents, a real gap in the evaluation design. |
| Does it generalise across generators?                        | Same answer, same caveat.                                                                                    |
| How does performance change with length?                     | Short documents are markedly worse. Every distributional estimate behind these features is noisy at low sentence counts. |
| Which features contribute most to false positives?           | Burstiness and corpus similarity, per the failure analysis.                                                  |
| Does it disproportionately flag L2 English writing?          | Not established either way — the subset is underpowered. Assume the published risk applies.                  |
| Where does it confidently fail?                              | Three documented cases in `failure_report.json`, each with measured features and a proposed fix.             |

## Do not

- Do not use this as evidence in an academic-integrity process.
- Do not reject an applicant based on it, in whole or in part.
- Do not present its output to a student as a finding about their honesty.
- Do not quote its accuracy as a real-world figure while the regime is `bootstrap`.

## Reasonable uses

- Deciding which essays in a large pile to read more attentively.
- Prompting a conversation with a writer about their process.
- Research into which textual features carry signal, and which do not.
- Demonstrating to a non-technical audience why this problem is hard.
