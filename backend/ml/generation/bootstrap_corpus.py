"""Offline bootstrap corpus generator.

Purpose
-------
Make the entire pipeline runnable and reproducible with **no network access and
no API key**, so that a reviewer can clone the repo and get real numbers out of a
real model rather than a placeholder.

What this is honestly NOT
-------------------------
It is not a substitute for real data. Three classes are produced:

``human``        variants of the hand-authored seed essays in ``data/seeds/``,
                 with human-style editing noise (typos, contraction toggles,
                 filler, uneven paragraphing).
``ai_generated`` a *procedural* generator that composes essays from a template
                 bank under four "model personas". It reproduces the measurable
                 register of instruction-tuned output - regular sentence lengths,
                 dense formal connectives, tricolons, nominalised abstractions,
                 symmetric paragraphs, zero typos - but it is not an LLM.
``ai_polished``  a *rule-based* editor applied to the real seed text: contraction
                 expansion, hedge and colloquialism removal, connective
                 insertion, vocabulary upgrading, sentence-length regularisation.
                 This one is the most faithful of the three, because the input is
                 genuine text and the transformation is a bounded edit - exactly
                 the shape of the real threat model.

A classifier trained only on this corpus is learning to separate *these
generators*, not humans from machines. Every metric produced from it is labelled
``bootstrap`` in the evaluation report, and ``/api/v1/model/info`` reports
``data_regime="bootstrap"`` so the UI can warn the user. Run
``ml.generation.generate_ai_essays`` and ``ml.generation.polish_essays`` with a
``GROQ_API_KEY`` to replace the two machine classes with real model output.

Usage
-----
    uv run python -m ml.generation.bootstrap_corpus
    uv run python -m ml.generation.bootstrap_corpus --human-variants 6 --ai-per-topic 14
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger, log_event
from app.services.lexicons import FUNCTION_WORDS
from ml.dataset_schema import Sample, dataset_paths, write_jsonl
from ml.generation.prompts import HELD_OUT_TOPICS, POLISH_TRANSFORMS, TRAIN_TOPICS

logger = get_logger("ml.bootstrap")

SEED_FILES = ("human_seeds_part1.json", "human_seeds_part2.json")
GLOBAL_SEED = 20260814

# ``grammar_only`` is excluded from the offline transform list on purpose: a
# rule-based spacing/punctuation fixer applied to already-clean seed text
# produces a byte-identical document, which would put the same text under two
# different labels. It IS generated on the Groq path, where the model also
# rephrases slightly. See docs/dataset.md.
OFFLINE_POLISH_TRANSFORMS: tuple[str, ...] = (
    "clarity",
    "vocabulary",
    "restructure",
    "formalize",
    "shorten",
    "partial_paragraph",
)

# Words that are safe to lower-case when a connective is prepended to a
# sentence. Prepending "Moreover," to "I began..." must not yield "Moreover, i
# began..." - a broken-capitalisation artefact would be a trivially learnable
# giveaway that has nothing to do with machine register.
_LOWERCASEABLE: frozenset[str] = frozenset(FUNCTION_WORDS) - {"i"} | {
    "progress", "success", "recognising", "instead", "each", "over", "my",
    "what", "there", "these", "those", "it", "the", "this", "when", "after",
    "before", "eventually", "gradually",
}


# =========================================================================== #
# 1. Human seed loading + human-style variation
# =========================================================================== #
def load_seeds(data_dir: Path) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    seed_dir = data_dir / "seeds"
    for name in SEED_FILES:
        path = seed_dir / name
        if not path.exists():
            continue
        seeds.extend(json.loads(path.read_text(encoding="utf-8")))
    if not seeds:
        raise FileNotFoundError(f"No seed essays found in {seed_dir}")
    return seeds


_FILLERS = (
    "Anyway.",
    "I mean, it worked.",
    "Which, fine.",
    "Not that it matters now.",
    "So there is that.",
    "I still think about it.",
)
_TYPO_SWAPS = (
    ("the ", "teh "),
    ("and ", "adn "),
    ("that ", "taht "),
    ("really", "realy"),
    ("definitely", "definately"),
    ("separate", "seperate"),
    ("receive", "recieve"),
    ("occurred", "occured"),
)
_CONTRACTIONS = (
    ("do not", "don't"),
    ("did not", "didn't"),
    ("does not", "doesn't"),
    ("cannot", "can't"),
    ("could not", "couldn't"),
    ("would not", "wouldn't"),
    ("is not", "isn't"),
    ("was not", "wasn't"),
    ("it is", "it's"),
    ("I am", "I'm"),
    ("I have", "I've"),
    ("I would", "I'd"),
    ("that is", "that's"),
    ("there is", "there's"),
)


def human_variants(
    seed: dict[str, Any], n_variants: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Return ``(variant_tag, text)`` pairs for one seed.

    The first entry is always the unmodified seed. The rest apply *human* editing
    noise - the kinds of change a student makes between drafts. Crucially these
    do not systematically reduce burstiness or add formal connectives, so they do
    not accidentally turn a human sample into a machine-register one.
    """
    paragraphs = list(seed["paragraphs"])
    variants: list[tuple[str, str]] = [("original", "\n\n".join(paragraphs))]

    recipes = [
        ("contract", ("contract",)),
        ("expand_typo", ("expand", "typo")),
        ("filler", ("filler",)),
        ("reorder", ("reorder_within_paragraph",)),
        ("trim", ("drop_paragraph",)),
        ("split_merge", ("split_paragraph", "contract")),
        ("typo_filler", ("typo", "filler")),
        ("rough_draft", ("expand", "filler", "typo", "reorder_within_paragraph")),
    ]
    rng.shuffle(recipes)

    for tag, operations in recipes[: max(0, n_variants - 1)]:
        current = [p for p in paragraphs]
        for op in operations:
            current = _apply_human_op(current, op, rng)
        variants.append((tag, "\n\n".join(p for p in current if p.strip())))
    return variants


def _apply_human_op(paragraphs: list[str], op: str, rng: random.Random) -> list[str]:
    if op == "contract":
        out = []
        for para in paragraphs:
            for long, short in _CONTRACTIONS:
                if rng.random() < 0.6:
                    para = para.replace(long, short)
            out.append(para)
        return out

    if op == "expand":
        out = []
        for para in paragraphs:
            for long, short in _CONTRACTIONS:
                if rng.random() < 0.5:
                    para = para.replace(short, long)
            out.append(para)
        return out

    if op == "typo":
        out = []
        for para in paragraphs:
            if rng.random() < 0.55:
                find, replace = rng.choice(_TYPO_SWAPS)
                idx = para.find(find)
                if idx > 0:
                    para = para[:idx] + replace + para[idx + len(find) :]
            out.append(para)
        return out

    if op == "filler":
        out = []
        for para in paragraphs:
            if rng.random() < 0.4:
                filler = rng.choice(_FILLERS)
                if rng.random() < 0.5:
                    para = f"{filler} {para}"
                else:
                    para = f"{para} {filler}"
            out.append(para)
        return out

    if op == "reorder_within_paragraph":
        out = []
        for para in paragraphs:
            sentences = _split_sentences(para)
            if len(sentences) >= 4 and rng.random() < 0.6:
                i = rng.randrange(1, len(sentences) - 1)
                sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]
            out.append(" ".join(sentences))
        return out

    if op == "drop_paragraph" and len(paragraphs) > 3:
        index = rng.randrange(1, len(paragraphs) - 1)
        return paragraphs[:index] + paragraphs[index + 1 :]

    if op == "split_paragraph":
        out: list[str] = []
        for para in paragraphs:
            sentences = _split_sentences(para)
            if len(sentences) >= 5 and rng.random() < 0.5:
                cut = len(sentences) // 2
                out.append(" ".join(sentences[:cut]))
                out.append(" ".join(sentences[cut:]))
            else:
                out.append(para)
        return out

    return paragraphs


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


# =========================================================================== #
# 2. Procedural AI_GENERATED proxy
# =========================================================================== #
# Four "model personas". They differ along the axes that actually separate
# instruction-tuned output from student drafts, so a classifier trained here
# learns transferable structure rather than a single fingerprint.
PERSONAS: dict[str, dict[str, Any]] = {
    "proxy-formal-70b": {
        "connective_rate": 0.75,
        "target_len": (17, 24),
        "tricolon_rate": 0.35,
        "nominalise": 0.7,
        "paragraphs": (5, 6),
        "sentences_per_paragraph": (4, 5),
    },
    "proxy-warm-8b": {
        "connective_rate": 0.45,
        "target_len": (13, 19),
        "tricolon_rate": 0.2,
        "nominalise": 0.4,
        "paragraphs": (4, 5),
        "sentences_per_paragraph": (3, 5),
    },
    "proxy-verbose-mixtral": {
        "connective_rate": 0.85,
        "target_len": (21, 28),
        "tricolon_rate": 0.45,
        "nominalise": 0.8,
        "paragraphs": (5, 7),
        "sentences_per_paragraph": (4, 6),
    },
    "proxy-terse-gemma": {
        "connective_rate": 0.3,
        "target_len": (11, 16),
        "tricolon_rate": 0.15,
        "nominalise": 0.3,
        "paragraphs": (4, 5),
        "sentences_per_paragraph": (3, 4),
    },
    # Reserved for the cross-model generalisation test: never appears in the
    # training split (see ml/training/prepare_dataset.HELD_OUT_MODELS). Its
    # parameters sit deliberately *between* the four training personas, so the
    # test asks "does this transfer to an unseen generator?" rather than "does it
    # transfer to an extreme?".
    "proxy-heldout-qwen": {
        "connective_rate": 0.6,
        "target_len": (15, 21),
        "tricolon_rate": 0.28,
        "nominalise": 0.55,
        "paragraphs": (4, 6),
        "sentences_per_paragraph": (3, 5),
    },
}

TRAINING_PERSONAS: tuple[str, ...] = (
    "proxy-formal-70b",
    "proxy-warm-8b",
    "proxy-verbose-mixtral",
    "proxy-terse-gemma",
)
HELD_OUT_PERSONA = "proxy-heldout-qwen"

CONNECTIVES = (
    "Moreover,",
    "Furthermore,",
    "Additionally,",
    "Consequently,",
    "Ultimately,",
    "Indeed,",
    "In addition,",
    "As a result,",
    "Nevertheless,",
    "Importantly,",
    "In essence,",
    "Notably,",
)

_HOOKS = (
    "From an early age, I have been drawn to {activity}.",
    "For as long as I can remember, {activity} has occupied a central place in my life.",
    "My relationship with {activity} began not with success but with uncertainty.",
    "The first time I encountered {activity}, I understood almost nothing about it.",
    "There are few pursuits that have shaped me as profoundly as {activity}.",
    "It was through {activity} that I first began to understand my own capacity for growth.",
)
_CONTEXT = (
    "What began as a modest interest gradually developed into a genuine commitment.",
    "Over the following years I devoted an increasing share of my time to the pursuit.",
    "The environment in which I worked was demanding, and it required consistent effort.",
    "I approached the work methodically, building my understanding one step at a time.",
    "My community played an essential role in supporting this developing interest.",
    "Each stage of the process demanded a different kind of attention and discipline.",
)
_OBSTACLE = (
    "The most significant challenge arose when my initial approach proved inadequate.",
    "Progress was neither linear nor guaranteed, and there were periods of real difficulty.",
    "I encountered a setback that forced me to reconsider my fundamental assumptions.",
    "The obstacle was not merely technical but also personal, testing my resolve.",
    "There came a point at which my existing methods were no longer sufficient.",
    "What I had believed to be a straightforward task revealed unexpected complexity.",
)
_TURNING = (
    "The turning point came when I decided to rebuild my approach from first principles.",
    "Recognising the limits of my method, I sought guidance and revised my strategy.",
    "I began to document my process carefully, which transformed how I understood it.",
    "It was at this moment that I learned to separate the problem from my ego.",
    "Instead of persisting with a flawed approach, I chose to begin again deliberately.",
    "This realisation reframed the entire endeavour for me.",
)
_REFLECTION = (
    "The experience instilled in me a deeper appreciation for patience and iteration.",
    "I came to understand that meaningful progress depends on {abstract_noun} rather than talent.",
    "This process cultivated in me a durable capacity for {abstract_noun}.",
    "What I gained was not simply a skill but a framework for approaching difficulty.",
    "The lesson extended well beyond {activity} into every area of my academic life.",
    "I emerged with a clearer understanding of both my strengths and my limitations.",
)
_CONCLUSION = (
    "As I look toward university, I intend to bring this same commitment to {field}.",
    "I am eager to continue this work within a rigorous academic community.",
    "My experience has solidified my determination to pursue {field} at the university level.",
    "I hope to contribute this perspective to a community of similarly motivated scholars.",
    "These lessons will continue to guide my study of {field} and my work beyond it.",
    "I look forward to building on this foundation in a more demanding environment.",
)

_TRICOLONS = (
    "It required patience, precision, and a willingness to fail.",
    "The work demanded curiosity, discipline, and humility.",
    "I learned to observe carefully, to question assumptions, and to revise without resentment.",
    "Success depended on preparation, collaboration, and persistence.",
    "The process taught me to listen, to adapt, and to persevere.",
)

_ABSTRACT_NOUNS = (
    "perseverance",
    "resilience",
    "intellectual humility",
    "sustained attention",
    "collaboration",
    "self-discipline",
    "adaptability",
    "critical reflection",
)

_FIELDS = (
    "engineering",
    "the biological sciences",
    "computer science",
    "public policy",
    "applied mathematics",
    "environmental science",
    "economics",
    "the humanities",
)

_MOVE_ORDER = (
    ("hook", _HOOKS),
    ("context", _CONTEXT),
    ("obstacle", _OBSTACLE),
    ("turning", _TURNING),
    ("reflection", _REFLECTION),
    ("conclusion", _CONCLUSION),
)

# The prompt topics are phrased for a *prompt* ("write about a robotics project
# that failed"). The templates slot them in as an activity ("drawn to ..."), so
# each topic gets an activity phrasing here. Without this the generator produces
# ungrammatical text whose oddness - not its register - is what a classifier
# would pick up on.
TOPIC_ACTIVITY: dict[str, str] = {
    "a robotics project that failed": "robotics",
    "working in a family business": "my family's business",
    "caring for an ill relative": "caring for my grandmother",
    "losing an important competition": "competitive debate",
    "immigrating and learning English": "learning a new language",
    "years of a demanding sport": "competitive swimming",
    "a community volunteering project": "community service",
    "learning an instrument late": "the piano",
    "a minimum-wage summer job": "my summer job",
    "building software nobody used": "software development",
    "supporting a sibling with a disability": "supporting my brother",
    "an ensemble or team performance activity": "marching band",
    "changing schools repeatedly": "adapting to new schools",
    "repairing things as a hobby": "repairing bicycles",
    "a job in a library or archive": "working at the library",
    "a strategy game and how it taught self-criticism": "chess",
    "an unexpected friendship with an elderly neighbour": "my neighbour's garden",
    "a summer spent surveying a wetland": "wetland fieldwork",
    "learning to repair a sewing machine": "restoring sewing machines",
    "organising a school response to a local flood": "disaster relief organising",
    "discovering an interest in cartography": "cartography",
    "running the sound desk at a community radio station": "community radio",
}


def _prepend_connective(sentence: str, connective: str) -> str:
    """Attach a connective without mangling capitalisation."""
    first_word = re.split(r"\W+", sentence, maxsplit=1)[0]
    if first_word.lower() in _LOWERCASEABLE and first_word != "I":
        sentence = sentence[0].lower() + sentence[1:]
    return f"{connective} {sentence}"


def generate_procedural_essay(
    topic: str,
    persona_name: str,
    rng: random.Random,
    *,
    target_words: int | None = None,
) -> str:
    """Compose one machine-register essay.

    ``target_words`` matches the generated length to the human class so that
    document length cannot act as a shortcut feature. See ``build()``.
    """
    persona = PERSONAS[persona_name]
    activity = TOPIC_ACTIVITY.get(topic, topic)
    field = rng.choice(_FIELDS)
    n_paragraphs = rng.randint(*persona["paragraphs"])

    moves = list(_MOVE_ORDER)
    if n_paragraphs < len(moves):
        # Drop from the middle so hook and conclusion survive.
        while len(moves) > n_paragraphs:
            moves.pop(rng.randrange(1, len(moves) - 1))
    while len(moves) < n_paragraphs:
        moves.insert(len(moves) - 1, ("context", _CONTEXT))

    paragraphs: list[str] = []
    used: set[str] = set()
    for move_index, (_move_name, bank) in enumerate(moves):
        n_sentences = rng.randint(*persona["sentences_per_paragraph"])
        sentences: list[str] = []
        for position in range(n_sentences):
            template = _pick_unused(bank, used, rng)
            sentence = template.format(
                activity=activity,
                abstract_noun=rng.choice(_ABSTRACT_NOUNS),
                field=field,
            )
            if position > 0 and rng.random() < persona["connective_rate"]:
                sentence = _prepend_connective(sentence, rng.choice(CONNECTIVES))
            sentence = _fit_length(sentence, persona["target_len"], rng)
            sentences.append(sentence)
            if rng.random() < persona["tricolon_rate"] and position == n_sentences - 2:
                sentences.append(rng.choice(_TRICOLONS))
        # Machine output tends to keep paragraph sizes close to each other.
        paragraphs.append(" ".join(sentences))
        if move_index == 0 and rng.random() < 0.3:
            paragraphs[-1] += " This realisation would shape the years that followed."

    if target_words:
        paragraphs = _fit_document_length(paragraphs, target_words)
    return "\n\n".join(paragraphs)


def _fit_document_length(paragraphs: list[str], target_words: int) -> list[str]:
    """Trim whole sentences (never mid-sentence) until close to ``target_words``.

    Trimming from the end of body paragraphs keeps the opening hook and the
    closing paragraph intact, which is what an over-long generation looks like
    after a length constraint is applied.
    """
    def total(ps: list[str]) -> int:
        return sum(len(p.split()) for p in ps)

    guard = 0
    while total(paragraphs) > target_words and guard < 200:
        guard += 1
        # Prefer trimming the longest body paragraph; fall back to any paragraph.
        candidates = list(range(1, max(1, len(paragraphs) - 1))) or [0]
        index = max(candidates, key=lambda i: len(paragraphs[i].split()))
        sentences = _split_sentences(paragraphs[index])
        if len(sentences) > 1:
            paragraphs[index] = " ".join(sentences[:-1])
        elif len(paragraphs) > 2:
            paragraphs.pop(index)
        else:
            break
    return [p for p in paragraphs if p.strip()]


def _pick_unused(bank: tuple[str, ...], used: set[str], rng: random.Random) -> str:
    available = [t for t in bank if t not in used]
    if not available:
        used.clear()
        available = list(bank)
    choice = rng.choice(available)
    used.add(choice)
    return choice


_EXPANSIONS = (
    " in a manner that proved unexpectedly instructive",
    " throughout the entirety of that period",
    " in ways I had not initially anticipated",
    " within the context of my broader academic interests",
    " with a degree of rigour I had not previously applied",
    " despite the considerable demands on my time",
)


def _fit_length(sentence: str, target: tuple[int, int], rng: random.Random) -> str:
    """Pad a sentence toward the persona's target length.

    This is what produces the low-variance sentence-length signature: the
    generator actively regresses toward a target band, which is exactly the
    behaviour burstiness features are designed to notice.
    """
    low, high = target
    words = sentence.split()
    guard = 0
    while len(words) < low and guard < 3:
        sentence = sentence.rstrip(".") + rng.choice(_EXPANSIONS) + "."
        words = sentence.split()
        guard += 1
    if len(words) > high + 6:
        keep = words[: high + 4]
        sentence = " ".join(keep).rstrip(",;") + "."
    return sentence


# =========================================================================== #
# 3. Rule-based AI_POLISHED proxy
# =========================================================================== #
_VOCAB_UPGRADES = (
    (r"\ba lot of\b", "a considerable amount of"),
    (r"\blots of\b", "numerous"),
    (r"\bgot\b", "obtained"),
    (r"\bgets\b", "obtains"),
    (r"\bbad\b", "suboptimal"),
    (r"\bgood\b", "commendable"),
    (r"\bbig\b", "substantial"),
    (r"\bstuff\b", "material"),
    (r"\bthings\b", "elements"),
    (r"\bkid\b", "child"),
    (r"\bkids\b", "children"),
    (r"\bguy\b", "individual"),
    (r"\bshowed up\b", "attended"),
    (r"\bfigured out\b", "determined"),
    (r"\bfound out\b", "discovered"),
    (r"\bmade me\b", "compelled me to"),
    (r"\bhelped me\b", "enabled me to"),
    (r"\bstarted\b", "commenced"),
    (r"\bended up\b", "ultimately came to be"),
    (r"\bpretty\b", "considerably"),
    (r"\bvery\b", "exceptionally"),
    (r"\bhard\b", "demanding"),
    (r"\btried\b", "endeavoured"),
    (r"\btook\b", "required"),
)

_HEDGE_REMOVALS = (
    r"\bI think that\b",
    r"\bI think\b",
    r"\bI guess\b",
    r"\bmaybe\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bpretty much\b",
    r"\bhonestly,?\b",
    r"\bbasically,?\b",
    r"\bI mean,?\b",
    r"\bAnyway,?\b",
    r"\byou know,?\b",
    r"\bactually\b",
    r"\bjust\b",
)

_EXPAND_CONTRACTIONS = (
    (r"\bdon't\b", "do not"),
    (r"\bdidn't\b", "did not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bcan't\b", "cannot"),
    (r"\bcouldn't\b", "could not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bisn't\b", "is not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),
    (r"\bit's\b", "it is"),
    (r"\bI'm\b", "I am"),
    (r"\bI've\b", "I have"),
    (r"\bI'd\b", "I would"),
    (r"\bthat's\b", "that is"),
    (r"\bthere's\b", "there is"),
    (r"\bhaven't\b", "have not"),
    (r"\bhadn't\b", "had not"),
)


def polish_text(text: str, transform: str, rng: random.Random) -> str:
    """Apply a bounded, rule-based "polish" to real text."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    if transform == "partial_paragraph":
        # The localised case: edit exactly one paragraph and leave the rest
        # byte-identical. This is what the style-shift detector must catch.
        target = 1 if len(paragraphs) > 1 else 0
        paragraphs[target] = _heavy_polish(paragraphs[target], rng)
        return "\n\n".join(paragraphs)

    operations = {
        "grammar_only": ("fix_spacing",),
        "clarity": ("expand_contractions", "remove_hedges", "fix_spacing"),
        "vocabulary": ("upgrade_vocabulary", "fix_spacing"),
        "restructure": ("expand_contractions", "add_connectives", "merge_short", "fix_spacing"),
        "formalize": (
            "expand_contractions",
            "remove_hedges",
            "upgrade_vocabulary",
            "add_connectives",
            "strip_initial_conjunctions",
            "normalise_dashes",
            "fix_spacing",
        ),
        "shorten": ("remove_hedges", "drop_trailing_sentence", "fix_spacing"),
    }.get(transform, ("fix_spacing",))

    return "\n\n".join(_apply_polish_ops(p, operations, rng) for p in paragraphs)


def _heavy_polish(paragraph: str, rng: random.Random) -> str:
    return _apply_polish_ops(
        paragraph,
        (
            "expand_contractions",
            "remove_hedges",
            "upgrade_vocabulary",
            "add_connectives",
            "strip_initial_conjunctions",
            "merge_short",
            "normalise_dashes",
            "fix_spacing",
        ),
        rng,
    )


def _apply_polish_ops(paragraph: str, operations: tuple[str, ...], rng: random.Random) -> str:
    out = paragraph
    for op in operations:
        if op == "expand_contractions":
            for pattern, replacement in _EXPAND_CONTRACTIONS:
                out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        elif op == "remove_hedges":
            for pattern in _HEDGE_REMOVALS:
                if rng.random() < 0.8:
                    out = re.sub(pattern, "", out, flags=re.IGNORECASE)
        elif op == "upgrade_vocabulary":
            for pattern, replacement in _VOCAB_UPGRADES:
                if rng.random() < 0.65:
                    out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        elif op == "add_connectives":
            out = _add_connectives(out, rng)
        elif op == "strip_initial_conjunctions":
            out = re.sub(r"(?<=[.!?]\s)(But|So|And|Then)\s+", "", out)
            out = re.sub(r"^(But|So|And|Then)\s+", "", out)
        elif op == "merge_short":
            out = _merge_short_sentences(out, rng)
        elif op == "drop_trailing_sentence":
            sentences = _split_sentences(out)
            if len(sentences) > 2:
                out = " ".join(sentences[:-1])
        elif op == "normalise_dashes":
            out = out.replace(" - ", ", ").replace("-", ", ").replace(" – ", ", ")
        elif op == "fix_spacing":
            out = re.sub(r"\s{2,}", " ", out)
            out = re.sub(r"\s+([,.;:!?])", r"\1", out)
            out = re.sub(r",\s*,", ",", out)
            out = re.sub(r"\.\s*\.", ".", out)
            out = out.strip()
            if out and out[0].islower():
                out = out[0].upper() + out[1:]
    return out


def _add_connectives(paragraph: str, rng: random.Random) -> str:
    sentences = _split_sentences(paragraph)
    out: list[str] = []
    for i, sentence in enumerate(sentences):
        if i > 0 and rng.random() < 0.45 and sentence[:1].isupper():
            sentence = _prepend_connective(sentence, rng.choice(CONNECTIVES))
        out.append(sentence)
    return " ".join(out)


def _merge_short_sentences(paragraph: str, rng: random.Random) -> str:
    """Join adjacent short sentences, which regularises sentence length.

    This is the single most consequential thing "improve the flow" edits do to
    the burstiness signature, so it is modelled explicitly.
    """
    sentences = _split_sentences(paragraph)
    out: list[str] = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        if (
            i + 1 < len(sentences)
            and len(current.split()) <= 9
            and len(sentences[i + 1].split()) <= 14
            and rng.random() < 0.7
        ):
            nxt = sentences[i + 1]
            joiner = rng.choice([", and ", ", which meant that ", "; ", ", although "])
            merged = current.rstrip(".!?") + joiner + nxt[0].lower() + nxt[1:]
            out.append(merged)
            i += 2
            continue
        out.append(current)
        i += 1
    return " ".join(out)


# =========================================================================== #
# Orchestration
# =========================================================================== #
def build(
    *,
    human_variants_per_seed: int = 6,
    ai_per_topic: int = 12,
    polish_transforms_per_seed: int = 6,
    held_out_topic_count: int = 4,
    held_out_model_count: int = 24,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    data_dir = data_dir or settings.data_path
    paths = dataset_paths(data_dir)
    # No shared RNG: every generator below derives its own seeded Random from a
    # stable key (seed id, topic, persona, transform), so adding or removing one
    # item cannot shift the output of any other.
    seeds = load_seeds(data_dir)
    log_event(logger, "bootstrap.seeds_loaded", seeds=len(seeds))

    # ---------------------------------------------------------------- human
    human_samples: list[Sample] = []
    for seed in seeds:
        seed_rng = random.Random(f"{GLOBAL_SEED}:{seed['seed_id']}")
        for tag, text in human_variants(seed, human_variants_per_seed, seed_rng):
            human_samples.append(
                Sample(
                    record_id=f"{seed['seed_id']}-{tag}",
                    label="human",
                    text=text,
                    group_id=f"seed:{seed['seed_id']}",
                    source="seed_authored" if tag == "original" else "seed_variant",
                    topic=seed["topic"],
                    length_band=seed.get("length_band", "medium"),
                    voice=seed.get("voice", "unspecified"),
                    l2_english=bool(seed.get("l2_english", False)),
                    strategy=tag,
                    notes=(
                        "Hand-authored seed essay written for this repository as a "
                        "bootstrap proxy for authentic student writing."
                        if tag == "original"
                        else f"Seed essay with human-style editing noise ({tag})."
                    ),
                )
            )

    # --------------------------------------------------------- ai_generated
    # Length is matched to the human class. Without this the two classes differ
    # in mean length by ~200 words, and a classifier would learn "long = machine"
    # - a shortcut that says nothing about writing and would collapse on real
    # essays. Targets are drawn from the empirical human word-count distribution.
    human_word_counts = sorted(s.n_words for s in human_samples)
    length_bands = _length_bands(human_word_counts)
    log_event(
        logger,
        "bootstrap.length_matching",
        human_min=human_word_counts[0],
        human_median=human_word_counts[len(human_word_counts) // 2],
        human_max=human_word_counts[-1],
        **{f"band_{k}": f"{v[0]}-{v[1]}" for k, v in length_bands.items()},
    )

    ai_samples: list[Sample] = []

    def _emit_ai(topic: str, persona: str, index: int) -> None:
        topic_rng = random.Random(f"{GLOBAL_SEED}:{topic}:{persona}:{index}")
        band = ("short", "medium", "long")[index % 3]
        low, high = length_bands[band]
        target_words = topic_rng.randint(low, high)
        text = generate_procedural_essay(
            topic, persona, topic_rng, target_words=target_words
        )
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:32].strip("-")
        ai_samples.append(
            Sample(
                record_id=f"gen-{slug}-{persona}-{index}",
                label="ai_generated",
                text=text,
                group_id=f"gen:{slug}:{persona}:{index}",
                source="bootstrap_procedural",
                topic=topic,
                length_band=band,
                voice="machine_register",
                model=persona,
                strategy="procedural_template",
                temperature=None,
                notes=(
                    "Offline procedural generator. NOT real model output - see "
                    "docs/dataset.md for what this does and does not represent."
                ),
            )
        )

    # Main body: training topics x the four training personas.
    for topic in TRAIN_TOPICS:
        for i in range(ai_per_topic):
            _emit_ai(topic, TRAINING_PERSONAS[i % len(TRAINING_PERSONAS)], i)

    # Held-out topics get a smaller slice: every one of these is forced into the
    # test split, and an over-large slice would push the test share well past 15%.
    for topic in HELD_OUT_TOPICS:
        for i in range(held_out_topic_count):
            _emit_ai(topic, TRAINING_PERSONAS[i % len(TRAINING_PERSONAS)], i)

    # Held-out persona on *training* topics, so the cross-model test isolates the
    # generator instead of confounding it with topic novelty.
    for i in range(held_out_model_count):
        _emit_ai(TRAIN_TOPICS[i % len(TRAIN_TOPICS)], HELD_OUT_PERSONA, i)

    # ---------------------------------------------------------- ai_polished
    transform_names = list(OFFLINE_POLISH_TRANSFORMS)[:polish_transforms_per_seed]
    polished_samples: list[Sample] = []
    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        original = "\n\n".join(seed["paragraphs"])
        for transform in transform_names:
            polish_rng = random.Random(f"{GLOBAL_SEED}:{seed['seed_id']}:{transform}")
            text = polish_text(original, transform, polish_rng)
            if _similarity(original, text) > 0.995:
                # A transform that changed nothing is not a valid AI_POLISHED
                # sample; dropping it keeps the class honest.
                continue
            record_id = f"{seed['seed_id']}-polish-{transform}"
            polished_samples.append(
                Sample(
                    record_id=record_id,
                    label="ai_polished",
                    text=text,
                    # Same group as the human original: an essay and its polished
                    # variant must never straddle a train/test boundary.
                    group_id=f"seed:{seed['seed_id']}",
                    source="bootstrap_rule_polish",
                    topic=seed["topic"],
                    length_band=seed.get("length_band", "medium"),
                    voice=seed.get("voice", "unspecified"),
                    l2_english=bool(seed.get("l2_english", False)),
                    model="rule_based_polisher",
                    strategy=transform,
                    parent_id=f"{seed['seed_id']}-original",
                    notes=(
                        f"Rule-based '{transform}' edit of a real seed essay "
                        f"(expected change: {POLISH_TRANSFORMS[transform]['expected_change']})."
                    ),
                )
            )
            pairs.append(
                {
                    "pair_id": record_id,
                    "seed_id": seed["seed_id"],
                    "transform": transform,
                    "expected_change": POLISH_TRANSFORMS[transform]["expected_change"],
                    "original": original,
                    "polished": text,
                    "similarity": round(_similarity(original, text), 4),
                    "word_delta": len(text.split()) - len(original.split()),
                }
            )

    write_jsonl(paths["human"], human_samples)
    write_jsonl(paths["ai_generated"], ai_samples)
    write_jsonl(paths["ai_polished"], polished_samples)

    paths["pairs"].parent.mkdir(parents=True, exist_ok=True)
    with paths["pairs"].open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    summary = {
        "human": len(human_samples),
        "ai_generated": len(ai_samples),
        "ai_polished": len(polished_samples),
        "polish_pairs": len(pairs),
        "seeds": len(seeds),
        "data_regime": "bootstrap",
    }
    log_event(logger, "bootstrap.complete", **summary)
    return summary


def _similarity(a: str, b: str) -> float:
    """Cheap word-level Jaccard, used only to reject no-op transforms."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _length_bands(sorted_counts: list[int]) -> dict[str, tuple[int, int]]:
    """Split the human word-count distribution into three matched bands."""
    if not sorted_counts:
        return {"short": (200, 260), "medium": (260, 340), "long": (340, 420)}

    def pct(q: float) -> int:
        return int(sorted_counts[min(len(sorted_counts) - 1, int(q * len(sorted_counts)))])

    p0, p33, p66, p100 = pct(0.0), pct(0.33), pct(0.66), pct(0.999)
    return {
        "short": (p0, max(p0 + 10, p33)),
        "medium": (p33, max(p33 + 10, p66)),
        "long": (p66, max(p66 + 10, p100)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the offline bootstrap corpus.")
    parser.add_argument("--human-variants", type=int, default=6)
    parser.add_argument("--ai-per-topic", type=int, default=12)
    parser.add_argument("--polish-transforms", type=int, default=6)
    parser.add_argument("--held-out-topic-count", type=int, default=4)
    parser.add_argument("--held-out-model-count", type=int, default=24)
    args = parser.parse_args()

    summary = build(
        human_variants_per_seed=args.human_variants,
        ai_per_topic=args.ai_per_topic,
        polish_transforms_per_seed=args.polish_transforms,
        held_out_topic_count=args.held_out_topic_count,
        held_out_model_count=args.held_out_model_count,
    )
    logger = get_logger("ml.bootstrap")
    logger.info(
        "bootstrap corpus written | "
        + " ".join(f"{k}={v}" for k, v in summary.items())
    )


if __name__ == "__main__":
    main()
