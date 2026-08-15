"""Prompt inventory for dataset generation.

Deliberate variety, because a dataset built from one prompt teaches a detector to
recognise one prompt. Every generated sample records which strategy, topic,
length band, temperature and model produced it, so the evaluation can slice by
each of those axes.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Topics. Kept disjoint from the human seed topics where possible, and split
# into a "held-out" group used only in the test split so topic generalisation is
# actually measurable.
# --------------------------------------------------------------------------- #
TRAIN_TOPICS: tuple[str, ...] = (
    "a robotics project that failed",
    "working in a family business",
    "caring for an ill relative",
    "losing an important competition",
    "immigrating and learning English",
    "years of a demanding sport",
    "a community volunteering project",
    "learning an instrument late",
    "a minimum-wage summer job",
    "building software nobody used",
    "supporting a sibling with a disability",
    "an ensemble or team performance activity",
    "changing schools repeatedly",
    "repairing things as a hobby",
    "a job in a library or archive",
    "a strategy game and how it taught self-criticism",
)

HELD_OUT_TOPICS: tuple[str, ...] = (
    "an unexpected friendship with an elderly neighbour",
    "a summer spent surveying a wetland",
    "learning to repair a sewing machine",
    "organising a school response to a local flood",
    "discovering an interest in cartography",
    "running the sound desk at a community radio station",
)

# --------------------------------------------------------------------------- #
# Prompting strategies for the AI_GENERATED class.
# --------------------------------------------------------------------------- #
GENERATION_STRATEGIES: dict[str, dict[str, str]] = {
    "plain": {
        "system": "You are a helpful assistant.",
        "user": (
            "Write a college admissions personal statement about {topic}. "
            "It should be about {words} words."
        ),
    },
    "coached": {
        "system": (
            "You are an experienced college admissions essay coach who helps "
            "students write compelling, authentic personal statements."
        ),
        "user": (
            "Write a college admissions essay of about {words} words about {topic}. "
            "Use a clear narrative arc, concrete detail, and end with reflection on "
            "what the student learned."
        ),
    },
    "persona": {
        "system": (
            "You are a 17-year-old high school senior writing your own college "
            "application essay. Write in your own voice."
        ),
        "user": (
            "Write your personal statement about {topic}. Aim for {words} words. "
            "Include specific details from your life."
        ),
    },
    "anti_detection": {
        "system": (
            "You are a skilled writer who produces natural, human-sounding prose."
        ),
        "user": (
            "Write a college admissions essay of about {words} words about {topic}. "
            "Vary your sentence lengths a lot. Use contractions and some informal "
            "phrasing. Avoid words like 'transformative', 'journey', 'furthermore', "
            "'moreover' and 'ultimately'. Do not make it sound polished."
        ),
    },
    "structured": {
        "system": "You are a helpful writing assistant.",
        "user": (
            "Write a {words}-word college admissions essay about {topic}. "
            "Structure it as: hook, background, obstacle, turning point, lesson, "
            "and a forward-looking conclusion. One paragraph each."
        ),
    },
    "sensory": {
        "system": "You are a literary writer with a gift for sensory description.",
        "user": (
            "Write a {words}-word college admissions personal statement about "
            "{topic}. Ground it in physical detail — sound, texture, smell."
        ),
    },
}

# --------------------------------------------------------------------------- #
# Transformations for the AI_POLISHED class. Each takes real human text and
# changes it in a bounded way, which is exactly the realistic threat model.
# --------------------------------------------------------------------------- #
POLISH_TRANSFORMS: dict[str, dict[str, str]] = {
    "grammar_only": {
        "system": "You are a careful copy editor.",
        "user": (
            "Fix only grammar, spelling and punctuation errors in the essay below. "
            "Do not change word choice, sentence structure, voice or length. "
            "Return only the corrected essay.\n\n{text}"
        ),
        "expected_change": "minimal",
    },
    "clarity": {
        "system": "You are an editor who improves clarity without changing voice.",
        "user": (
            "Improve the clarity of the essay below. Keep the author's voice and all "
            "specific details. Return only the edited essay.\n\n{text}"
        ),
        "expected_change": "moderate",
    },
    "vocabulary": {
        "system": "You are an editor who strengthens word choice.",
        "user": (
            "Improve the vocabulary and word choice in the essay below. Keep the "
            "structure and all facts the same. Return only the edited essay.\n\n{text}"
        ),
        "expected_change": "moderate",
    },
    "restructure": {
        "system": "You are a developmental editor.",
        "user": (
            "Restructure the essay below for better flow: reorder ideas, merge or "
            "split sentences, and add transitions where needed. Keep the content and "
            "the first-person voice. Return only the edited essay.\n\n{text}"
        ),
        "expected_change": "heavy",
    },
    "formalize": {
        "system": "You are an editor preparing a piece for formal submission.",
        "user": (
            "Rewrite the essay below in a more formal, polished register suitable for "
            "a competitive college application. Keep every fact. "
            "Return only the edited essay.\n\n{text}"
        ),
        "expected_change": "heavy",
    },
    "shorten": {
        "system": "You are a ruthless but respectful editor.",
        "user": (
            "Tighten the essay below by about 25% without losing any substantive "
            "detail. Return only the edited essay.\n\n{text}"
        ),
        "expected_change": "moderate",
    },
    "partial_paragraph": {
        "system": "You are an editor who improves one section of a draft.",
        "user": (
            "Rewrite ONLY the second paragraph of the essay below to be more vivid "
            "and better written. Leave every other paragraph exactly as it is, "
            "character for character. Return the whole essay.\n\n{text}"
        ),
        "expected_change": "localised",
    },
}

LENGTH_BANDS: dict[str, tuple[int, int]] = {
    "short": (250, 350),
    "medium": (450, 600),
    "long": (700, 900),
}

TEMPERATURES: tuple[float, ...] = (0.5, 0.7, 0.9, 1.1)
TOP_P_VALUES: tuple[float, ...] = (0.9, 1.0)
