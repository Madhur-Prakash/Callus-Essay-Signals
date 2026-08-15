"""Word lists used by the stylometric layer.

These are fixed, version-controlled lexicons rather than model outputs, so a
feature value can always be traced back to a concrete rule. ``LEXICON_VERSION``
is recorded in the model metadata: changing a list changes the feature space.
"""

from __future__ import annotations

LEXICON_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Function words. The classic authorship-attribution signal: high-frequency,
# topic-independent words whose relative rates are stable per author.
# --------------------------------------------------------------------------- #
FUNCTION_WORDS: tuple[str, ...] = (
    "a", "about", "above", "after", "again", "against", "all", "almost", "also",
    "although", "always", "am", "among", "an", "and", "another", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing",
    "down", "during", "each", "either", "else", "enough", "even", "ever",
    "every", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "however", "i", "if", "in", "instead", "into", "is", "it", "its", "itself",
    "just", "least", "less", "like", "many", "may", "me", "might", "mine",
    "more", "most", "much", "must", "my", "myself", "neither", "never", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "others", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "perhaps", "quite", "rather", "same", "seem", "several", "shall", "she",
    "should", "since", "so", "some", "still", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "though", "through", "thus", "to", "too", "toward",
    "under", "until", "up", "upon", "us", "very", "was", "we", "were", "what",
    "when", "where", "whether", "which", "while", "who", "whom", "whose",
    "why", "will", "with", "within", "without", "would", "yet", "you", "your",
    "yours", "yourself",
)

STOPWORDS: frozenset[str] = frozenset(FUNCTION_WORDS)

# Words that make an n-gram uninformative *for display purposes*. Kept separate
# from FUNCTION_WORDS on purpose: FUNCTION_WORDS defines a feature vector whose
# width is baked into every trained artifact, so it must not drift. This set only
# decides whether a repeated phrase is worth showing a user — "is one of the"
# repeating three times is not evidence of anything — and can grow freely.
UNINFORMATIVE_NGRAM_WORDS: frozenset[str] = STOPWORDS | frozenset(
    {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "first", "last", "next", "thing", "things", "way", "ways", "time",
        "times", "lot", "kind", "sort", "part", "parts", "item", "items", "got",
        "get", "go", "went", "come", "came", "make", "made", "say", "said",
        "know", "knew", "think", "thought", "want", "wanted", "really", "also",
        "back", "out", "well", "much", "many", "s", "t", "re", "ve", "ll", "d",
        "m",
    }
)

# --------------------------------------------------------------------------- #
# Discourse / transition markers. Instruction-tuned models over-produce explicit
# connectives, especially sentence-initial ones.
# --------------------------------------------------------------------------- #
TRANSITION_WORDS: tuple[str, ...] = (
    "additionally", "alternatively", "besides", "consequently", "conversely",
    "finally", "first", "firstly", "furthermore", "hence", "however", "indeed",
    "instead", "likewise", "meanwhile", "moreover", "nevertheless",
    "nonetheless", "notably", "overall", "second", "secondly", "similarly",
    "specifically", "subsequently", "therefore", "third", "thirdly", "thus",
    "ultimately", "whereas",
)

TRANSITION_PHRASES: tuple[str, ...] = (
    "as a result", "at the same time", "by contrast", "for example",
    "for instance", "in addition", "in conclusion", "in contrast",
    "in essence", "in other words", "in particular", "in summary",
    "in this way", "not only", "on the contrary", "on the other hand",
    "that being said", "to begin with", "to that end", "to this day",
    "what is more", "with that said",
)

# Phrases whose density is conspicuously high in instruction-tuned output.
LLM_REGISTER_PHRASES: tuple[str, ...] = (
    "a testament to", "a tapestry of", "delve into", "embark on",
    "ever-evolving", "far-reaching", "foster a", "fostered a",
    "gain a deeper understanding", "hone my", "honed my", "in today's world",
    "instilled in me", "invaluable", "it is important to note",
    "meaningful impact", "multifaceted", "navigate the complexities",
    "not merely", "pivotal", "profound", "resonated with me",
    "serves as a", "shed light on", "solidified my",
    "testament to my", "transformative journey", "underscore",
    "unwavering", "vibrant tapestry", "which in turn",
)

# --------------------------------------------------------------------------- #
# Hedges, boosters, and other stance markers.
# --------------------------------------------------------------------------- #
HEDGES: tuple[str, ...] = (
    "apparently", "arguably", "assume", "assumed", "basically", "generally",
    "guess", "kind of", "likely", "maybe", "might", "mostly", "perhaps",
    "possibly", "presumably", "probably", "roughly", "seemed", "seems",
    "somehow", "somewhat", "sort of", "suppose", "supposedly", "tend",
    "tended", "typically", "usually",
)

INTENSIFIERS: tuple[str, ...] = (
    "absolutely", "always", "completely", "deeply", "definitely", "entirely",
    "especially", "extremely", "genuinely", "greatly", "highly", "immensely",
    "incredibly", "never", "particularly", "profoundly", "really",
    "remarkably", "significantly", "substantially", "totally", "truly",
    "utterly", "very", "vitally",
)

FIRST_PERSON: frozenset[str] = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)

# Colloquial / conversational markers: comparatively common in unedited human
# drafts, systematically removed by "polish this" style edits.
COLLOQUIAL_MARKERS: tuple[str, ...] = (
    "anyway", "basically", "honestly", "i mean", "kind of", "kinda", "look",
    "okay", "pretty much", "so yeah", "sort of", "stuff", "thing", "things",
    "well", "you know",
)

CONTRACTION_PATTERN = (
    r"\b(?:"
    r"i'm|i've|i'd|i'll|"
    r"you're|you've|you'd|you'll|"
    r"he's|she's|it's|we're|we've|we'd|we'll|they're|they've|they'd|they'll|"
    r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|"
    r"won't|wouldn't|shouldn't|hasn't|haven't|hadn't|that's|there's|"
    r"what's|who's|let's|ain't"
    r")\b"
)

NOMINALIZATION_SUFFIXES: tuple[str, ...] = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship", "ism",
    "ability", "ibility",
)

# --------------------------------------------------------------------------- #
# Fixed tag inventories. Feature vectors must have a stable width, so the tag
# lists are frozen here rather than derived from whatever appears in a document.
# --------------------------------------------------------------------------- #
POS_TAGS: tuple[str, ...] = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)

DEP_LABELS: tuple[str, ...] = (
    "acl", "acomp", "advcl", "advmod", "amod", "appos", "attr", "aux",
    "auxpass", "cc", "ccomp", "compound", "conj", "csubj", "dative", "dep",
    "det", "dobj", "mark", "neg", "nmod", "npadvmod", "nsubj", "nsubjpass",
    "pcomp", "pobj", "poss", "prep", "prt", "punct", "relcl", "xcomp",
)

CLAUSAL_DEPS: frozenset[str] = frozenset(
    {"ccomp", "xcomp", "advcl", "acl", "relcl", "csubj", "csubjpass", "pcomp"}
)

PASSIVE_DEPS: frozenset[str] = frozenset({"nsubjpass", "auxpass", "csubjpass"})
