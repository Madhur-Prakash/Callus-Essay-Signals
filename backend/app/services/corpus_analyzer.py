"""Cross-essay comparison against the reference corpora.

Detection should not only look at an essay in isolation, so this layer asks:
*how close is this writing to the human reference corpus, and how close to the
machine reference corpus?*

Deliberate design choice: the representations here are **style-bearing but
largely topic-free**.

* character 3-5 grams (``char_wb``) - captures affixes, punctuation habits and
  orthographic rhythm
* POS 1-3 grams - pure syntax, no content words at all
* function-word frequency profile - the classic authorship-attribution signal

Topical word n-grams are intentionally *excluded*. A TF-IDF model over content
words would learn "essays about robotics are human" from whatever topics happen
to dominate the training split, and would then collapse on the held-out-topic
evaluation. Keeping the corpus features topic-free is what makes the topic
generalisation numbers in the evaluation report meaningful.

Centroids are fit on the **training split only** and stored in the model
artifacts. Feature prefix: ``cor_``
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger, log_event
from app.services.lexicons import FUNCTION_WORDS, POS_TAGS

logger = get_logger("app.corpus")

CLASS_ORDER: tuple[str, ...] = ("human", "ai_generated", "ai_polished")

COR_FEATURE_NAMES: tuple[str, ...] = (
    "cor_char_sim_human",
    "cor_char_sim_ai_generated",
    "cor_char_sim_ai_polished",
    "cor_char_sim_delta_ai_human",
    "cor_char_nearest_is_human",
    "cor_pos_sim_human",
    "cor_pos_sim_ai_generated",
    "cor_pos_sim_ai_polished",
    "cor_pos_sim_delta_ai_human",
    "cor_pos_nearest_is_human",
    "cor_funcword_sim_human",
    "cor_funcword_sim_ai_generated",
    "cor_funcword_sim_ai_polished",
    "cor_funcword_sim_delta_ai_human",
    "cor_posdist_js_human",
    "cor_posdist_js_ai_generated",
    "cor_posdist_js_ai_polished",
    "cor_posdist_js_delta_ai_human",
)

_NEUTRAL = dict.fromkeys(COR_FEATURE_NAMES, 0.0)
ARTIFACT_NAME = "corpus_reference.joblib"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1]."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(x * np.log2(x / y)))

    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


class CorpusReference:
    """Fitted reference corpora + similarity scoring."""

    def __init__(self) -> None:
        self.char_vectorizer = None
        self.pos_vectorizer = None
        self.char_centroids: dict[str, np.ndarray] = {}
        self.pos_centroids: dict[str, np.ndarray] = {}
        self.funcword_centroids: dict[str, np.ndarray] = {}
        self.posdist_centroids: dict[str, np.ndarray] = {}
        self.n_documents: dict[str, int] = {}
        self.fitted = False
        self.dataset_version: str | None = None

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        documents: list[dict[str, Any]],
        *,
        dataset_version: str | None = None,
    ) -> CorpusReference:
        """Fit centroids from training documents.

        Each entry needs ``label``, ``text``, ``pos_sequence`` (list[str]),
        ``function_word_profile`` (dict) and ``pos_distribution`` (dict).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [d["text"] for d in documents]
        pos_docs = [" ".join(d.get("pos_sequence") or []) for d in documents]
        labels = [d["label"] for d in documents]

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=20_000,
            sublinear_tf=True,
            lowercase=True,
        )
        char_matrix = self.char_vectorizer.fit_transform(texts)

        if any(pos_docs):
            self.pos_vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"\S+",
                ngram_range=(1, 3),
                min_df=3,
                max_features=8_000,
                sublinear_tf=True,
                lowercase=False,
            )
            pos_matrix = self.pos_vectorizer.fit_transform(pos_docs)
        else:  # spaCy unavailable during training
            self.pos_vectorizer = None
            pos_matrix = None

        for label in CLASS_ORDER:
            rows = [i for i, lab in enumerate(labels) if lab == label]
            self.n_documents[label] = len(rows)
            if not rows:
                continue
            self.char_centroids[label] = np.asarray(
                char_matrix[rows].mean(axis=0)
            ).ravel()
            if pos_matrix is not None:
                self.pos_centroids[label] = np.asarray(
                    pos_matrix[rows].mean(axis=0)
                ).ravel()
            self.funcword_centroids[label] = np.mean(
                [
                    _profile_vector(documents[i].get("function_word_profile") or {}, FUNCTION_WORDS)
                    for i in rows
                ],
                axis=0,
            )
            self.posdist_centroids[label] = np.mean(
                [
                    _profile_vector(documents[i].get("pos_distribution") or {}, POS_TAGS)
                    for i in rows
                ],
                axis=0,
            )

        self.fitted = bool(self.char_centroids)
        self.dataset_version = dataset_version
        log_event(
            logger,
            "corpus_reference.fitted",
            documents=len(documents),
            char_features=int(char_matrix.shape[1]),
            pos_features=int(pos_matrix.shape[1]) if pos_matrix is not None else 0,
            **{f"n_{k}": v for k, v in self.n_documents.items()},
        )
        return self

    # ----------------------------------------------------------- inference
    def score(
        self,
        text: str,
        *,
        pos_sequence: list[str] | None = None,
        function_word_profile: dict[str, float] | None = None,
        pos_distribution: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """``cor_*`` features for a document or a single sentence."""
        if not self.fitted:
            return dict(_NEUTRAL)

        features = dict(_NEUTRAL)

        char_vec = self.char_vectorizer.transform([text])
        char_arr = np.asarray(char_vec.todense()).ravel()
        char_sims = {
            label: _cosine(char_arr, centroid)
            for label, centroid in self.char_centroids.items()
        }

        pos_sims: dict[str, float] = {}
        if self.pos_vectorizer is not None and pos_sequence:
            pos_vec = self.pos_vectorizer.transform([" ".join(pos_sequence)])
            pos_arr = np.asarray(pos_vec.todense()).ravel()
            pos_sims = {
                label: _cosine(pos_arr, centroid)
                for label, centroid in self.pos_centroids.items()
            }

        fw_vec = _profile_vector(function_word_profile or {}, FUNCTION_WORDS)
        fw_sims = {
            label: _cosine(fw_vec, centroid)
            for label, centroid in self.funcword_centroids.items()
        }

        pd_vec = _profile_vector(pos_distribution or {}, POS_TAGS)
        pd_js = {
            label: _js_divergence(pd_vec, centroid)
            for label, centroid in self.posdist_centroids.items()
        }

        for label in CLASS_ORDER:
            features[f"cor_char_sim_{label}"] = char_sims.get(label, 0.0)
            features[f"cor_pos_sim_{label}"] = pos_sims.get(label, 0.0)
            features[f"cor_funcword_sim_{label}"] = fw_sims.get(label, 0.0)
            features[f"cor_posdist_js_{label}"] = pd_js.get(label, 0.0)

        ai_char = max(
            char_sims.get("ai_generated", 0.0), char_sims.get("ai_polished", 0.0)
        )
        features["cor_char_sim_delta_ai_human"] = ai_char - char_sims.get("human", 0.0)
        features["cor_char_nearest_is_human"] = (
            1.0 if char_sims and max(char_sims, key=char_sims.get) == "human" else 0.0
        )

        if pos_sims:
            ai_pos = max(pos_sims.get("ai_generated", 0.0), pos_sims.get("ai_polished", 0.0))
            features["cor_pos_sim_delta_ai_human"] = ai_pos - pos_sims.get("human", 0.0)
            features["cor_pos_nearest_is_human"] = (
                1.0 if max(pos_sims, key=pos_sims.get) == "human" else 0.0
            )

        ai_fw = max(fw_sims.get("ai_generated", 0.0), fw_sims.get("ai_polished", 0.0))
        features["cor_funcword_sim_delta_ai_human"] = ai_fw - fw_sims.get("human", 0.0)

        if pd_js:
            ai_js = min(
                pd_js.get("ai_generated", 1.0), pd_js.get("ai_polished", 1.0)
            )
            features["cor_posdist_js_delta_ai_human"] = ai_js - pd_js.get("human", 1.0)

        return features

    # ---------------------------------------------------------- persistence
    def save(self, directory: Path) -> Path:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ARTIFACT_NAME
        joblib.dump(self, path, compress=3)
        log_event(logger, "corpus_reference.saved", path=path.name)
        return path

    @staticmethod
    def load(directory: Path) -> CorpusReference | None:
        import joblib

        path = Path(directory) / ARTIFACT_NAME
        if not path.exists():
            log_event(logger, "corpus_reference.missing", level="warning", path=str(path.name))
            return None
        try:
            reference: CorpusReference = joblib.load(path)
            log_event(
                logger,
                "corpus_reference.loaded",
                **{f"n_{k}": v for k, v in reference.n_documents.items()},
            )
            return reference
        except Exception as exc:  # pragma: no cover
            log_event(
                logger,
                "corpus_reference.load_failed",
                level="error",
                type=type(exc).__name__,
            )
            return None

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "documents_per_class": dict(self.n_documents),
            "dataset_version": self.dataset_version,
            "representations": [
                "char_wb 3-5 grams (TF-IDF)",
                "POS 1-3 grams (TF-IDF)",
                "function-word frequency profile",
                "POS distribution (Jensen-Shannon)",
            ],
            "topic_words_used": False,
        }


def _profile_vector(profile: dict[str, float], keys) -> np.ndarray:
    return np.array([float(profile.get(k, 0.0)) for k in keys], dtype=np.float64)


def aggregate_document_views(document) -> dict[str, Any]:  # noqa: ANN001
    """Document-level POS sequence / function-word / POS-distribution views."""
    pos_sequence: list[str] = []
    for sentence in document.sentences:
        pos_sequence.extend(sentence.pos_sequence)

    from app.services.stylometry import function_word_profile

    counts = {tag: 0.0 for tag in POS_TAGS}
    for tag in pos_sequence:
        if tag in counts:
            counts[tag] += 1.0
    total = sum(counts.values()) or 1.0
    return {
        "pos_sequence": pos_sequence,
        "function_word_profile": function_word_profile(document.text),
        "pos_distribution": {k: v / total for k, v in counts.items()},
    }


def js_divergence(p: dict[str, float], q: dict[str, float], keys=POS_TAGS) -> float:
    """Public helper reused by the style-shift layer."""
    return _js_divergence(_profile_vector(p, keys), _profile_vector(q, keys))


def cosine(p: dict[str, float], q: dict[str, float], keys) -> float:
    return _cosine(_profile_vector(p, keys), _profile_vector(q, keys))


def entropy_bits(profile: dict[str, float]) -> float:
    total = sum(profile.values())
    if total <= 0:
        return 0.0
    return float(
        -sum((v / total) * math.log2(v / total) for v in profile.values() if v > 0)
    )
