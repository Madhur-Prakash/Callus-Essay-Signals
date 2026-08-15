"""Causal language model used as a *measuring instrument*.

The model never answers "is this AI?". It answers a much narrower, mechanical
question — "how surprising is each token given the tokens before it?" — and this
module turns those answers into numbers:

* per-token log probability, probability, rank and predictive entropy
* the gap between the token the author used and the model's own top choice
* per-sentence aggregates of all of the above

The classifier downstream is ours. This file is the sensor, not the judge.

Implementation notes
--------------------
* One forward pass per sliding window over the whole essay, not one pass per
  sentence: with ``lm_max_window=512`` and ``lm_stride=384`` every token is
  scored exactly once with up to 128 tokens of left context carried over. Essays
  longer than the model's context window are therefore handled correctly
  instead of being truncated.
* The model is loaded once per process (:class:`LanguageModelService` is a
  singleton) because loading dominates the cost of a short analysis.
* Token→sentence attribution uses the tokenizer's character offsets, so it
  agrees exactly with the offsets the frontend uses for highlighting.

Feature prefix: ``lm_``
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.core.exceptions import ModelUnavailableError
from app.core.logging import get_logger, log_event, quiet_third_party

logger = get_logger("app.lm")

LM_FEATURE_NAMES: tuple[str, ...] = (
    "lm_n_tokens",
    "lm_mean_logprob",
    "lm_median_logprob",
    "lm_std_logprob",
    "lm_min_logprob",
    "lm_max_logprob",
    "lm_p10_logprob",
    "lm_p90_logprob",
    "lm_logprob_iqr",
    "lm_perplexity",
    "lm_log_perplexity",
    "lm_mean_prob",
    "lm_std_prob",
    "lm_prob_variance",
    "lm_mean_entropy",
    "lm_std_entropy",
    "lm_min_entropy",
    "lm_max_entropy",
    "lm_mean_log_rank",
    "lm_median_log_rank",
    "lm_std_log_rank",
    "lm_frac_top1",
    "lm_frac_top10",
    "lm_frac_top100",
    "lm_frac_rank_gt_1000",
    "lm_frac_prob_gt_50",
    "lm_frac_prob_gt_90",
    "lm_frac_prob_lt_5",
    "lm_frac_prob_lt_1",
    "lm_mean_top1_gap",
    "lm_max_top1_gap",
    "lm_mean_normalised_surprisal",
)

_NEUTRAL_LM = dict.fromkeys(LM_FEATURE_NAMES, 0.0)


@dataclass(slots=True)
class TokenScore:
    """One scored token."""

    index: int
    start: int
    end: int
    text: str
    logprob: float
    entropy: float
    rank: int
    top1_logprob: float

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    @property
    def top1_gap(self) -> float:
        """How much more likely the model's preferred token was."""
        return self.top1_logprob - self.logprob


@dataclass(slots=True)
class TokenScoreSet:
    tokens: list[TokenScore] = field(default_factory=list)
    n_windows: int = 0
    scoring_ms: float = 0.0
    truncated: bool = False
    total_tokens: int = 0


class LanguageModelService:
    """Process-wide singleton wrapping the causal LM."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.lm_model_name
        self.device = settings.lm_device
        self.max_window = settings.lm_max_window
        self.stride = min(settings.lm_stride, settings.lm_max_window - 1)
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self.load_time_ms: float | None = None
        self.n_parameters: int | None = None

    # ------------------------------------------------------------- loading
    def load(self) -> None:
        """Load tokenizer + model. Idempotent and thread-safe."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # torch/transformers install their own INFO-level child loggers
                # at import time; re-apply our quieting now that they exist.
                quiet_third_party()

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                model.eval()
                model.to(self.device)
                self._torch = torch
                self._model = model
                self.n_parameters = sum(p.numel() for p in model.parameters())
                self.load_time_ms = round((time.perf_counter() - started) * 1000, 2)
                self._load_error = None
                log_event(
                    logger,
                    "lm.loaded",
                    model=self.model_name,
                    device=self.device,
                    parameters=self.n_parameters,
                    load_ms=self.load_time_ms,
                    max_window=self.max_window,
                    stride=self.stride,
                )
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                log_event(
                    logger,
                    "lm.load_failed",
                    level="error",
                    model=self.model_name,
                    type=type(exc).__name__,
                )
                raise ModelUnavailableError(
                    f"Could not load language model '{self.model_name}'."
                ) from exc

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "role": "feature instrument (token probabilities); does not make the classification",
            "device": self.device,
            "loaded": self.loaded,
            "parameters": self.n_parameters,
            "load_time_ms": self.load_time_ms,
            "max_window_tokens": self.max_window,
            "stride_tokens": self.stride,
            "error": self._load_error,
        }

    # ------------------------------------------------------------- scoring
    def score_text(self, text: str) -> TokenScoreSet:
        """Score every token of ``text`` with a sliding window.

        Returns a :class:`TokenScoreSet`. The very first token of the document is
        scored too: a BOS token is prepended so it has a conditioning context.
        """
        import torch

        self.load()
        tokenizer, model = self._tokenizer, self._model
        assert tokenizer is not None and model is not None  # for type checkers

        started = time.perf_counter()
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids: list[int] = list(encoded["input_ids"])
        offsets: list[tuple[int, int]] = [tuple(o) for o in encoded["offset_mapping"]]

        result = TokenScoreSet(total_tokens=len(ids))
        if not ids:
            return result

        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.eos_token_id
        if bos is None:  # pragma: no cover - every GPT-2 family model has one
            bos = ids[0]

        # Prepend BOS so position 0 of the real text is predictable.
        padded_ids = [bos, *ids]
        # ``padded index i`` predicts ``padded_ids[i+1]`` => real token i.
        window = max(2, self.max_window)
        step = max(1, min(self.stride, window - 1))

        scores: list[TokenScore] = []
        scored_upto = 0  # index into `ids` of the next token needing a score
        start = 0
        n_windows = 0

        while scored_upto < len(ids):
            end = min(start + window, len(padded_ids))
            chunk = padded_ids[start:end]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = model(input_ids).logits[0]  # (chunk_len, vocab)

            # Local position j predicts chunk[j+1], i.e. real token (start + j).
            first_real = scored_upto
            local_start = first_real - start
            if local_start < 0:
                local_start = 0
            local_end = len(chunk) - 1  # last position with a next-token target
            if local_end > local_start:
                self._score_positions(
                    logits=logits,
                    chunk=chunk,
                    local_start=local_start,
                    local_end=local_end,
                    global_offset=start,
                    ids=ids,
                    offsets=offsets,
                    text=text,
                    out=scores,
                )
                scored_upto = start + local_end
            n_windows += 1

            if end >= len(padded_ids):
                break
            start += step
            if n_windows > 4096:  # pragma: no cover - runaway guard
                result.truncated = True
                break

        result.tokens = scores
        result.n_windows = n_windows
        result.scoring_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event(
            logger,
            "lm.scored",
            tokens=len(scores),
            windows=n_windows,
            duration_ms=result.scoring_ms,
        )
        return result

    def _score_positions(
        self,
        *,
        logits,  # noqa: ANN001 - torch.Tensor
        chunk: list[int],
        local_start: int,
        local_end: int,
        global_offset: int,
        ids: list[int],
        offsets: list[tuple[int, int]],
        text: str,
        out: list[TokenScore],
    ) -> None:
        """Compute log-prob / entropy / rank for a slice of window positions.

        Positions are processed in sub-batches: full-vocabulary entropy over 512
        positions x 50k vocab would allocate ~100 MB, which is wasteful on CPU.
        """
        import torch

        batch = 64
        for begin in range(local_start, local_end, batch):
            stop = min(begin + batch, local_end)
            sub = logits[begin:stop].float()
            log_probs = torch.log_softmax(sub, dim=-1)
            targets = torch.tensor(
                chunk[begin + 1 : stop + 1], dtype=torch.long, device=log_probs.device
            )
            target_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            top1_lp, _ = log_probs.max(dim=-1)
            # Predictive entropy of the full next-token distribution.
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            # Rank of the observed token (1 = the model's own top choice).
            rank = (sub > sub.gather(-1, targets.unsqueeze(-1))).sum(dim=-1) + 1

            for k in range(stop - begin):
                real_index = global_offset + begin + k
                if real_index >= len(ids):
                    break
                start_char, end_char = offsets[real_index]
                out.append(
                    TokenScore(
                        index=real_index,
                        start=int(start_char),
                        end=int(end_char),
                        text=text[start_char:end_char],
                        logprob=float(target_lp[k]),
                        entropy=float(entropy[k]),
                        rank=int(rank[k]),
                        top1_logprob=float(top1_lp[k]),
                    )
                )


lm_service = LanguageModelService()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else float((s[mid - 1] + s[mid]) / 2)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q / 100 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def aggregate_tokens(tokens: list[TokenScore]) -> dict[str, float]:
    """Turn a list of scored tokens into the ``lm_*`` feature block."""
    if not tokens:
        return dict(_NEUTRAL_LM)

    logprobs = [t.logprob for t in tokens]
    probs = [t.prob for t in tokens]
    entropies = [t.entropy for t in tokens]
    ranks = [t.rank for t in tokens]
    log_ranks = [math.log(r) for r in ranks]
    gaps = [t.top1_gap for t in tokens]
    n = len(tokens)

    mean_lp = _mean(logprobs)
    # Perplexity = exp(mean negative log-likelihood). Clamped so a single very
    # surprising token in a two-token sentence cannot produce an inf that
    # poisons downstream standardisation.
    log_ppl = min(-mean_lp, 20.0)
    prob_std = _std(probs)

    return {
        "lm_n_tokens": float(n),
        "lm_mean_logprob": mean_lp,
        "lm_median_logprob": _median(logprobs),
        "lm_std_logprob": _std(logprobs),
        "lm_min_logprob": float(min(logprobs)),
        "lm_max_logprob": float(max(logprobs)),
        "lm_p10_logprob": _percentile(logprobs, 10),
        "lm_p90_logprob": _percentile(logprobs, 90),
        "lm_logprob_iqr": _percentile(logprobs, 75) - _percentile(logprobs, 25),
        "lm_perplexity": float(math.exp(log_ppl)),
        "lm_log_perplexity": float(log_ppl),
        "lm_mean_prob": _mean(probs),
        "lm_std_prob": prob_std,
        "lm_prob_variance": prob_std**2,
        "lm_mean_entropy": _mean(entropies),
        "lm_std_entropy": _std(entropies),
        "lm_min_entropy": float(min(entropies)),
        "lm_max_entropy": float(max(entropies)),
        "lm_mean_log_rank": _mean(log_ranks),
        "lm_median_log_rank": _median(log_ranks),
        "lm_std_log_rank": _std(log_ranks),
        "lm_frac_top1": sum(1 for r in ranks if r == 1) / n,
        "lm_frac_top10": sum(1 for r in ranks if r <= 10) / n,
        "lm_frac_top100": sum(1 for r in ranks if r <= 100) / n,
        "lm_frac_rank_gt_1000": sum(1 for r in ranks if r > 1000) / n,
        "lm_frac_prob_gt_50": sum(1 for p in probs if p > 0.5) / n,
        "lm_frac_prob_gt_90": sum(1 for p in probs if p > 0.9) / n,
        "lm_frac_prob_lt_5": sum(1 for p in probs if p < 0.05) / n,
        "lm_frac_prob_lt_1": sum(1 for p in probs if p < 0.01) / n,
        "lm_mean_top1_gap": _mean(gaps),
        "lm_max_top1_gap": float(max(gaps)),
        # Surprisal normalised by the distribution's own entropy: "was this token
        # unexpected *relative to how uncertain the model was here?*"
        "lm_mean_normalised_surprisal": _mean(
            [(-t.logprob) / t.entropy if t.entropy > 1e-6 else 0.0 for t in tokens]
        ),
    }


def assign_to_sentences(document, score_set: TokenScoreSet) -> None:  # noqa: ANN001
    """Attach ``lm_*`` features to each sentence via character offsets."""
    buckets: list[list[TokenScore]] = [[] for _ in document.sentences]
    bounds = [(s.start, s.end) for s in document.sentences]
    if not bounds:
        return

    cursor = 0
    for token in score_set.tokens:
        if token.end <= token.start:  # zero-width piece
            continue
        # A GPT-2 token usually carries its leading space (" have"), so anchor on
        # the first non-whitespace character; otherwise the opening token of every
        # sentence would fall into the preceding sentence's span.
        anchor = token.start + (len(token.text) - len(token.text.lstrip()))
        if anchor >= token.end:
            continue  # pure whitespace
        # Sentences are ordered, so advance a cursor instead of searching.
        while cursor < len(bounds) - 1 and anchor >= bounds[cursor][1]:
            cursor += 1
        buckets[cursor].append(token)

    doc_tokens = score_set.tokens
    doc_aggregate = aggregate_tokens(doc_tokens)
    for sentence, tokens in zip(document.sentences, buckets, strict=False):
        if tokens:
            sentence.lm = aggregate_tokens(tokens)
        else:
            # Extremely short sentence that got no scored token (e.g. "Yes.").
            # Fall back to the document aggregate and mark the token count as 0
            # so the classifier can see it was imputed.
            sentence.lm = {**doc_aggregate, "lm_n_tokens": 0.0}


def token_evidence(
    tokens: list[TokenScore], *, top_k: int = 6, most_predictable: bool = True
) -> list[dict[str, Any]]:
    """The most / least predictable tokens, for the evidence panel.

    Only tokens that carry visible text are returned, and each entry includes the
    raw probability so the UI never has to invent a number.
    """
    candidates = [t for t in tokens if t.text.strip() and len(t.text.strip()) > 1]
    if not candidates:
        candidates = list(tokens)
    candidates.sort(key=lambda t: t.logprob, reverse=most_predictable)
    return [
        {
            "token": t.text.strip(),
            "probability": round(t.prob, 5),
            "logprob": round(t.logprob, 4),
            "rank": t.rank,
            "was_model_top_choice": t.rank == 1,
        }
        for t in candidates[:top_k]
    ]
