from dataclasses import dataclass
from collections import defaultdict

from .evaluator import EvalResult


@dataclass
class DimensionStats:
    mean: float
    min: int
    max: int


@dataclass
class AggregateStats:
    count: int
    relevance: DimensionStats
    tone: DimensionStats


def compute_aggregates(results: list[EvalResult]) -> dict:
    """Compute aggregates by model, prompt_version, and combination."""
    by_model: dict[str, list[EvalResult]] = defaultdict(list)
    by_prompt: dict[str, list[EvalResult]] = defaultdict(list)
    by_combo: dict[str, list[EvalResult]] = defaultdict(list)

    for r in results:
        by_model[r.model].append(r)
        by_prompt[r.prompt_version].append(r)
        by_combo[f"{r.model}|{r.prompt_version}"].append(r)

    return {
        "by_model": {k: _compute_stats(v) for k, v in by_model.items()},
        "by_prompt_version": {k: _compute_stats(v) for k, v in by_prompt.items()},
        "by_model_and_prompt_version": {k: _compute_stats(v) for k, v in by_combo.items()},
    }


def _compute_stats(results: list[EvalResult]) -> dict:
    """Compute stats for a group of results."""
    relevance_scores = [r.relevance.score for r in results]
    tone_scores = [r.tone.score for r in results]

    return {
        "count": len(results),
        "relevance": {
            "mean": round(sum(relevance_scores) / len(relevance_scores), 2),
            "min": min(relevance_scores),
            "max": max(relevance_scores),
        },
        "tone": {
            "mean": round(sum(tone_scores) / len(tone_scores), 2),
            "min": min(tone_scores),
            "max": max(tone_scores),
        },
    }
