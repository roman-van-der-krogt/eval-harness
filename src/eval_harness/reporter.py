import json
from pathlib import Path

from .evaluator import EvalResult


def write_results(
    results: list[EvalResult],
    skipped: list[dict],
    failed: list[dict],
    aggregates: dict,
    output_path: Path
) -> None:
    """Write evaluation results to JSON file."""
    output = {
        "results": [_result_to_dict(r) for r in results],
        "skipped": skipped,
        "failed": failed,
        "aggregates": aggregates
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def _result_to_dict(result: EvalResult) -> dict:
    """Convert EvalResult to dictionary for JSON serialization."""
    return {
        "id": result.id,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "relevance": {
            "score": result.relevance.score,
            "reasoning": result.relevance.reasoning
        },
        "tone": {
            "score": result.tone.score,
            "reasoning": result.tone.reasoning
        }
    }
