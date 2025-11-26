import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Example:
    id: str
    ticket: str
    response: str


@dataclass
class LoadResult:
    examples: list[Example]
    skipped: list[dict]  # {"index": int, "reason": str}


def load_examples(path: Path) -> LoadResult:
    """Load examples from JSON file, validating each entry."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array")

    examples = []
    skipped = []

    for i, item in enumerate(data):
        error = _validate_example(item)
        if error:
            skipped.append({"index": i, "reason": error})
            continue
        examples.append(Example(
            id=item["id"],
            ticket=item["ticket"],
            response=item["response"]
        ))

    return LoadResult(examples=examples, skipped=skipped)


def _validate_example(item: dict) -> str | None:
    """Return error message if invalid, None if valid."""
    if not isinstance(item, dict):
        return "Item is not an object"

    for field in ("id", "ticket", "response"):
        if field not in item:
            return f"Missing '{field}' field"
        if not isinstance(item[field], str):
            return f"'{field}' must be a string"
        if not item[field].strip():
            return f"'{field}' is empty"

    return None
