# Eval Harness Implementation Plan

## Overview

This plan is structured for **parallel agent execution**. Each task is independent and can be worked on simultaneously. Tasks have no dependencies on each other until the final integration.

---

## Task 1: Loader Module

**File:** `src/eval_harness/loader.py`

**Description:** Load and validate JSON input files.

**Implementation:**

```python
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
```

**Verification:**
- Create a test file with valid and invalid examples
- Run: `uv run python -c "from eval_harness.loader import load_examples; from pathlib import Path; print(load_examples(Path('data/examples.json')))"`

---

## Task 2: Evaluator Module

**File:** `src/eval_harness/evaluator.py`

**Description:** Score examples using OpenAI as judge.

**Implementation:**

```python
import json
from dataclasses import dataclass
from openai import OpenAI

from .loader import Example


@dataclass
class Score:
    score: int
    reasoning: str


@dataclass
class EvalResult:
    id: str
    relevance: Score
    tone: Score


RELEVANCE_RUBRIC = """
Rate the relevance of this support bot response on a 1-5 scale:
- 5: Directly addresses the ticket issue, technically accurate, no irrelevant information
- 4: Addresses the issue correctly, minor omissions or slightly tangential details
- 3: Partially relevant, misses key aspects or includes notable off-topic content
- 2: Loosely related but doesn't solve the actual problem
- 1: Completely off-topic or technically incorrect
"""

TONE_RUBRIC = """
Rate the tone of this support bot response on a 1-5 scale:
- 5: Professional and concise - clear, direct, no fluff
- 4: Mostly professional/concise, minor verbosity or slight tone issues
- 3: Acceptable but noticeably verbose, overly casual, or slightly robotic
- 2: Too informal, too wordy, or awkwardly phrased
- 1: Unprofessional, confusing, or inappropriate tone
"""


def evaluate_example(client: OpenAI, example: Example, model: str = "gpt-4o-mini") -> EvalResult:
    """Evaluate a single example for relevance and tone."""
    relevance = _score_dimension(client, example, "relevance", RELEVANCE_RUBRIC, model)
    tone = _score_dimension(client, example, "tone", TONE_RUBRIC, model)

    return EvalResult(id=example.id, relevance=relevance, tone=tone)


def _score_dimension(
    client: OpenAI,
    example: Example,
    dimension: str,
    rubric: str,
    model: str
) -> Score:
    """Score a single dimension using the LLM judge."""
    prompt = f"""You are evaluating a support bot response for {dimension}.

TICKET:
{example.ticket}

RESPONSE:
{example.response}

{rubric}

Respond with JSON: {{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return Score(score=result["score"], reasoning=result["reasoning"])
```

**Verification:**
- Set OPENAI_API_KEY environment variable
- Run: `uv run python -c "from openai import OpenAI; from eval_harness.evaluator import evaluate_example; from eval_harness.loader import Example; c = OpenAI(); e = Example('test', 'test ticket', 'test response'); print(evaluate_example(c, e))"`

---

## Task 3: Reporter Module

**File:** `src/eval_harness/reporter.py`

**Description:** Write evaluation results to JSON file.

**Implementation:**

```python
import json
from pathlib import Path
from dataclasses import asdict

from .evaluator import EvalResult


def write_results(
    results: list[EvalResult],
    skipped: list[dict],
    output_path: Path
) -> None:
    """Write evaluation results to JSON file."""
    output = {
        "results": [_result_to_dict(r) for r in results],
        "skipped": skipped
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def _result_to_dict(result: EvalResult) -> dict:
    """Convert EvalResult to dictionary for JSON serialization."""
    return {
        "id": result.id,
        "relevance": {
            "score": result.relevance.score,
            "reasoning": result.relevance.reasoning
        },
        "tone": {
            "score": result.tone.score,
            "reasoning": result.tone.reasoning
        }
    }
```

**Verification:**
- Run: `uv run python -c "from pathlib import Path; from eval_harness.reporter import write_results; from eval_harness.evaluator import EvalResult, Score; r = EvalResult('test', Score(5, 'good'), Score(4, 'ok')); write_results([r], [], Path('results/test.json')); print(open('results/test.json').read())"`

---

## Task 4: CLI Main Module

**File:** `src/eval_harness/main.py`

**Description:** CLI entry point that ties everything together.

**Implementation:**

```python
import argparse
import sys
from pathlib import Path

from openai import OpenAI

from .loader import load_examples
from .evaluator import evaluate_example
from .reporter import write_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate support bot responses for relevance and tone"
    )
    parser.add_argument("input", type=Path, help="Input JSON file with examples")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/output.json"),
                        help="Output JSON file (default: results/output.json)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Loading examples from {args.input}...")
    load_result = load_examples(args.input)

    if load_result.skipped:
        print(f"Warning: Skipped {len(load_result.skipped)} invalid examples")

    print(f"Evaluating {len(load_result.examples)} examples with {args.model}...")

    client = OpenAI()
    results = []

    for i, example in enumerate(load_result.examples, 1):
        print(f"  [{i}/{len(load_result.examples)}] {example.id}")
        result = evaluate_example(client, example, args.model)
        results.append(result)

    print(f"Writing results to {args.output}...")
    write_results(results, load_result.skipped, args.output)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Verification:**
- Run: `uv run python -m eval_harness.main data/examples.json --output results/output.json`
- Check: `cat results/output.json`

---

## Agent Dispatch Instructions

Run these 4 agents in parallel:

```
Agent 1: "Implement loader.py as specified in Task 1 of docs/plans/2025-11-26-implementation-plan.md. Follow the implementation exactly. Run the verification command to confirm it works."

Agent 2: "Implement evaluator.py as specified in Task 2 of docs/plans/2025-11-26-implementation-plan.md. Follow the implementation exactly. Run the verification command to confirm it works (requires OPENAI_API_KEY)."

Agent 3: "Implement reporter.py as specified in Task 3 of docs/plans/2025-11-26-implementation-plan.md. Follow the implementation exactly. Run the verification command to confirm it works."

Agent 4: "Implement main.py as specified in Task 4 of docs/plans/2025-11-26-implementation-plan.md. Follow the implementation exactly."
```

After all agents complete, run the full verification:
```bash
OPENAI_API_KEY=<key> uv run python -m eval_harness.main data/examples.json
```
