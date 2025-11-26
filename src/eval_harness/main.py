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
