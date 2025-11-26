import argparse
import sys
from pathlib import Path

from openai import OpenAI
from anthropic import Anthropic

from .config import load_config
from .loader import load_examples
from .evaluator import evaluate_example
from .aggregator import compute_aggregates
from .reporter import write_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate support bot responses for relevance and tone"
    )
    parser.add_argument("input", type=Path, help="Input JSON file with examples")
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="Config YAML file with judge mapping")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/output.json"),
                        help="Output JSON file (default: results/output.json)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    print(f"Loading examples from {args.input}...")
    load_result = load_examples(args.input)

    if load_result.skipped:
        print(f"Warning: Skipped {len(load_result.skipped)} invalid examples")

    if not load_result.examples:
        print("No valid examples to evaluate")
        return 1

    print(f"Evaluating {len(load_result.examples)} examples...")

    openai_client = OpenAI()
    anthropic_client = Anthropic()
    results = []

    for i, example in enumerate(load_result.examples, 1):
        print(f"  [{i}/{len(load_result.examples)}] {example.id} (model: {example.model})")
        result = evaluate_example(
            example, config,
            openai_client=openai_client,
            anthropic_client=anthropic_client
        )
        results.append(result)

    print("Computing aggregates...")
    aggregates = compute_aggregates(results)

    print(f"Writing results to {args.output}...")
    write_results(results, load_result.skipped, aggregates, args.output)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
