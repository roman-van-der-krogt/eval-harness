# Eval Harness Design

An eval harness that scores support bot responses on relevance and tone using an LLM judge.

## Context

- **Bot type**: Support bot helping engineers with tickets
- **Scoring dimensions**: Relevance and Tone
- **Scale**: 1-5 (5 = best)
- **Judge**: OpenAI LLM-as-judge
- **Dataset size**: Small (10-50 examples)
- **Output**: Per-example scores with reasoning

## Architecture

```
input.json → Loader → Evaluator (LLM calls) → Reporter → results.json
```

Three components:
1. **Loader** - Reads JSON input, validates examples
2. **Evaluator** - Sends each pair to LLM judge with scoring prompts
3. **Reporter** - Outputs per-example scores with reasoning

## Data Format

### Input (`data/examples.json`)

```json
[
  {
    "id": "ticket-001",
    "ticket": "Jenkins pipeline failing with OOM error on build step...",
    "response": "This typically happens when the heap size is too small..."
  }
]
```

### Output (`results/output.json`)

```json
{
  "results": [
    {
      "id": "ticket-001",
      "relevance": {
        "score": 4,
        "reasoning": "Correctly identifies memory issue and suggests heap adjustment. Minor deduction: didn't ask about container memory limits."
      },
      "tone": {
        "score": 5,
        "reasoning": "Professional and concise. Gets straight to the point without unnecessary preamble."
      }
    }
  ],
  "skipped": [
    {"index": 2, "reason": "Missing 'ticket' field"}
  ]
}
```

## Scoring Rubrics

### Relevance (1-5)

- **5**: Directly addresses the ticket issue, technically accurate, no irrelevant information
- **4**: Addresses the issue correctly, minor omissions or slightly tangential details
- **3**: Partially relevant, misses key aspects or includes notable off-topic content
- **2**: Loosely related but doesn't solve the actual problem
- **1**: Completely off-topic or technically incorrect

### Tone (1-5)

- **5**: Professional and concise - clear, direct, no fluff
- **4**: Mostly professional/concise, minor verbosity or slight tone issues
- **3**: Acceptable but noticeably verbose, overly casual, or slightly robotic
- **2**: Too informal, too wordy, or awkwardly phrased
- **1**: Unprofessional, confusing, or inappropriate tone

## Project Structure

```
metadata/
├── pyproject.toml          # uv project config, dependencies
├── src/
│   └── eval_harness/
│       ├── __init__.py
│       ├── main.py         # CLI entry point
│       ├── loader.py       # Load JSON input, validation
│       ├── evaluator.py    # LLM judge logic + prompts
│       └── reporter.py     # Write results JSON
├── data/
│   └── examples.json       # Eval dataset
└── results/
    └── (output files)
```

## Usage

```bash
uv run python -m eval_harness.main data/examples.json --output results/output.json
```

## Implementation Details

### Evaluator Flow

1. Load all examples from input JSON
2. Validate each example (id, ticket, response must be non-empty strings)
3. Skip invalid examples, log warnings
4. For each valid example, make two LLM calls:
   - Relevance scoring (includes relevance rubric)
   - Tone scoring (includes tone rubric)
5. Parse structured JSON responses (score + reasoning)
6. Collect all results and write to output JSON

### LLM Response Format

Using OpenAI's `response_format: { type: "json_object" }`:

```json
{"score": 4, "reasoning": "..."}
```

### Error Handling

- Retry on rate limits (exponential backoff)
- Log and skip examples that fail after retries
- Report failed examples in output

### Input Validation

Required fields for each example:
- `id` - non-empty string
- `ticket` - non-empty string
- `response` - non-empty string

Invalid examples are skipped with reason logged in output.

## Dependencies

- `openai` - LLM API client
