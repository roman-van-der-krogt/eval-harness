# Eval Harness Design

An eval harness that scores support bot responses on relevance and tone using an LLM judge.

## Context

- **Bot type**: Support bot helping engineers with tickets
- **Scoring dimensions**: Relevance and Tone
- **Scale**: 1-5 (5 = best)
- **Judge**: Cross-provider LLM-as-judge (OpenAI responses judged by Anthropic, and vice versa)
- **Dataset size**: Small (10-50 examples)
- **Output**: Per-example scores with reasoning, aggregated by model and prompt version

## Architecture

```
config.yaml + input.json → Loader → Evaluator (LLM calls) → Reporter → results.json
```

Four components:
1. **Config** - YAML file mapping response providers to judge providers
2. **Loader** - Reads JSON input, validates examples
3. **Evaluator** - Sends each pair to appropriate LLM judge based on config
4. **Reporter** - Outputs per-example scores with reasoning and aggregates

## Data Format

### Config (`config.yaml`)

```yaml
judge_mapping:
  openai: anthropic    # responses from OpenAI models → judged by Anthropic
  anthropic: openai    # responses from Anthropic models → judged by OpenAI

judge_models:
  openai: gpt-4o-mini
  anthropic: claude-sonnet-4-20250514
```

### Input (`data/examples.json`)

```json
[
  {
    "id": "ticket-001",
    "ticket": "Jenkins pipeline failing with OOM error on build step...",
    "response": "This typically happens when the heap size is too small...",
    "model": "gpt-4o",
    "prompt_version": "v1.2"
  }
]
```

### Output (`results/output.json`)

```json
{
  "results": [
    {
      "id": "ticket-001",
      "model": "gpt-4o",
      "prompt_version": "v1.2",
      "relevance": {
        "score": 4,
        "reasoning": "Correctly identifies memory issue and suggests heap adjustment."
      },
      "tone": {
        "score": 5,
        "reasoning": "Professional and concise."
      }
    }
  ],
  "skipped": [
    {"index": 2, "reason": "Missing 'ticket' field"}
  ],
  "aggregates": {
    "by_model": {
      "gpt-4o": {
        "count": 10,
        "relevance": { "mean": 4.2, "min": 3, "max": 5 },
        "tone": { "mean": 4.5, "min": 3, "max": 5 }
      }
    },
    "by_prompt_version": {
      "v1.2": {
        "count": 10,
        "relevance": { "mean": 4.2, "min": 3, "max": 5 },
        "tone": { "mean": 4.5, "min": 3, "max": 5 }
      }
    },
    "by_model_and_prompt_version": {
      "gpt-4o|v1.2": {
        "count": 10,
        "relevance": { "mean": 4.2, "min": 3, "max": 5 },
        "tone": { "mean": 4.5, "min": 3, "max": 5 }
      }
    }
  }
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
├── config.yaml             # Judge mapping configuration
├── src/
│   └── eval_harness/
│       ├── __init__.py
│       ├── main.py         # CLI entry point
│       ├── config.py       # Load YAML config
│       ├── loader.py       # Load JSON input, validation
│       ├── evaluator.py    # LLM judge logic + prompts
│       ├── aggregator.py   # Compute aggregates by model/prompt
│       └── reporter.py     # Write results JSON
├── data/
│   └── examples.json       # Eval dataset
└── results/
    └── (output files)
```

## Usage

```bash
uv run python -m eval_harness.main data/examples.json --config config.yaml --output results/output.json
```

## Implementation Details

### Judge Selection Flow

1. Load config from YAML (judge_mapping and judge_models)
2. For each example, determine response provider from model name:
   - Models starting with `gpt-` or `o1-` → provider is `openai`
   - Models starting with `claude-` → provider is `anthropic`
3. Look up judge provider from `judge_mapping[response_provider]`
4. Look up judge model from `judge_models[judge_provider]`
5. Use appropriate client (OpenAI or Anthropic) for scoring

### Evaluator Flow

1. Load config and all examples from input JSON
2. Validate each example (id, ticket, response, model, prompt_version must be non-empty strings)
3. Skip invalid examples, log warnings
4. For each valid example:
   - Determine which judge to use based on config
   - Make two LLM calls (relevance + tone) using appropriate client
5. Parse structured JSON responses (score + reasoning)
6. Compute aggregates by model, prompt_version, and combination
7. Write results with aggregates to output JSON

### LLM Response Format

Both OpenAI and Anthropic return JSON:

```json
{"score": 4, "reasoning": "..."}
```

- OpenAI: Uses `response_format: { type: "json_object" }`
- Anthropic: Prompt instructs JSON response, parsed from text

### Error Handling

- Retry on rate limits (exponential backoff)
- Log and skip examples that fail after retries
- Report failed examples in output

### Input Validation

Required fields for each example:
- `id` - non-empty string
- `ticket` - non-empty string
- `response` - non-empty string
- `model` - non-empty string
- `prompt_version` - non-empty string

Invalid examples are skipped with reason logged in output.

## Dependencies

- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `pyyaml` - YAML config parsing
