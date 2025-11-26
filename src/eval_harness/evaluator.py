import json
from dataclasses import dataclass
from openai import OpenAI
from anthropic import Anthropic

from .loader import Example
from .config import Config, get_provider_from_model


@dataclass
class Score:
    score: int
    reasoning: str


@dataclass
class EvalResult:
    id: str
    model: str
    prompt_version: str
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


def evaluate_example(
    example: Example,
    config: Config,
    openai_client: OpenAI | None = None,
    anthropic_client: Anthropic | None = None,
) -> EvalResult:
    """Evaluate a single example for relevance and tone."""
    response_provider = get_provider_from_model(example.model)
    judge_provider = config.judge_mapping[response_provider]
    judge_model = config.judge_models[judge_provider]

    if judge_provider == "openai":
        if openai_client is None:
            openai_client = OpenAI()
        relevance = _score_with_openai(openai_client, example, "relevance", RELEVANCE_RUBRIC, judge_model)
        tone = _score_with_openai(openai_client, example, "tone", TONE_RUBRIC, judge_model)
    else:
        if anthropic_client is None:
            anthropic_client = Anthropic()
        relevance = _score_with_anthropic(anthropic_client, example, "relevance", RELEVANCE_RUBRIC, judge_model)
        tone = _score_with_anthropic(anthropic_client, example, "tone", TONE_RUBRIC, judge_model)

    return EvalResult(
        id=example.id,
        model=example.model,
        prompt_version=example.prompt_version,
        relevance=relevance,
        tone=tone
    )


def _build_prompt(example: Example, dimension: str, rubric: str) -> str:
    """Build the scoring prompt."""
    return f"""You are evaluating a support bot response for {dimension}.

TICKET:
{example.ticket}

RESPONSE:
{example.response}

{rubric}

Respond with JSON only: {{"score": <1-5>, "reasoning": "<brief explanation>"}}"""


def _score_with_openai(
    client: OpenAI,
    example: Example,
    dimension: str,
    rubric: str,
    model: str
) -> Score:
    """Score using OpenAI."""
    prompt = _build_prompt(example, dimension, rubric)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return Score(score=result["score"], reasoning=result["reasoning"])


def _score_with_anthropic(
    client: Anthropic,
    example: Example,
    dimension: str,
    rubric: str,
    model: str
) -> Score:
    """Score using Anthropic."""
    prompt = _build_prompt(example, dimension, rubric)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(response.content[0].text)
    return Score(score=result["score"], reasoning=result["reasoning"])
