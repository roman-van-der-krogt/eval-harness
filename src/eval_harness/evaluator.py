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
