import json
from unittest.mock import Mock, MagicMock, call
import pytest

from eval_harness.evaluator import (
    Score,
    EvalResult,
    RELEVANCE_RUBRIC,
    TONE_RUBRIC,
    evaluate_example,
    _build_prompt,
    _score_with_openai,
    _score_with_anthropic,
)
from eval_harness.loader import Example
from eval_harness.config import Config


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return Mock()


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    return Mock()


@pytest.fixture
def sample_config():
    """Create a sample Config for testing."""
    return Config(
        judge_mapping={"openai": "anthropic", "anthropic": "openai"},
        judge_models={"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-20250514"}
    )


@pytest.fixture
def sample_example():
    """Create a sample Example for testing."""
    return Example(
        id="test-123",
        ticket="User cannot log in with their password",
        response="Please try resetting your password using the forgot password link.",
        model="gpt-4o",
        prompt_version="v1.0"
    )


def create_mock_completion(score: int, reasoning: str):
    """Helper to create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_choice = MagicMock()

    # Structure: response.choices[0].message.content
    json_content = json.dumps({"score": score, "reasoning": reasoning})
    mock_message.content = json_content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    return mock_response


def create_mock_anthropic_message(score: int, reasoning: str):
    """Helper to create a mock Anthropic message response."""
    mock_response = MagicMock()
    mock_content_block = MagicMock()

    # Structure: response.content[0].text
    json_content = json.dumps({"score": score, "reasoning": reasoning})
    mock_content_block.text = json_content
    mock_response.content = [mock_content_block]

    return mock_response


class TestEvaluateExample:
    """Tests for evaluate_example function."""

    def test_uses_anthropic_judge_for_openai_models(self, sample_config, mock_anthropic_client):
        """Test that evaluate_example uses Anthropic judge for OpenAI models."""
        example = Example(
            id="test-123",
            ticket="User cannot log in",
            response="Try resetting your password",
            model="gpt-4o",
            prompt_version="v1.0"
        )

        # Setup mock responses for relevance and tone
        relevance_response = create_mock_anthropic_message(4, "Good relevance")
        tone_response = create_mock_anthropic_message(5, "Excellent tone")

        mock_anthropic_client.messages.create.side_effect = [
            relevance_response,
            tone_response,
        ]

        # Call the function
        result = evaluate_example(example, sample_config, anthropic_client=mock_anthropic_client)

        # Assertions
        assert isinstance(result, EvalResult)
        assert result.id == "test-123"
        assert result.model == "gpt-4o"
        assert result.prompt_version == "v1.0"
        assert isinstance(result.relevance, Score)
        assert result.relevance.score == 4
        assert result.relevance.reasoning == "Good relevance"
        assert isinstance(result.tone, Score)
        assert result.tone.score == 5
        assert result.tone.reasoning == "Excellent tone"

        # Verify the Anthropic client was called twice with correct model
        assert mock_anthropic_client.messages.create.call_count == 2
        calls = mock_anthropic_client.messages.create.call_args_list
        assert all(call.kwargs["model"] == "claude-sonnet-4-20250514" for call in calls)

    def test_uses_openai_judge_for_anthropic_models(self, sample_config, mock_openai_client):
        """Test that evaluate_example uses OpenAI judge for Anthropic models."""
        example = Example(
            id="test-456",
            ticket="App is crashing on startup",
            response="Please update to the latest version",
            model="claude-3-5-sonnet-20241022",
            prompt_version="v2.0"
        )

        # Setup mock responses for relevance and tone
        relevance_response = create_mock_completion(3, "Moderate relevance")
        tone_response = create_mock_completion(4, "Good tone")

        mock_openai_client.chat.completions.create.side_effect = [
            relevance_response,
            tone_response,
        ]

        # Call the function
        result = evaluate_example(example, sample_config, openai_client=mock_openai_client)

        # Assertions
        assert isinstance(result, EvalResult)
        assert result.id == "test-456"
        assert result.model == "claude-3-5-sonnet-20241022"
        assert result.prompt_version == "v2.0"
        assert result.relevance.score == 3
        assert result.tone.score == 4

        # Verify the OpenAI client was called twice with correct model
        assert mock_openai_client.chat.completions.create.call_count == 2
        calls = mock_openai_client.chat.completions.create.call_args_list
        assert all(call.kwargs["model"] == "gpt-4o-mini" for call in calls)

    def test_includes_model_and_prompt_version_in_result(self, sample_config, mock_anthropic_client, sample_example):
        """Test that evaluate_example includes model and prompt_version in EvalResult."""
        mock_anthropic_client.messages.create.return_value = create_mock_anthropic_message(3, "Test")

        result = evaluate_example(sample_example, sample_config, anthropic_client=mock_anthropic_client)

        assert result.model == "gpt-4o"
        assert result.prompt_version == "v1.0"


class TestBuildPrompt:
    """Tests for _build_prompt function."""

    def test_builds_prompt_with_all_components(self, sample_example):
        """Test that _build_prompt includes all required components."""
        prompt = _build_prompt(sample_example, "relevance", RELEVANCE_RUBRIC)

        assert "evaluating a support bot response for relevance" in prompt
        assert "TICKET:" in prompt
        assert sample_example.ticket in prompt
        assert "RESPONSE:" in prompt
        assert sample_example.response in prompt
        assert RELEVANCE_RUBRIC in prompt
        assert '{"score": <1-5>, "reasoning": "<brief explanation>"}' in prompt

    def test_builds_prompt_with_tone_dimension(self, sample_example):
        """Test that _build_prompt works with tone dimension."""
        prompt = _build_prompt(sample_example, "tone", TONE_RUBRIC)

        assert "evaluating a support bot response for tone" in prompt
        assert TONE_RUBRIC in prompt
        assert sample_example.ticket in prompt
        assert sample_example.response in prompt


class TestScoreWithOpenAI:
    """Tests for _score_with_openai function."""

    def test_returns_score_from_openai_response(self, mock_openai_client, sample_example):
        """Test that _score_with_openai correctly parses OpenAI response."""
        mock_response = create_mock_completion(4, "Good relevance")
        mock_openai_client.chat.completions.create.return_value = mock_response

        score = _score_with_openai(
            mock_openai_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "gpt-4o-mini"
        )

        assert isinstance(score, Score)
        assert score.score == 4
        assert score.reasoning == "Good relevance"

    def test_calls_openai_with_correct_model(self, mock_openai_client, sample_example):
        """Test that _score_with_openai uses the correct model."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        _score_with_openai(
            mock_openai_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "gpt-4o"
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"

    def test_uses_json_response_format(self, mock_openai_client, sample_example):
        """Test that _score_with_openai requests JSON response format."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        _score_with_openai(
            mock_openai_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "gpt-4o-mini"
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_constructs_correct_prompt(self, mock_openai_client, sample_example):
        """Test that _score_with_openai constructs the correct prompt."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        _score_with_openai(
            mock_openai_client,
            sample_example,
            "tone",
            TONE_RUBRIC,
            "gpt-4o-mini"
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert "evaluating a support bot response for tone" in content
        assert TONE_RUBRIC in content


class TestScoreWithAnthropic:
    """Tests for _score_with_anthropic function."""

    def test_returns_score_from_anthropic_response(self, mock_anthropic_client, sample_example):
        """Test that _score_with_anthropic correctly parses Anthropic response."""
        mock_response = create_mock_anthropic_message(5, "Excellent tone")
        mock_anthropic_client.messages.create.return_value = mock_response

        score = _score_with_anthropic(
            mock_anthropic_client,
            sample_example,
            "tone",
            TONE_RUBRIC,
            "claude-sonnet-4-20250514"
        )

        assert isinstance(score, Score)
        assert score.score == 5
        assert score.reasoning == "Excellent tone"

    def test_calls_anthropic_with_correct_model(self, mock_anthropic_client, sample_example):
        """Test that _score_with_anthropic uses the correct model."""
        mock_anthropic_client.messages.create.return_value = create_mock_anthropic_message(3, "Test")

        _score_with_anthropic(
            mock_anthropic_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "claude-3-5-sonnet-20241022"
        )

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"

    def test_sets_max_tokens(self, mock_anthropic_client, sample_example):
        """Test that _score_with_anthropic sets max_tokens parameter."""
        mock_anthropic_client.messages.create.return_value = create_mock_anthropic_message(3, "Test")

        _score_with_anthropic(
            mock_anthropic_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "claude-sonnet-4-20250514"
        )

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == 256

    def test_constructs_correct_prompt(self, mock_anthropic_client, sample_example):
        """Test that _score_with_anthropic constructs the correct prompt."""
        mock_anthropic_client.messages.create.return_value = create_mock_anthropic_message(3, "Test")

        _score_with_anthropic(
            mock_anthropic_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "claude-sonnet-4-20250514"
        )

        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert "evaluating a support bot response for relevance" in content
        assert RELEVANCE_RUBRIC in content


class TestDataClasses:
    """Tests for Score and EvalResult dataclasses."""

    def test_score_dataclass(self):
        """Test Score dataclass initialization."""
        score = Score(score=4, reasoning="Good response")
        assert score.score == 4
        assert score.reasoning == "Good response"

    def test_eval_result_dataclass(self):
        """Test EvalResult dataclass initialization."""
        relevance = Score(score=4, reasoning="Relevant")
        tone = Score(score=5, reasoning="Professional")

        result = EvalResult(
            id="test-456",
            model="gpt-4o",
            prompt_version="v2.0",
            relevance=relevance,
            tone=tone
        )

        assert result.id == "test-456"
        assert result.model == "gpt-4o"
        assert result.prompt_version == "v2.0"
        assert result.relevance == relevance
        assert result.tone == tone


class TestRubrics:
    """Tests for rubric constants."""

    def test_relevance_rubric_exists(self):
        """Test that RELEVANCE_RUBRIC is defined and non-empty."""
        assert RELEVANCE_RUBRIC
        assert isinstance(RELEVANCE_RUBRIC, str)
        assert "1-5 scale" in RELEVANCE_RUBRIC
        assert "5:" in RELEVANCE_RUBRIC
        assert "1:" in RELEVANCE_RUBRIC

    def test_tone_rubric_exists(self):
        """Test that TONE_RUBRIC is defined and non-empty."""
        assert TONE_RUBRIC
        assert isinstance(TONE_RUBRIC, str)
        assert "1-5 scale" in TONE_RUBRIC
        assert "5:" in TONE_RUBRIC
        assert "1:" in TONE_RUBRIC
