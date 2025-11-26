import json
from unittest.mock import Mock, MagicMock, call
import pytest

from eval_harness.evaluator import (
    Score,
    EvalResult,
    RELEVANCE_RUBRIC,
    TONE_RUBRIC,
    evaluate_example,
    _score_dimension,
)
from eval_harness.loader import Example


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return Mock()


@pytest.fixture
def sample_example():
    """Create a sample Example for testing."""
    return Example(
        id="test-123",
        ticket="User cannot log in with their password",
        response="Please try resetting your password using the forgot password link.",
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


class TestEvaluateExample:
    """Tests for evaluate_example function."""

    def test_returns_eval_result_with_both_scores(self, mock_openai_client, sample_example):
        """Test that evaluate_example returns an EvalResult with both relevance and tone scores."""
        # Setup mock responses for relevance and tone
        relevance_response = create_mock_completion(4, "Good relevance")
        tone_response = create_mock_completion(5, "Excellent tone")

        mock_openai_client.chat.completions.create.side_effect = [
            relevance_response,
            tone_response,
        ]

        # Call the function
        result = evaluate_example(mock_openai_client, sample_example)

        # Assertions
        assert isinstance(result, EvalResult)
        assert result.id == "test-123"
        assert isinstance(result.relevance, Score)
        assert result.relevance.score == 4
        assert result.relevance.reasoning == "Good relevance"
        assert isinstance(result.tone, Score)
        assert result.tone.score == 5
        assert result.tone.reasoning == "Excellent tone"

        # Verify the client was called twice
        assert mock_openai_client.chat.completions.create.call_count == 2

    def test_uses_default_model(self, mock_openai_client, sample_example):
        """Test that evaluate_example uses gpt-4o-mini as default model."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        evaluate_example(mock_openai_client, sample_example)

        # Check that both calls used the default model
        calls = mock_openai_client.chat.completions.create.call_args_list
        assert all(call.kwargs["model"] == "gpt-4o-mini" for call in calls)

    def test_uses_custom_model(self, mock_openai_client, sample_example):
        """Test that evaluate_example uses custom model when specified."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        evaluate_example(mock_openai_client, sample_example, model="gpt-4o")

        # Check that both calls used the custom model
        calls = mock_openai_client.chat.completions.create.call_args_list
        assert all(call.kwargs["model"] == "gpt-4o" for call in calls)

    def test_calls_score_dimension_for_relevance_and_tone(self, mock_openai_client, sample_example):
        """Test that evaluate_example scores both dimensions."""
        relevance_response = create_mock_completion(3, "Moderate relevance")
        tone_response = create_mock_completion(4, "Good tone")

        mock_openai_client.chat.completions.create.side_effect = [
            relevance_response,
            tone_response,
        ]

        result = evaluate_example(mock_openai_client, sample_example)

        # Verify both dimensions were scored
        assert result.relevance.score == 3
        assert result.tone.score == 4


class TestScoreDimension:
    """Tests for _score_dimension function."""

    def test_parses_json_response_correctly(self, mock_openai_client, sample_example):
        """Test that _score_dimension correctly parses JSON from OpenAI response."""
        mock_response = create_mock_completion(5, "Perfectly addresses the issue")
        mock_openai_client.chat.completions.create.return_value = mock_response

        score = _score_dimension(
            mock_openai_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "gpt-4o-mini",
        )

        assert isinstance(score, Score)
        assert score.score == 5
        assert score.reasoning == "Perfectly addresses the issue"

    def test_constructs_correct_prompt_with_ticket_response_rubric(
        self, mock_openai_client, sample_example
    ):
        """Test that _score_dimension constructs the correct prompt."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        _score_dimension(
            mock_openai_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "gpt-4o-mini",
        )

        # Get the actual call arguments
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        # Verify the prompt structure
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content = messages[0]["content"]
        # Check that the prompt contains all required elements
        assert "evaluating a support bot response for relevance" in content
        assert "TICKET:" in content
        assert sample_example.ticket in content
        assert "RESPONSE:" in content
        assert sample_example.response in content
        assert RELEVANCE_RUBRIC in content
        assert '{"score": <1-5>, "reasoning": "<brief explanation>"}' in content

    def test_constructs_prompt_with_tone_dimension(self, mock_openai_client, sample_example):
        """Test that _score_dimension constructs prompt with tone dimension and rubric."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(4, "Test")

        _score_dimension(
            mock_openai_client,
            sample_example,
            "tone",
            TONE_RUBRIC,
            "gpt-4o-mini",
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        content = messages[0]["content"]

        # Check that the prompt contains tone-specific elements
        assert "evaluating a support bot response for tone" in content
        assert TONE_RUBRIC in content

    def test_model_parameter_passed_through(self, mock_openai_client, sample_example):
        """Test that the model parameter is correctly passed to the OpenAI API."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        custom_model = "gpt-4-turbo"
        _score_dimension(
            mock_openai_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            custom_model,
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == custom_model

    def test_uses_json_response_format(self, mock_openai_client, sample_example):
        """Test that _score_dimension requests JSON response format."""
        mock_openai_client.chat.completions.create.return_value = create_mock_completion(3, "Test")

        _score_dimension(
            mock_openai_client,
            sample_example,
            "relevance",
            RELEVANCE_RUBRIC,
            "gpt-4o-mini",
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_handles_different_score_values(self, mock_openai_client, sample_example):
        """Test that _score_dimension handles all valid score values (1-5)."""
        for score_value in range(1, 6):
            mock_openai_client.chat.completions.create.return_value = create_mock_completion(
                score_value, f"Reasoning for score {score_value}"
            )

            score = _score_dimension(
                mock_openai_client,
                sample_example,
                "relevance",
                RELEVANCE_RUBRIC,
                "gpt-4o-mini",
            )

            assert score.score == score_value
            assert score.reasoning == f"Reasoning for score {score_value}"


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

        result = EvalResult(id="test-456", relevance=relevance, tone=tone)

        assert result.id == "test-456"
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
