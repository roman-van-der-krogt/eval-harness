import json
from pathlib import Path

import pytest

from eval_harness.evaluator import EvalResult, Score
from eval_harness.reporter import write_results, _result_to_dict


@pytest.fixture
def sample_eval_result():
    """Create a sample EvalResult for testing."""
    return EvalResult(
        id="test-001",
        model="gpt-4o",
        prompt_version="v1",
        relevance=Score(score=5, reasoning="Directly addresses the issue"),
        tone=Score(score=4, reasoning="Professional with minor verbosity")
    )


@pytest.fixture
def sample_skipped():
    """Create sample skipped items for testing."""
    return [
        {"id": "skip-001", "reason": "Missing ticket field"},
        {"id": "skip-002", "reason": "Invalid format"}
    ]


@pytest.fixture
def sample_aggregates():
    return {
        "by_model": {
            "gpt-4o": {
                "count": 1,
                "relevance": {"mean": 4.0, "min": 4, "max": 4},
                "tone": {"mean": 5.0, "min": 5, "max": 5}
            }
        },
        "by_prompt_version": {},
        "by_model_and_prompt_version": {}
    }


def test_write_results_to_file_with_correct_json_structure(tmp_path, sample_eval_result, sample_skipped, sample_aggregates):
    """Test that write_results creates a file with the correct JSON structure."""
    output_path = tmp_path / "output.json"
    results = [sample_eval_result]

    write_results(results, sample_skipped, sample_aggregates, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)

    assert "results" in data
    assert "skipped" in data
    assert len(data["results"]) == 1
    assert len(data["skipped"]) == 2

    # Verify structure of results
    result = data["results"][0]
    assert result["id"] == "test-001"
    assert result["relevance"]["score"] == 5
    assert result["relevance"]["reasoning"] == "Directly addresses the issue"
    assert result["tone"]["score"] == 4
    assert result["tone"]["reasoning"] == "Professional with minor verbosity"

    # Verify skipped items
    assert data["skipped"] == sample_skipped


def test_creates_parent_directories_if_not_exist(tmp_path, sample_eval_result, sample_aggregates):
    """Test that write_results creates parent directories if they don't exist."""
    output_path = tmp_path / "nested" / "directories" / "output.json"

    # Verify parent directories don't exist yet
    assert not output_path.parent.exists()

    write_results([sample_eval_result], [], sample_aggregates, output_path)

    # Verify directories were created and file exists
    assert output_path.parent.exists()
    assert output_path.exists()


def test_handles_empty_results_list(tmp_path):
    """Test that write_results handles an empty results list correctly."""
    output_path = tmp_path / "empty_results.json"

    write_results([], [], {}, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)

    assert data["results"] == []
    assert data["skipped"] == []


def test_handles_empty_skipped_list(tmp_path, sample_eval_result, sample_aggregates):
    """Test that write_results handles an empty skipped list correctly."""
    output_path = tmp_path / "no_skipped.json"
    results = [sample_eval_result]

    write_results(results, [], sample_aggregates, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)

    assert len(data["results"]) == 1
    assert data["skipped"] == []


def test_includes_skipped_items_in_output(tmp_path, sample_skipped):
    """Test that skipped items are correctly included in the output."""
    output_path = tmp_path / "with_skipped.json"

    write_results([], sample_skipped, {}, output_path)

    with open(output_path) as f:
        data = json.load(f)

    assert data["skipped"] == sample_skipped
    assert len(data["skipped"]) == 2


def test_result_to_dict_converts_eval_result_correctly(sample_eval_result):
    """Test that _result_to_dict correctly converts EvalResult to dictionary."""
    result_dict = _result_to_dict(sample_eval_result)

    assert isinstance(result_dict, dict)
    assert result_dict["id"] == "test-001"
    assert result_dict["model"] == "gpt-4o"
    assert result_dict["prompt_version"] == "v1"

    # Check relevance structure
    assert "relevance" in result_dict
    assert result_dict["relevance"]["score"] == 5
    assert result_dict["relevance"]["reasoning"] == "Directly addresses the issue"

    # Check tone structure
    assert "tone" in result_dict
    assert result_dict["tone"]["score"] == 4
    assert result_dict["tone"]["reasoning"] == "Professional with minor verbosity"


def test_result_to_dict_with_different_scores():
    """Test _result_to_dict with various score values."""
    result = EvalResult(
        id="test-002",
        model="claude-3-5-sonnet-20241022",
        prompt_version="v2",
        relevance=Score(score=1, reasoning="Completely off-topic"),
        tone=Score(score=3, reasoning="Acceptable but verbose")
    )

    result_dict = _result_to_dict(result)

    assert result_dict["id"] == "test-002"
    assert result_dict["model"] == "claude-3-5-sonnet-20241022"
    assert result_dict["prompt_version"] == "v2"
    assert result_dict["relevance"]["score"] == 1
    assert result_dict["relevance"]["reasoning"] == "Completely off-topic"
    assert result_dict["tone"]["score"] == 3
    assert result_dict["tone"]["reasoning"] == "Acceptable but verbose"


def test_write_results_with_multiple_results(tmp_path):
    """Test writing multiple results to file."""
    output_path = tmp_path / "multiple_results.json"

    results = [
        EvalResult(
            id="test-001",
            model="gpt-4o",
            prompt_version="v1",
            relevance=Score(score=5, reasoning="Perfect"),
            tone=Score(score=5, reasoning="Excellent")
        ),
        EvalResult(
            id="test-002",
            model="gpt-4o",
            prompt_version="v1",
            relevance=Score(score=3, reasoning="Partial"),
            tone=Score(score=2, reasoning="Too informal")
        ),
        EvalResult(
            id="test-003",
            model="gpt-4o",
            prompt_version="v1",
            relevance=Score(score=4, reasoning="Good"),
            tone=Score(score=4, reasoning="Mostly good")
        )
    ]

    skipped = [{"id": "skip-001", "reason": "Error"}]
    aggregates = {
        "by_model": {
            "gpt-4o": {
                "count": 3,
                "relevance": {"mean": 4.0, "min": 3, "max": 5},
                "tone": {"mean": 3.67, "min": 2, "max": 5}
            }
        },
        "by_prompt_version": {},
        "by_model_and_prompt_version": {}
    }

    write_results(results, skipped, aggregates, output_path)

    with open(output_path) as f:
        data = json.load(f)

    assert len(data["results"]) == 3
    assert len(data["skipped"]) == 1

    # Verify each result has correct structure
    for i, result in enumerate(data["results"]):
        assert result["id"] == f"test-00{i+1}"
        assert "model" in result
        assert "prompt_version" in result
        assert "relevance" in result
        assert "tone" in result
        assert "score" in result["relevance"]
        assert "reasoning" in result["relevance"]
        assert "score" in result["tone"]
        assert "reasoning" in result["tone"]


def test_write_results_json_is_properly_formatted(tmp_path, sample_eval_result, sample_aggregates):
    """Test that the output JSON is properly formatted with indentation."""
    output_path = tmp_path / "formatted.json"

    write_results([sample_eval_result], [], sample_aggregates, output_path)

    # Read the raw file content to check formatting
    with open(output_path) as f:
        content = f.read()

    # Check that it has proper indentation (indent=2)
    assert "\n" in content
    assert "  " in content  # Should have 2-space indentation

    # Verify it's still valid JSON
    data = json.loads(content)
    assert "results" in data
    assert "skipped" in data


def test_write_results_includes_aggregates(tmp_path, sample_eval_result, sample_aggregates):
    """Test that write_results includes aggregates in the output."""
    output_path = tmp_path / "with_aggregates.json"

    write_results([sample_eval_result], [], sample_aggregates, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)

    assert "aggregates" in data
    assert data["aggregates"] == sample_aggregates
    assert "by_model" in data["aggregates"]
    assert "gpt-4o" in data["aggregates"]["by_model"]
    assert data["aggregates"]["by_model"]["gpt-4o"]["count"] == 1
