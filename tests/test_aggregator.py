import pytest
from eval_harness.evaluator import EvalResult, Score
from eval_harness.aggregator import compute_aggregates


@pytest.fixture
def sample_results():
    return [
        EvalResult("1", "gpt-4o", "v1.0", Score(4, "ok"), Score(5, "good")),
        EvalResult("2", "gpt-4o", "v1.0", Score(3, "ok"), Score(4, "good")),
        EvalResult("3", "claude-sonnet", "v0.9", Score(5, "great"), Score(5, "great")),
    ]


def test_compute_aggregates_by_model(sample_results):
    """Test that results are grouped by model correctly."""
    result = compute_aggregates(sample_results)

    assert "by_model" in result
    assert "gpt-4o" in result["by_model"]
    assert "claude-sonnet" in result["by_model"]

    # gpt-4o should have 2 results
    assert result["by_model"]["gpt-4o"]["count"] == 2
    # claude-sonnet should have 1 result
    assert result["by_model"]["claude-sonnet"]["count"] == 1


def test_compute_aggregates_by_prompt_version(sample_results):
    """Test that results are grouped by prompt_version correctly."""
    result = compute_aggregates(sample_results)

    assert "by_prompt_version" in result
    assert "v1.0" in result["by_prompt_version"]
    assert "v0.9" in result["by_prompt_version"]

    # v1.0 should have 2 results
    assert result["by_prompt_version"]["v1.0"]["count"] == 2
    # v0.9 should have 1 result
    assert result["by_prompt_version"]["v0.9"]["count"] == 1


def test_compute_aggregates_by_model_and_prompt_version(sample_results):
    """Test that results are grouped by combination of model and prompt_version."""
    result = compute_aggregates(sample_results)

    assert "by_model_and_prompt_version" in result
    assert "gpt-4o|v1.0" in result["by_model_and_prompt_version"]
    assert "claude-sonnet|v0.9" in result["by_model_and_prompt_version"]

    # gpt-4o|v1.0 should have 2 results
    assert result["by_model_and_prompt_version"]["gpt-4o|v1.0"]["count"] == 2
    # claude-sonnet|v0.9 should have 1 result
    assert result["by_model_and_prompt_version"]["claude-sonnet|v0.9"]["count"] == 1


def test_compute_stats_count(sample_results):
    """Test that count is calculated correctly."""
    result = compute_aggregates(sample_results)

    # Check count for gpt-4o
    assert result["by_model"]["gpt-4o"]["count"] == 2
    # Check count for claude-sonnet
    assert result["by_model"]["claude-sonnet"]["count"] == 1


def test_compute_stats_mean(sample_results):
    """Test that mean is calculated correctly and rounded to 2 decimals."""
    result = compute_aggregates(sample_results)

    # gpt-4o has relevance scores [4, 3] -> mean = 3.5
    assert result["by_model"]["gpt-4o"]["relevance"]["mean"] == 3.5
    # gpt-4o has tone scores [5, 4] -> mean = 4.5
    assert result["by_model"]["gpt-4o"]["tone"]["mean"] == 4.5

    # claude-sonnet has relevance score [5] -> mean = 5.0
    assert result["by_model"]["claude-sonnet"]["relevance"]["mean"] == 5.0
    # claude-sonnet has tone score [5] -> mean = 5.0
    assert result["by_model"]["claude-sonnet"]["tone"]["mean"] == 5.0


def test_compute_stats_min_max(sample_results):
    """Test that min and max are calculated correctly."""
    result = compute_aggregates(sample_results)

    # gpt-4o has relevance scores [4, 3]
    assert result["by_model"]["gpt-4o"]["relevance"]["min"] == 3
    assert result["by_model"]["gpt-4o"]["relevance"]["max"] == 4

    # gpt-4o has tone scores [5, 4]
    assert result["by_model"]["gpt-4o"]["tone"]["min"] == 4
    assert result["by_model"]["gpt-4o"]["tone"]["max"] == 5

    # claude-sonnet has relevance score [5]
    assert result["by_model"]["claude-sonnet"]["relevance"]["min"] == 5
    assert result["by_model"]["claude-sonnet"]["relevance"]["max"] == 5


def test_compute_aggregates_single_result():
    """Test that aggregates work with a single result."""
    single_result = [
        EvalResult("1", "gpt-4o", "v1.0", Score(4, "ok"), Score(5, "good"))
    ]

    result = compute_aggregates(single_result)

    assert result["by_model"]["gpt-4o"]["count"] == 1
    assert result["by_model"]["gpt-4o"]["relevance"]["mean"] == 4.0
    assert result["by_model"]["gpt-4o"]["relevance"]["min"] == 4
    assert result["by_model"]["gpt-4o"]["relevance"]["max"] == 4
    assert result["by_model"]["gpt-4o"]["tone"]["mean"] == 5.0
    assert result["by_model"]["gpt-4o"]["tone"]["min"] == 5
    assert result["by_model"]["gpt-4o"]["tone"]["max"] == 5


def test_compute_aggregates_empty_results():
    """Test that empty results return empty dicts."""
    empty_results = []

    result = compute_aggregates(empty_results)

    assert result["by_model"] == {}
    assert result["by_prompt_version"] == {}
    assert result["by_model_and_prompt_version"] == {}
