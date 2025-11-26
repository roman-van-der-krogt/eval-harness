import json
import pytest
from pathlib import Path
from eval_harness.loader import Example, LoadResult, load_examples, _validate_example


def test_load_valid_examples_successfully(tmp_path):
    """Test loading a file with all valid examples."""
    test_file = tmp_path / "valid.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Response 1"},
        {"id": "2", "ticket": "TICKET-456", "response": "Response 2"},
        {"id": "3", "ticket": "TICKET-789", "response": "Response 3"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert isinstance(result, LoadResult)
    assert len(result.examples) == 3
    assert len(result.skipped) == 0

    assert result.examples[0].id == "1"
    assert result.examples[0].ticket == "TICKET-123"
    assert result.examples[0].response == "Response 1"

    assert result.examples[1].id == "2"
    assert result.examples[2].id == "3"


def test_reject_non_array_json(tmp_path):
    """Test that non-array JSON raises ValueError."""
    test_file = tmp_path / "object.json"
    test_data = {"id": "1", "ticket": "TICKET-123", "response": "Response"}
    test_file.write_text(json.dumps(test_data))

    with pytest.raises(ValueError, match="Input must be a JSON array"):
        load_examples(test_file)


def test_skip_items_that_arent_objects(tmp_path):
    """Test that items that aren't objects are skipped."""
    test_file = tmp_path / "mixed.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid"},
        "string item",
        123,
        None,
        True,
        ["nested", "array"]
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 5

    assert result.examples[0].id == "1"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "Item is not an object"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "Item is not an object"
    assert result.skipped[2]["index"] == 3
    assert result.skipped[2]["reason"] == "Item is not an object"
    assert result.skipped[3]["index"] == 4
    assert result.skipped[3]["reason"] == "Item is not an object"
    assert result.skipped[4]["index"] == 5
    assert result.skipped[4]["reason"] == "Item is not an object"


def test_skip_items_missing_required_fields(tmp_path):
    """Test that items missing required fields are skipped."""
    test_file = tmp_path / "missing_fields.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid"},
        {"ticket": "TICKET-456", "response": "Missing id"},
        {"id": "2", "response": "Missing ticket"},
        {"id": "3", "ticket": "TICKET-789"},
        {"id": "4"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 4

    assert result.examples[0].id == "1"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "Missing 'id' field"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "Missing 'ticket' field"
    assert result.skipped[2]["index"] == 3
    assert result.skipped[2]["reason"] == "Missing 'response' field"
    assert result.skipped[3]["index"] == 4
    assert result.skipped[3]["reason"] == "Missing 'ticket' field"


def test_skip_items_with_non_string_fields(tmp_path):
    """Test that items with non-string field values are skipped."""
    test_file = tmp_path / "non_string.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid"},
        {"id": 123, "ticket": "TICKET-456", "response": "Numeric id"},
        {"id": "2", "ticket": 456, "response": "Numeric ticket"},
        {"id": "3", "ticket": "TICKET-789", "response": 789},
        {"id": None, "ticket": "TICKET-000", "response": "Null id"},
        {"id": "4", "ticket": ["TICKET-111"], "response": "Array ticket"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 5

    assert result.examples[0].id == "1"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "'id' must be a string"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "'ticket' must be a string"
    assert result.skipped[2]["index"] == 3
    assert result.skipped[2]["reason"] == "'response' must be a string"
    assert result.skipped[3]["index"] == 4
    assert result.skipped[3]["reason"] == "'id' must be a string"
    assert result.skipped[4]["index"] == 5
    assert result.skipped[4]["reason"] == "'ticket' must be a string"


def test_skip_items_with_empty_or_whitespace_only_fields(tmp_path):
    """Test that items with empty or whitespace-only fields are skipped."""
    test_file = tmp_path / "empty_fields.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid"},
        {"id": "", "ticket": "TICKET-456", "response": "Empty id"},
        {"id": "2", "ticket": "", "response": "Empty ticket"},
        {"id": "3", "ticket": "TICKET-789", "response": ""},
        {"id": "  ", "ticket": "TICKET-000", "response": "Whitespace id"},
        {"id": "4", "ticket": "  \t\n  ", "response": "Whitespace ticket"},
        {"id": "5", "ticket": "TICKET-111", "response": "  \t  "}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 6

    assert result.examples[0].id == "1"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "'id' is empty"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "'ticket' is empty"
    assert result.skipped[2]["index"] == 3
    assert result.skipped[2]["reason"] == "'response' is empty"
    assert result.skipped[3]["index"] == 4
    assert result.skipped[3]["reason"] == "'id' is empty"
    assert result.skipped[4]["index"] == 5
    assert result.skipped[4]["reason"] == "'ticket' is empty"
    assert result.skipped[5]["index"] == 6
    assert result.skipped[5]["reason"] == "'response' is empty"


def test_mixed_valid_and_invalid_examples(tmp_path):
    """Test that valid examples are loaded while invalid ones are skipped."""
    test_file = tmp_path / "mixed.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid 1"},
        "not an object",
        {"id": "2", "ticket": "TICKET-456", "response": "Valid 2"},
        {"ticket": "TICKET-789", "response": "Missing id"},
        {"id": "3", "ticket": "TICKET-000", "response": "Valid 3"},
        {"id": 123, "ticket": "TICKET-111", "response": "Non-string id"},
        {"id": "4", "ticket": "", "response": "Empty ticket"},
        {"id": "5", "ticket": "TICKET-222", "response": "Valid 4"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 4
    assert len(result.skipped) == 4

    # Check valid examples
    assert result.examples[0].id == "1"
    assert result.examples[0].ticket == "TICKET-123"
    assert result.examples[1].id == "2"
    assert result.examples[1].ticket == "TICKET-456"
    assert result.examples[2].id == "3"
    assert result.examples[2].ticket == "TICKET-000"
    assert result.examples[3].id == "5"
    assert result.examples[3].ticket == "TICKET-222"

    # Check skipped items
    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "Item is not an object"
    assert result.skipped[1]["index"] == 3
    assert result.skipped[1]["reason"] == "Missing 'id' field"
    assert result.skipped[2]["index"] == 5
    assert result.skipped[2]["reason"] == "'id' must be a string"
    assert result.skipped[3]["index"] == 6
    assert result.skipped[3]["reason"] == "'ticket' is empty"


def test_validate_example_returns_none_for_valid():
    """Test that _validate_example returns None for valid examples."""
    valid_item = {"id": "1", "ticket": "TICKET-123", "response": "Test response"}
    assert _validate_example(valid_item) is None


def test_validate_example_detects_non_object():
    """Test that _validate_example detects non-dict items."""
    assert _validate_example("string") == "Item is not an object"
    assert _validate_example(123) == "Item is not an object"
    assert _validate_example(None) == "Item is not an object"
    assert _validate_example([]) == "Item is not an object"


def test_validate_example_detects_missing_fields():
    """Test that _validate_example detects missing fields."""
    assert _validate_example({"ticket": "T", "response": "R"}) == "Missing 'id' field"
    assert _validate_example({"id": "1", "response": "R"}) == "Missing 'ticket' field"
    assert _validate_example({"id": "1", "ticket": "T"}) == "Missing 'response' field"


def test_validate_example_detects_non_string_fields():
    """Test that _validate_example detects non-string field values."""
    assert _validate_example({"id": 123, "ticket": "T", "response": "R"}) == "'id' must be a string"
    assert _validate_example({"id": "1", "ticket": 456, "response": "R"}) == "'ticket' must be a string"
    assert _validate_example({"id": "1", "ticket": "T", "response": 789}) == "'response' must be a string"


def test_validate_example_detects_empty_fields():
    """Test that _validate_example detects empty or whitespace-only fields."""
    assert _validate_example({"id": "", "ticket": "T", "response": "R"}) == "'id' is empty"
    assert _validate_example({"id": "1", "ticket": "", "response": "R"}) == "'ticket' is empty"
    assert _validate_example({"id": "1", "ticket": "T", "response": ""}) == "'response' is empty"
    assert _validate_example({"id": "  ", "ticket": "T", "response": "R"}) == "'id' is empty"
    assert _validate_example({"id": "1", "ticket": "  \t\n  ", "response": "R"}) == "'ticket' is empty"
