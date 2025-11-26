import json
import pytest
from pathlib import Path
from eval_harness.loader import Example, LoadResult, load_examples, _validate_example


def test_load_valid_examples_successfully(tmp_path):
    """Test loading a file with all valid examples."""
    test_file = tmp_path / "valid.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Response 1", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "ticket": "TICKET-456", "response": "Response 2", "model": "claude-3", "prompt_version": "v2"},
        {"id": "3", "ticket": "TICKET-789", "response": "Response 3", "model": "gpt-3.5", "prompt_version": "v1"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert isinstance(result, LoadResult)
    assert len(result.examples) == 3
    assert len(result.skipped) == 0

    assert result.examples[0].id == "1"
    assert result.examples[0].ticket == "TICKET-123"
    assert result.examples[0].response == "Response 1"
    assert result.examples[0].model == "gpt-4"
    assert result.examples[0].prompt_version == "v1"

    assert result.examples[1].id == "2"
    assert result.examples[1].model == "claude-3"
    assert result.examples[1].prompt_version == "v2"

    assert result.examples[2].id == "3"
    assert result.examples[2].model == "gpt-3.5"
    assert result.examples[2].prompt_version == "v1"


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
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
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
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
        {"ticket": "TICKET-456", "response": "Missing id", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "response": "Missing ticket", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "3", "ticket": "TICKET-789", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "4", "ticket": "TICKET-000", "response": "Valid", "prompt_version": "v1"}
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
    assert result.skipped[3]["reason"] == "Missing 'model' field"


def test_skip_items_missing_model_field(tmp_path):
    """Test that items missing the model field are skipped."""
    test_file = tmp_path / "missing_model.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "ticket": "TICKET-456", "response": "Missing model", "prompt_version": "v1"},
        {"id": "3", "ticket": "TICKET-789", "response": "Also missing model", "prompt_version": "v2"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 2

    assert result.examples[0].id == "1"
    assert result.examples[0].model == "gpt-4"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "Missing 'model' field"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "Missing 'model' field"


def test_skip_items_missing_prompt_version_field(tmp_path):
    """Test that items missing the prompt_version field are skipped."""
    test_file = tmp_path / "missing_prompt_version.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "ticket": "TICKET-456", "response": "Missing prompt_version", "model": "gpt-4"},
        {"id": "3", "ticket": "TICKET-789", "response": "Also missing prompt_version", "model": "claude-3"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 2

    assert result.examples[0].id == "1"
    assert result.examples[0].prompt_version == "v1"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "Missing 'prompt_version' field"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "Missing 'prompt_version' field"


def test_skip_items_with_non_string_fields(tmp_path):
    """Test that items with non-string field values are skipped."""
    test_file = tmp_path / "non_string.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
        {"id": 123, "ticket": "TICKET-456", "response": "Numeric id", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "ticket": 456, "response": "Numeric ticket", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "3", "ticket": "TICKET-789", "response": 789, "model": "gpt-4", "prompt_version": "v1"},
        {"id": None, "ticket": "TICKET-000", "response": "Null id", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "4", "ticket": ["TICKET-111"], "response": "Array ticket", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "5", "ticket": "TICKET-222", "response": "Numeric model", "model": 123, "prompt_version": "v1"},
        {"id": "6", "ticket": "TICKET-333", "response": "Numeric prompt_version", "model": "gpt-4", "prompt_version": 456}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 7

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
    assert result.skipped[5]["index"] == 6
    assert result.skipped[5]["reason"] == "'model' must be a string"
    assert result.skipped[6]["index"] == 7
    assert result.skipped[6]["reason"] == "'prompt_version' must be a string"


def test_skip_items_with_empty_or_whitespace_only_fields(tmp_path):
    """Test that items with empty or whitespace-only fields are skipped."""
    test_file = tmp_path / "empty_fields.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "", "ticket": "TICKET-456", "response": "Empty id", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "ticket": "", "response": "Empty ticket", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "3", "ticket": "TICKET-789", "response": "", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "  ", "ticket": "TICKET-000", "response": "Whitespace id", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "4", "ticket": "  \t\n  ", "response": "Whitespace ticket", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "5", "ticket": "TICKET-111", "response": "  \t  ", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "6", "ticket": "TICKET-222", "response": "Empty model", "model": "", "prompt_version": "v1"},
        {"id": "7", "ticket": "TICKET-333", "response": "Empty prompt_version", "model": "gpt-4", "prompt_version": ""}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 8

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
    assert result.skipped[6]["index"] == 7
    assert result.skipped[6]["reason"] == "'model' is empty"
    assert result.skipped[7]["index"] == 8
    assert result.skipped[7]["reason"] == "'prompt_version' is empty"


def test_skip_items_with_empty_model(tmp_path):
    """Test that items with empty model field are skipped."""
    test_file = tmp_path / "empty_model.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "ticket": "TICKET-456", "response": "Empty model", "model": "", "prompt_version": "v1"},
        {"id": "3", "ticket": "TICKET-789", "response": "Whitespace model", "model": "  \t  ", "prompt_version": "v1"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 2

    assert result.examples[0].id == "1"
    assert result.examples[0].model == "gpt-4"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "'model' is empty"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "'model' is empty"


def test_skip_items_with_empty_prompt_version(tmp_path):
    """Test that items with empty prompt_version field are skipped."""
    test_file = tmp_path / "empty_prompt_version.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "2", "ticket": "TICKET-456", "response": "Empty prompt_version", "model": "gpt-4", "prompt_version": ""},
        {"id": "3", "ticket": "TICKET-789", "response": "Whitespace prompt_version", "model": "gpt-4", "prompt_version": "  \t  "}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 1
    assert len(result.skipped) == 2

    assert result.examples[0].id == "1"
    assert result.examples[0].prompt_version == "v1"

    assert result.skipped[0]["index"] == 1
    assert result.skipped[0]["reason"] == "'prompt_version' is empty"
    assert result.skipped[1]["index"] == 2
    assert result.skipped[1]["reason"] == "'prompt_version' is empty"


def test_mixed_valid_and_invalid_examples(tmp_path):
    """Test that valid examples are loaded while invalid ones are skipped."""
    test_file = tmp_path / "mixed.json"
    test_data = [
        {"id": "1", "ticket": "TICKET-123", "response": "Valid 1", "model": "gpt-4", "prompt_version": "v1"},
        "not an object",
        {"id": "2", "ticket": "TICKET-456", "response": "Valid 2", "model": "claude-3", "prompt_version": "v2"},
        {"ticket": "TICKET-789", "response": "Missing id", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "3", "ticket": "TICKET-000", "response": "Valid 3", "model": "gpt-3.5", "prompt_version": "v1"},
        {"id": 123, "ticket": "TICKET-111", "response": "Non-string id", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "4", "ticket": "", "response": "Empty ticket", "model": "gpt-4", "prompt_version": "v1"},
        {"id": "5", "ticket": "TICKET-222", "response": "Valid 4", "model": "gpt-4", "prompt_version": "v3"}
    ]
    test_file.write_text(json.dumps(test_data))

    result = load_examples(test_file)

    assert len(result.examples) == 4
    assert len(result.skipped) == 4

    # Check valid examples
    assert result.examples[0].id == "1"
    assert result.examples[0].ticket == "TICKET-123"
    assert result.examples[0].model == "gpt-4"
    assert result.examples[0].prompt_version == "v1"
    assert result.examples[1].id == "2"
    assert result.examples[1].ticket == "TICKET-456"
    assert result.examples[1].model == "claude-3"
    assert result.examples[1].prompt_version == "v2"
    assert result.examples[2].id == "3"
    assert result.examples[2].ticket == "TICKET-000"
    assert result.examples[2].model == "gpt-3.5"
    assert result.examples[2].prompt_version == "v1"
    assert result.examples[3].id == "5"
    assert result.examples[3].ticket == "TICKET-222"
    assert result.examples[3].model == "gpt-4"
    assert result.examples[3].prompt_version == "v3"

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
    valid_item = {"id": "1", "ticket": "TICKET-123", "response": "Test response", "model": "gpt-4", "prompt_version": "v1"}
    assert _validate_example(valid_item) is None


def test_validate_example_detects_non_object():
    """Test that _validate_example detects non-dict items."""
    assert _validate_example("string") == "Item is not an object"
    assert _validate_example(123) == "Item is not an object"
    assert _validate_example(None) == "Item is not an object"
    assert _validate_example([]) == "Item is not an object"


def test_validate_example_detects_missing_fields():
    """Test that _validate_example detects missing fields."""
    assert _validate_example({"ticket": "T", "response": "R", "model": "M", "prompt_version": "P"}) == "Missing 'id' field"
    assert _validate_example({"id": "1", "response": "R", "model": "M", "prompt_version": "P"}) == "Missing 'ticket' field"
    assert _validate_example({"id": "1", "ticket": "T", "model": "M", "prompt_version": "P"}) == "Missing 'response' field"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "prompt_version": "P"}) == "Missing 'model' field"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "model": "M"}) == "Missing 'prompt_version' field"


def test_validate_example_detects_non_string_fields():
    """Test that _validate_example detects non-string field values."""
    assert _validate_example({"id": 123, "ticket": "T", "response": "R", "model": "M", "prompt_version": "P"}) == "'id' must be a string"
    assert _validate_example({"id": "1", "ticket": 456, "response": "R", "model": "M", "prompt_version": "P"}) == "'ticket' must be a string"
    assert _validate_example({"id": "1", "ticket": "T", "response": 789, "model": "M", "prompt_version": "P"}) == "'response' must be a string"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "model": 123, "prompt_version": "P"}) == "'model' must be a string"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "model": "M", "prompt_version": 456}) == "'prompt_version' must be a string"


def test_validate_example_detects_empty_fields():
    """Test that _validate_example detects empty or whitespace-only fields."""
    assert _validate_example({"id": "", "ticket": "T", "response": "R", "model": "M", "prompt_version": "P"}) == "'id' is empty"
    assert _validate_example({"id": "1", "ticket": "", "response": "R", "model": "M", "prompt_version": "P"}) == "'ticket' is empty"
    assert _validate_example({"id": "1", "ticket": "T", "response": "", "model": "M", "prompt_version": "P"}) == "'response' is empty"
    assert _validate_example({"id": "  ", "ticket": "T", "response": "R", "model": "M", "prompt_version": "P"}) == "'id' is empty"
    assert _validate_example({"id": "1", "ticket": "  \t\n  ", "response": "R", "model": "M", "prompt_version": "P"}) == "'ticket' is empty"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "model": "", "prompt_version": "P"}) == "'model' is empty"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "model": "M", "prompt_version": ""}) == "'prompt_version' is empty"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "model": "  \t  ", "prompt_version": "P"}) == "'model' is empty"
    assert _validate_example({"id": "1", "ticket": "T", "response": "R", "model": "M", "prompt_version": "  \t  "}) == "'prompt_version' is empty"
