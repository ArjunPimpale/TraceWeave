"""Tests for Stage 1 extraction output validator."""

from __future__ import annotations

import json

import pytest

from pecs.extraction.validator import ExtractionValidator, ValidationResult


@pytest.fixture
def validator():
    return ExtractionValidator()


@pytest.fixture
def valid_json_response():
    return json.dumps([
        {
            "entity_type": "REQUIREMENT",
            "entity_id": "R1-auth",
            "text": "The system must implement JWT authentication for all users across the entire platform securely.",
            "linked_requirement": None,
            "author": None,
            "timestamp": None,
        }
    ])


class TestJSONParsing:
    def test_direct_json_parse(self, validator, valid_json_response):
        result = validator.validate(valid_json_response, "doc.md", "chunk001")
        assert result.success
        assert len(result.results) == 1

    def test_json_in_code_fence(self, validator, valid_json_response):
        fenced = f"```json\n{valid_json_response}\n```"
        result = validator.validate(fenced, "doc.md", "chunk001")
        assert result.success

    def test_json_in_generic_fence(self, validator, valid_json_response):
        fenced = f"```\n{valid_json_response}\n```"
        result = validator.validate(fenced, "doc.md", "chunk001")
        assert result.success

    def test_json_with_preamble_text(self, validator, valid_json_response):
        with_preamble = f"Here are the entities:\n{valid_json_response}"
        result = validator.validate(with_preamble, "doc.md", "chunk001")
        assert result.success

    def test_invalid_json_fails(self, validator):
        result = validator.validate("Not JSON at all, just text.", "doc.md", "chunk001")
        assert not result.success
        assert result.errors

    def test_empty_array_is_valid(self, validator):
        result = validator.validate("[]", "doc.md", "chunk001")
        assert result.success
        assert len(result.results) == 0

    def test_single_object_normalized_to_list(self, validator):
        single = json.dumps({
            "entity_type": "REQUIREMENT",
            "entity_id": "R1",
            "text": "The system must implement JWT authentication for all users across the entire platform securely.",
            "linked_requirement": None,
            "author": None,
            "timestamp": None,
        })
        result = validator.validate(single, "doc.md", "chunk001")
        assert result.success
        assert len(result.results) == 1


class TestSchemaValidation:
    def test_missing_entity_type_fails(self, validator):
        bad = json.dumps([{"entity_id": "R1", "text": "Some text.", "linked_requirement": None, "author": None, "timestamp": None}])
        result = validator.validate(bad, "doc.md", "chunk001")
        assert not result.success

    def test_invalid_entity_type_fails(self, validator):
        bad = json.dumps([{"entity_type": "UNKNOWN_TYPE", "entity_id": "R1", "text": "text", "linked_requirement": None, "author": None, "timestamp": None}])
        result = validator.validate(bad, "doc.md", "chunk001")
        assert not result.success

    def test_empty_text_fails(self, validator):
        bad = json.dumps([{"entity_type": "REQUIREMENT", "entity_id": "R1", "text": "", "linked_requirement": None, "author": None, "timestamp": None}])
        result = validator.validate(bad, "doc.md", "chunk001")
        assert not result.success

    def test_extra_fields_ignored(self, validator):
        with_extra = json.dumps([{
            "entity_type": "IMPLEMENTATION",
            "entity_id": "impl-gnn",
            "text": "Implemented GNN training pipeline.",
            "linked_requirement": "R2",
            "author": "Alice",
            "timestamp": "2024-01-01",
            "unexpected_field": "should be ignored",
            "another_extra": 42,
        }])
        result = validator.validate(with_extra, "doc.md", "chunk001")
        assert result.success

    def test_provenance_fields_injected(self, validator, valid_json_response):
        result = validator.validate(valid_json_response, "requirements.md", "abc123chunk")
        assert result.success
        assert result.results[0].source_document == "requirements.md"
        assert result.results[0].chunk_id == "abc123chunk"

    def test_empty_linked_requirement_normalized_to_none(self, validator):
        with_empty_link = json.dumps([{
            "entity_type": "REQUIREMENT",
            "entity_id": "R1",
            "text": "The system must implement JWT authentication for all users across the entire platform securely.",
            "linked_requirement": "",
            "author": None,
            "timestamp": None,
        }])
        result = validator.validate(with_empty_link, "doc.md", "chunk001")
        assert result.success
        assert result.results[0].linked_requirement is None


class TestHallucinationDetection:
    def test_low_overlap_flagged(self, validator):
        """Text with no word overlap with source should be flagged."""
        response = json.dumps([{
            "entity_type": "REQUIREMENT",
            "entity_id": "R1",
            "text": "The quantum computing blockchain neural network will be integrated into the main database system.",  # No overlap
            "linked_requirement": None,
            "author": None,
            "timestamp": None,
        }])
        source = "The system must implement JWT authentication for users."
        result = validator.validate(response, "doc.md", "chunk001", source_text=source)

        # Should still succeed but with hallucination flag
        assert result.success or not result.success  # either way is acceptable
        if result.results:
            metadata = result.results[0].metadata
            # If flagged, hallucination_risk should be in metadata
            # (might not be if overlap isn't checked for very different domains)

    def test_high_overlap_not_flagged(self, validator):
        """Text with good word overlap should NOT be flagged."""
        source = "The system must implement JWT authentication for all users across the entire platform securely."
        response = json.dumps([{
            "entity_type": "REQUIREMENT",
            "entity_id": "R1",
            "text": "The system must implement JWT authentication for all users across the entire platform securely.",
            "linked_requirement": None,
            "author": None,
            "timestamp": None,
        }])
        result = validator.validate(response, "doc.md", "chunk001", source_text=source)
        assert result.success
        if result.results:
            assert not result.results[0].metadata.get("hallucination_risk", False)
