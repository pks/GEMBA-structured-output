"""Tests for gemba.gemba_mqm_utils MQM answer parsing."""

import json

import pytest

from gemba.gemba_mqm_utils import parse_mqm_answer


class TestParseMqmAnswer:
    """Tests for parse_mqm_answer with structured JSON and text input."""

    def test_none_returns_none(self):
        assert parse_mqm_answer(None) is None

    # --- Structured JSON input (from response_format) ---

    def test_structured_json_no_errors(self):
        answer = json.dumps({"errors": {"critical": [], "major": [], "minor": []}})
        score = parse_mqm_answer(answer)
        assert score == 0

    def test_structured_json_one_major(self):
        answer = json.dumps({
            "errors": {
                "critical": [],
                "major": [{"category": "accuracy/mistranslation", "description": "wrong word"}],
                "minor": [],
            }
        })
        score = parse_mqm_answer(answer)
        assert score == -5

    def test_structured_json_one_critical(self):
        answer = json.dumps({
            "errors": {
                "critical": [{"category": "accuracy/omission", "description": "missing phrase"}],
                "major": [],
                "minor": [],
            }
        })
        score = parse_mqm_answer(answer)
        assert score == -25

    def test_structured_json_one_minor(self):
        answer = json.dumps({
            "errors": {
                "critical": [],
                "major": [],
                "minor": [{"category": "fluency/grammar", "description": "verb tense"}],
            }
        })
        score = parse_mqm_answer(answer)
        assert score == -1

    def test_structured_json_mixed_errors(self):
        answer = json.dumps({
            "errors": {
                "critical": [],
                "major": [{"category": "accuracy/mistranslation", "description": "wrong word"}],
                "minor": [{"category": "fluency/grammar", "description": "verb tense"}],
            }
        })
        score = parse_mqm_answer(answer)
        assert score == -6  # -5 (major) + -1 (minor)

    def test_structured_json_list_mqm_errors(self):
        answer = json.dumps({
            "errors": {
                "critical": [],
                "major": [{"category": "accuracy/mistranslation", "description": "wrong word"}],
                "minor": [],
            }
        })
        result = parse_mqm_answer(answer, list_mqm_errors=True)
        assert "major" in result
        assert len(result["major"]) == 1

    # --- Plain text input (legacy format) ---

    def test_text_no_errors(self):
        text = "Critical:\nno-error\nMajor:\nno-error\nMinor:\nno-error"
        score = parse_mqm_answer(text)
        assert score == 0

    def test_text_with_errors(self):
        text = "Critical:\nno-error\nMajor:\naccuracy/mistranslation - wrong word\nMinor:\nno-error"
        score = parse_mqm_answer(text)
        assert score == -5

    def test_invalid_json_falls_through_to_text_parser(self):
        """Non-JSON text should be handled by the text parser without error."""
        text = "not json {at all}"
        # Should not raise; text parser handles it (possibly with warnings)
        parse_mqm_answer(text)
