"""Tests for gemba.prompt parsing functions."""

import pytest

from gemba.prompt import parse_numerical_answer, validate_number, validate_stars


class TestParseNumericalAnswer:
    """Tests for parse_numerical_answer with JSON and extended fallbacks."""

    # --- Structured JSON output ---

    def test_structured_json(self):
        assert parse_numerical_answer('{"score": 98}') == 98

    def test_structured_json_zero(self):
        assert parse_numerical_answer('{"score": 0}') == 0

    def test_json_without_score_key_falls_through(self):
        # No "score" key, so JSON path doesn't match.
        # But the string contains a single number (42), so the original regex picks it up.
        assert parse_numerical_answer('{"value": 42}') == 42

    def test_invalid_json_falls_through(self):
        assert parse_numerical_answer("not json at all") is None

    # --- Original single-number logic ---

    def test_single_number(self):
        assert parse_numerical_answer("85") == 85

    def test_bracket_format(self):
        assert parse_numerical_answer("['72']") == 72

    def test_fraction_exact(self):
        assert parse_numerical_answer("85/100", max=100) == 85

    # --- Extended fallbacks for verbose model responses ---

    def test_markdown_bold_single_number(self):
        assert parse_numerical_answer("**78**") == 78

    def test_markdown_bold_fraction(self):
        assert parse_numerical_answer("**85/100**", max=100) == 85

    def test_fraction_in_longer_text(self):
        assert parse_numerical_answer("Score: 85/100 - good", max=100) == 85

    def test_two_number_heuristic(self):
        assert parse_numerical_answer("**92/100**", max=100) == 92

    def test_verbose_response_returns_none(self):
        """Many numbers and no clear pattern should return None."""
        verbose = "Score: 98\n\nJustification:\nMeaning preservation: 100\nGrammar: 98\n2012\n20"
        assert parse_numerical_answer(verbose, max=100) is None


class TestValidateNumber:
    """Tests for validate_number (range-checked wrapper)."""

    def test_in_range(self):
        assert validate_number('{"score": 50}') == 50

    def test_above_range(self):
        assert validate_number('{"score": 200}') is None

    def test_below_range(self):
        assert validate_number('{"score": -5}') is None


class TestValidateStars:
    """Tests for validate_stars with markdown stripping."""

    def test_star_characters(self):
        assert validate_stars("***") == 3

    def test_word_form(self):
        assert validate_stars("five stars") == 5

    def test_digit_form(self):
        assert validate_stars("3 stars") == 3

    def test_markdown_bold_does_not_add_stars(self):
        """Markdown bold wrapping should be stripped, not counted as stars."""
        # "**3 stars**" should parse as "3 stars" = 3, not count ** as stars
        assert validate_stars("**3 stars**") == 3

    def test_out_of_range(self):
        assert validate_stars("10 stars") is None
