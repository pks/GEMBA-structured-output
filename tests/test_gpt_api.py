"""Tests for gemba.gpt_api error handling."""

from unittest.mock import MagicMock, patch

import pytest

# We need to set env vars before importing GptApi since it checks them in __init__
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gemba.gpt_api import GptApi


@pytest.fixture
def gpt_api():
    """Create a GptApi instance with a mocked OpenAI client."""
    with patch("openai.OpenAI"):
        return GptApi()


class TestRequestApiContentFilter:
    """Tests for content_filter and 4xx error handling in request_api."""

    def test_content_filter_returns_empty(self, gpt_api):
        """BadRequestError with code='content_filter' should return []."""
        from openai import BadRequestError

        err = BadRequestError(
            message="content filter triggered",
            response=MagicMock(status_code=400),
            body={"code": "content_filter"},
        )
        err.code = "content_filter"
        gpt_api.call_api = MagicMock(side_effect=err)
        assert gpt_api.request_api("prompt", "gpt-4") == []

    def test_bad_request_without_content_filter_raises(self, gpt_api):
        """BadRequestError without content_filter code should raise."""
        from openai import BadRequestError

        err = BadRequestError(
            message="invalid request",
            response=MagicMock(status_code=400),
            body={"code": "invalid_request"},
        )
        err.code = "invalid_request"
        gpt_api.call_api = MagicMock(side_effect=err)
        with pytest.raises(BadRequestError):
            gpt_api.request_api("prompt", "gpt-4")

    def test_not_found_error_raises(self, gpt_api):
        """NotFoundError should raise."""
        from openai import NotFoundError

        err = NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404),
            body={},
        )
        gpt_api.call_api = MagicMock(side_effect=err)
        with pytest.raises(NotFoundError):
            gpt_api.request_api("prompt", "gpt-4")

    def test_permission_denied_raises(self, gpt_api):
        """PermissionDeniedError should raise."""
        from openai import PermissionDeniedError

        err = PermissionDeniedError(
            message="permission denied",
            response=MagicMock(status_code=403),
            body={},
        )
        gpt_api.call_api = MagicMock(side_effect=err)
        with pytest.raises(PermissionDeniedError):
            gpt_api.request_api("prompt", "gpt-4")


class TestCallApiParameters:
    """Tests for parameter construction in call_api."""

    def test_max_completion_tokens_for_new_models(self, gpt_api):
        """Newer models should use max_completion_tokens instead of max_tokens."""
        gpt_api.client = MagicMock()
        gpt_api.call_api("test prompt", "gpt-4o", temperature=0, max_tokens=500)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs

    def test_max_tokens_for_old_models(self, gpt_api):
        """Older models should use max_tokens."""
        gpt_api.client = MagicMock()
        gpt_api.call_api("test prompt", "gpt-4", temperature=0, max_tokens=500)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs

    def test_response_format_passed_through(self, gpt_api):
        """response_format should be included in API parameters when set."""
        gpt_api.client = MagicMock()
        rf = {"type": "json_schema", "json_schema": {"name": "test"}}
        gpt_api.call_api("test prompt", "gpt-4o", temperature=0, max_tokens=None, response_format=rf)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == rf

    def test_response_format_omitted_when_none(self, gpt_api):
        """response_format should not be in parameters when None."""
        gpt_api.client = MagicMock()
        gpt_api.call_api("test prompt", "gpt-4o", temperature=0, max_tokens=None)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert "response_format" not in call_kwargs
