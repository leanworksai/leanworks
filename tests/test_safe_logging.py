"""Regression tests for credential-safe request logging."""

import importlib.util
import json
import logging
from pathlib import Path


# Import the pure helper without executing app/__init__.py, which intentionally
# initializes the service's external infrastructure at process startup.
MODULE_PATH = Path(__file__).parents[1] / "app" / "utils" / "safe_logging.py"
SPEC = importlib.util.spec_from_file_location("safe_logging", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SAFE_LOGGING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFE_LOGGING)
build_request_log_metadata = SAFE_LOGGING.build_request_log_metadata
build_response_log_metadata = SAFE_LOGGING.build_response_log_metadata
redact_sensitive_text = SAFE_LOGGING.redact_sensitive_text
SensitiveDataFilter = SAFE_LOGGING.SensitiveDataFilter


def test_request_log_metadata_excludes_request_contents():
    sensitive_values = {
        "user_id": "person@example.com",
        "org_slug": "private-org",
        "session_id": "session-secret-value",
        "query": "Use API_KEY=super-secret-token",
        "cited_context": {"authorization": "Bearer hidden-token"},
        "tools": ["tool-with-private-input"],
        "images": [{"url": "https://example.test/private-image"}],
    }

    metadata = build_request_log_metadata(**sensitive_values)
    serialized = json.dumps(metadata)

    for raw_value in (
        sensitive_values["user_id"],
        sensitive_values["org_slug"],
        sensitive_values["session_id"],
        sensitive_values["query"],
        "hidden-token",
        sensitive_values["tools"][0],
        sensitive_values["images"][0]["url"],
    ):
        assert raw_value not in serialized

    assert metadata == {
        "has_user_id": True,
        "has_org_slug": True,
        "has_session_id": True,
        "query_chars": len(sensitive_values["query"]),
        "has_cited_context": True,
        "cited_context_type": "dict",
        "tool_count": 1,
        "image_count": 1,
    }


def test_request_log_metadata_handles_empty_values():
    metadata = build_request_log_metadata(
        user_id=None,
        org_slug=None,
        session_id=None,
        query=None,
        cited_context=None,
        tools=None,
        images=None,
    )

    assert metadata["query_chars"] == 0
    assert metadata["tool_count"] == 0
    assert metadata["image_count"] == 0
    assert metadata["has_cited_context"] is False


def test_response_log_metadata_excludes_model_and_tool_output():
    response = {
        "content": "The pasted token is super-secret-response-value",
        "data_sources": ["https://example.test/signed?token=hidden"],
        "error": "private upstream error",
    }

    metadata = build_response_log_metadata(response, images=[{"data": "private"}])
    serialized = json.dumps(metadata)

    assert response["content"] not in serialized
    assert response["data_sources"][0] not in serialized
    assert response["error"] not in serialized
    assert metadata == {
        "response_type": "dict",
        "content_chars": len(response["content"]),
        "data_source_count": 1,
        "image_count": 1,
        "has_error": True,
    }


def test_sensitive_data_filter_redacts_credentials_and_tracebacks():
    github_token = "_".join(["github", "pat", "abcdefghijklmnopqrstuvwxyz123456"])
    private_key_header = "".join(["-----BEGIN ", "PRIVATE KEY-----"])
    private_key_footer = "".join(["-----END ", "PRIVATE KEY-----"])
    raw_values = [
        "bearer-secret-value",
        "client-secret-value",
        "query-secret-value",
        "basic-password-value",
        github_token,
        "private-key-body-value",
    ]
    message = (
        "Authorization: Bearer bearer-secret-value "
        "client_secret='client-secret-value' "
        "https://example.test/path?api_key=query-secret-value "
        "https://user:basic-password-value@example.test "
        f"{github_token} "
        f"{private_key_header}private-key-body-value{private_key_footer}"
    )

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="upstream failed: %s",
        args=(message,),
        exc_info=(ValueError, ValueError("password=traceback-secret"), None),
    )
    assert SensitiveDataFilter().filter(record) is True
    rendered = record.getMessage()

    for raw_value in raw_values:
        assert raw_value not in rendered
    assert "[REDACTED" in rendered
    assert record.exc_info is None
