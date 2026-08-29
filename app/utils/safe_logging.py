"""Helpers for application logging without exposing credentials or contents."""

import logging
import re
from typing import Any, Optional


_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
    r"-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_COOKIE_RE = re.compile(r"(?i)\b(?:set-cookie|cookie)\s*:\s*[^\r\n]+")
_BASIC_AUTH_URL_RE = re.compile(r"(https?://[^\s/:]+:)[^\s/@]+(@)", re.IGNORECASE)
_LABELED_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|"
    r"client[_-]?secret|private[_-]?key|password|passwd|secret|token)"
    r"\b[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)
_KNOWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{35}|"
    r"gh[pousr]_[0-9A-Za-z]{20,}|"
    r"github_pat_[0-9A-Za-z_]{20,}|"
    r"sk-(?:proj-)?[0-9A-Za-z_-]{20,}|"
    r"xox[baprs]-[0-9A-Za-z-]{10,}"
    r")"
)


def redact_sensitive_text(value: Any) -> str:
    """Redact common credential forms from text destined for logs."""
    text = str(value)
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    text = _COOKIE_RE.sub("Cookie: [REDACTED]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _BASIC_AUTH_URL_RE.sub(r"\1[REDACTED]\2", text)
    text = _LABELED_SECRET_RE.sub(r"\1[REDACTED]", text)
    return _KNOWN_TOKEN_RE.sub("[REDACTED TOKEN]", text)


class SensitiveDataFilter(logging.Filter):
    """Sanitize rendered records before handlers write them to stdout."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        record.msg = redact_sensitive_text(rendered)
        record.args = ()

        # Traceback formatting can reintroduce unredacted request data through
        # exception messages, locals, URLs, or upstream response bodies.
        record.exc_info = None
        record.exc_text = None
        return True


def install_sensitive_data_filter(target_logger: Optional[logging.Logger] = None) -> None:
    """Install one sensitive-data filter on each handler of a logger."""
    target = target_logger or logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(item, SensitiveDataFilter) for item in handler.filters):
            handler.addFilter(SensitiveDataFilter())


def _collection_size(value: Any) -> int:
    """Return a collection size without serializing or inspecting its values."""
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 1


def build_request_log_metadata(
    *,
    user_id: Optional[str],
    org_slug: Optional[str],
    session_id: Optional[str],
    query: Any,
    cited_context: Any,
    tools: Any,
    images: Any,
) -> dict[str, Any]:
    """Build non-sensitive request metadata suitable for application logs.

    Raw identifiers, query text, cited context, tool values, and image values are
    intentionally excluded because any of them may contain credentials or private
    customer data.
    """
    query_chars = len(query) if isinstance(query, str) else 0
    return {
        "has_user_id": bool(user_id),
        "has_org_slug": bool(org_slug),
        "has_session_id": bool(session_id),
        "query_chars": query_chars,
        "has_cited_context": cited_context is not None,
        "cited_context_type": type(cited_context).__name__ if cited_context is not None else None,
        "tool_count": _collection_size(tools),
        "image_count": _collection_size(images),
    }


def build_response_log_metadata(response: Any, images: Any = None) -> dict[str, Any]:
    """Build response metadata without serializing model or tool output."""
    if not isinstance(response, dict):
        return {
            "response_type": type(response).__name__,
            "content_chars": 0,
            "data_source_count": 0,
            "image_count": _collection_size(images),
            "has_error": False,
        }

    content = response.get("content")
    return {
        "response_type": "dict",
        "content_chars": len(content) if isinstance(content, str) else 0,
        "data_source_count": _collection_size(response.get("data_sources")),
        "image_count": _collection_size(images),
        "has_error": "error" in response,
    }
