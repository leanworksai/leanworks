"""Static regression checks for sensitive values written to application logs."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
SOURCE_ROOTS = (REPO_ROOT / "app", REPO_ROOT / "leanworks")
LOGGER_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
SENSITIVE_NAMES = {
    "actual_user_message",
    "actual_message",
    "answer",
    "all_queries",
    "cited_context",
    "content",
    "critique_message",
    "data_sources",
    "formatted_result",
    "full_query",
    "final_text",
    "error_msg",
    "eval_explanation",
    "existing_logs",
    "log_data",
    "log_entry",
    "output",
    "params",
    "payload",
    "preview",
    "query",
    "remaining",
    "response_content",
    "response_text",
    "result",
    "result_content",
    "retry_text",
    "rewrites",
    "stderr",
    "stdout",
    "source_content",
    "text",
    "tool_input",
    "tool_result",
    "tool_results",
    "unique_sources",
    "user_message",
}
SAFE_TRANSFORMS = {"bool", "len", "type"}
SENSITIVE_ATTRIBUTES = {
    "e.response.content",
    "e.response.text",
    "response.content",
    "response.text",
    "result.stderr",
    "result.stdout",
    "self.original_user_query",
}


def _attribute_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_safe_transform(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in SAFE_TRANSFORMS
    )


def _contains_sensitive_value(node: ast.AST) -> bool:
    if _is_safe_transform(node):
        return False
    if (
        isinstance(node, ast.IfExp)
        and _is_safe_transform(node.body)
        and isinstance(node.orelse, ast.Constant)
    ):
        return False
    if isinstance(node, ast.Name):
        return node.id in SENSITIVE_NAMES
    if isinstance(node, ast.Attribute):
        if _attribute_name(node) in SENSITIVE_ATTRIBUTES:
            return True
        if _is_safe_transform(node.value):
            return False
    return any(_contains_sensitive_value(child) for child in ast.iter_child_nodes(node))


def _is_logger_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in LOGGER_METHODS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    )


def _is_console_write(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "print":
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
        and _attribute_name(node.func.value) in {"sys.stdout", "sys.stderr"}
    )


def test_logger_calls_do_not_receive_raw_sensitive_values():
    violations = []

    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not _is_logger_call(node):
                    continue
                if any(_contains_sensitive_value(arg) for arg in node.args):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert violations == [], "Raw sensitive values passed to logger: " + ", ".join(violations)


def test_application_code_has_no_direct_console_writes():
    violations = []

    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_console_write(node):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert violations == [], "Direct console writes can bypass log redaction: " + ", ".join(violations)


def test_logger_calls_do_not_emit_unfiltered_tracebacks():
    violations = []

    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not _is_logger_call(node):
                    continue
                uses_exception_method = node.func.attr == "exception"
                enables_exc_info = any(
                    keyword.arg == "exc_info"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
                if uses_exception_method or enables_exc_info:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert violations == [], "Tracebacks can contain unredacted request data: " + ", ".join(violations)
