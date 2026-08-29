#!/usr/bin/env python3
"""Check README claims that can be validated cheaply against the source tree.

The checker deliberately uses only the Python standard library.  It validates
high-value contracts without importing the application (which would require
credentials and optional runtime dependencies):

* the retired Pinecone backend is not described as current;
* documented API endpoint methods and paths match API route decorators;
* ``Chat``/``AsyncChat`` examples use valid imports, methods, and arguments; and
* selected documented configuration defaults exist and match source defaults.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import textwrap
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Only compare values that are genuine user-facing defaults.  Environment
# placeholders such as GCP_PROJECT_ID=xxx are intentionally excluded.
DOCUMENTED_DEFAULT_KEYS = frozenset(
    {
        "ALPHA",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_MODEL",
        "EMBEDDING_REQUESTS_PER_MINUTE",
        "GENERATION_MODEL",
        "GCP_VECTOR_SEARCH_BATCH_SIZE",
        "GCP_VECTOR_SEARCH_COLLECTION_CODES",
        "GCP_VECTOR_SEARCH_COLLECTION_TEXT",
        "GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES",
        "GCP_VECTOR_SEARCH_LOCATION",
        "GCP_VECTOR_SEARCH_REQUEST_TIMEOUT",
        "INCLUDE_MEMORY",
        "MIN_SCORE_THRESHOLD",
        "QUERY_REWRITES",
        "RECENCY_COEFFICIENT",
        "RECENCY_WEIGHT",
        "RERANKER_TYPE",
        "RERANK_MODEL",
        "RERANK_TOP_K",
        "RETRIEVE_TOP_K",
        "SPAN_SELECTION_TYPE",
        "SPAN_SELECTION_CONTEXT_WINDOW",
        "SPAN_SELECTION_TOP_SENTENCES",
        "USE_RERANKER",
        "USE_SPAN_SELECTION",
    }
)

_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"})
_ROUTE_DECORATOR_NAMES = {"route", "get", "post", "put", "patch", "delete", "options", "head"}
_ROUTE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.])/(?:[A-Za-z0-9_{}<>:.-]+(?:/[A-Za-z0-9_{}<>:.-]+)*)?"
)
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_PYTHON_FENCE_RE = re.compile(
    r"^```(?P<language>python|py)\s*$\n(?P<code>.*?)^```\s*$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_DOC_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>[^\n#]+?)\s*(?:#.*)?$"
)
_NEGATED_ROUTE_RE = re.compile(
    r"\b(?:does\s+not|doesn't|do\s+not|don't|never)\s+"
    r"(?:expose|implement|provide|support)\b"
    r"|\b(?:is|are)\s+not\s+(?:exposed|implemented|provided|supported|available)\b"
    r"|\bno\b.{0,80}\b(?:endpoint|route)\b"
    r"|\bnot\s+(?:an?\s+)?(?:api\s+)?(?:endpoint|route)\b",
    re.IGNORECASE,
)
_EXTERNAL_ROUTE_RE = re.compile(
    r"\bdelegates?\s+to\b"
    r"|\b(?:external|upstream)\s+(?:service|api)\b"
    r"|\bleanworks-hub\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(
    r";|[.!?](?=\s|$)|\s+[—–]\s+|,?\s+\b(?:but|while|whereas)\b\s+",
    re.IGNORECASE,
)
_MISSING = object()


@dataclass(frozen=True)
class DriftIssue:
    """One actionable mismatch between README content and source code."""

    code: str
    message: str
    line: int | None = None


@dataclass(frozen=True)
class _DocumentedRoute:
    method: str | None
    path: str
    line: int


@dataclass(frozen=True)
class _CallableSignature:
    positional: tuple[str, ...]
    keyword_names: frozenset[str]
    has_varargs: bool
    has_varkw: bool


@dataclass(frozen=True)
class _ClassInfo:
    methods: dict[str, _CallableSignature]
    bases: tuple[str, ...]


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _canonical_route(route: str) -> str:
    """Normalize Flask and OpenAPI parameter spellings for comparison."""

    route = route.strip().rstrip(",.;:")
    if route != "/":
        route = route.rstrip("/")
    route = re.sub(r"<(?:(?:[^:>]+):)?[^>]+>", "{}", route)
    route = re.sub(r"\{[^}]+\}", "{}", route)
    return route or "/"


def _route_clause(line: str, route_start: int) -> tuple[str, int]:
    """Return the prose clause containing one route and its offset in ``line``."""

    clause_start = 0
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(line):
        if boundary.start() >= route_start:
            return line[clause_start : boundary.start()], clause_start
        clause_start = boundary.end()
    return line[clause_start:], clause_start


def _documented_route_method(clause: str, route_start: int) -> str | None:
    prefix = clause[:route_start]
    match = re.search(
        r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)[\s`*_]*$",
        prefix,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def _documented_routes(readme_text: str) -> list[_DocumentedRoute]:
    """Find route-like tokens in explicit endpoint documentation.

    Route-looking filesystem examples are ignored unless their line says it is
    an endpoint/route or they appear under a heading such as "API Endpoints".
    """

    routes: list[_DocumentedRoute] = []
    endpoint_heading_level: int | None = None
    session_manager_heading_level: int | None = None

    for line_number, line in enumerate(readme_text.splitlines(), start=1):
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group("marks"))
            title = heading.group("title")
            if endpoint_heading_level is not None and level <= endpoint_heading_level:
                endpoint_heading_level = None
            if (
                session_manager_heading_level is not None
                and level <= session_manager_heading_level
            ):
                session_manager_heading_level = None

            if re.search(r"\bsession[-_\s]*manager\b", title, re.IGNORECASE):
                # Session-manager endpoints live in a separate service and are
                # intentionally not part of the app/api route inventory.
                session_manager_heading_level = level
            elif (
                session_manager_heading_level is None
                and re.search(r"\b(?:api\s+)?endpoints?\b", title, re.IGNORECASE)
            ):
                endpoint_heading_level = level

        if session_manager_heading_level is not None:
            continue
        explicit_route_context = bool(
            re.search(r"\b(?:api\s+)?(?:endpoints?|routes?)\b", line, re.IGNORECASE)
        )
        if endpoint_heading_level is None and not explicit_route_context:
            continue
        for match in _ROUTE_TOKEN_RE.finditer(line):
            clause, clause_start = _route_clause(line, match.start())
            # A single line may document a real local endpoint and contrast it
            # with an absent or upstream route.  Exclude only the route's own
            # clause so the positive claim still gets checked.
            if _NEGATED_ROUTE_RE.search(clause) or _EXTERNAL_ROUTE_RE.search(clause):
                continue
            token = match.group(0)
            # Markdown emphasis/backticks are outside the route regex, while
            # punctuation may be captured at the end.
            routes.append(
                _DocumentedRoute(
                    method=_documented_route_method(clause, match.start() - clause_start),
                    path=_canonical_route(token),
                    line=line_number,
                )
            )

    return routes


def _literal_http_methods(node: ast.expr) -> set[str] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return None
    methods = {item.upper() for item in value if isinstance(item, str)}
    return methods & _HTTP_METHODS


def _decorator_routes(decorator: ast.expr) -> set[tuple[str, str]]:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return set()
    decorator_name = decorator.func.attr.lower()
    if decorator_name not in _ROUTE_DECORATOR_NAMES:
        return set()

    route_node: ast.expr | None = decorator.args[0] if decorator.args else None
    if route_node is None:
        for keyword in decorator.keywords:
            if keyword.arg in {"path", "rule", "route"}:
                route_node = keyword.value
                break
    if not isinstance(route_node, ast.Constant) or not isinstance(route_node.value, str):
        return set()

    route = _canonical_route(route_node.value)
    if decorator_name != "route":
        return {(decorator_name.upper(), route)}

    methods: set[str] | None = None
    for keyword in decorator.keywords:
        if keyword.arg == "methods":
            methods = _literal_http_methods(keyword.value)
            break
    # Flask/Quart route decorators default to GET when methods is omitted.
    methods = {"GET"} if methods is None else methods
    return {(method, route) for method in methods}


def _source_routes(route_paths: Iterable[Path]) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for path in route_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                routes.update(_decorator_routes(decorator))
    return routes


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _callable_signature(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _CallableSignature:
    positional_only = [argument.arg for argument in function.args.posonlyargs]
    positional_or_keyword = [argument.arg for argument in function.args.args]
    decorator_names = {_call_name(decorator) for decorator in function.decorator_list}
    if "staticmethod" not in decorator_names:
        if positional_only:
            positional_only.pop(0)
        elif positional_or_keyword:
            positional_or_keyword.pop(0)
    keyword_only = [argument.arg for argument in function.args.kwonlyargs]
    return _CallableSignature(
        positional=tuple(positional_only + positional_or_keyword),
        keyword_names=frozenset(positional_or_keyword + keyword_only),
        has_varargs=function.args.vararg is not None,
        has_varkw=function.args.kwarg is not None,
    )


def _class_inventory(paths: Iterable[Path]) -> dict[str, _ClassInfo]:
    inventory: dict[str, _ClassInfo] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                item.name: _callable_signature(item)
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            bases = tuple(name for base in node.bases if (name := _base_name(base)))
            inventory[node.name] = _ClassInfo(methods=methods, bases=bases)
    return inventory


def _resolve_method_signature(
    class_name: str,
    method_name: str,
    inventory: dict[str, _ClassInfo],
    resolving: frozenset[str] = frozenset(),
) -> _CallableSignature | None:
    if class_name in resolving or class_name not in inventory:
        return None
    info = inventory[class_name]
    if method_name in info.methods:
        return info.methods[method_name]
    next_resolving = resolving | {class_name}
    for base in info.bases:
        signature = _resolve_method_signature(base, method_name, inventory, next_resolving)
        if signature is not None:
            return signature
    return None


def _call_compatibility_issues(
    call: ast.Call,
    signature: _CallableSignature,
    *,
    label: str,
    line: int,
) -> list[DriftIssue]:
    issues: list[DriftIssue] = []
    has_starred_args = any(isinstance(argument, ast.Starred) for argument in call.args)
    positional_count = sum(not isinstance(argument, ast.Starred) for argument in call.args)
    if (
        not has_starred_args
        and not signature.has_varargs
        and positional_count > len(signature.positional)
    ):
        issues.append(
            DriftIssue(
                code="rag-call-too-many-positionals",
                line=line,
                message=(
                    f"RAG example calls {label} with {positional_count} positional "
                    f"arguments, but it accepts at most {len(signature.positional)}."
                ),
            )
        )

    supplied_keywords = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    if not signature.has_varkw:
        unknown = sorted(supplied_keywords - signature.keyword_names)
        if unknown:
            issues.append(
                DriftIssue(
                    code="rag-call-unknown-keyword",
                    line=line,
                    message=(
                        f"RAG example calls {label} with unsupported keyword "
                        f"argument(s): {', '.join(unknown)}."
                    ),
                )
            )

    if not has_starred_args:
        positionally_supplied = set(signature.positional[:positional_count])
        duplicates = sorted(positionally_supplied & supplied_keywords)
        if duplicates:
            issues.append(
                DriftIssue(
                    code="rag-call-duplicate-argument",
                    line=line,
                    message=(
                        f"RAG example supplies {', '.join(duplicates)} both positionally "
                        f"and by keyword when calling {label}."
                    ),
                )
            )
    return issues


def _rag_example_method_issues(
    readme_text: str,
    class_source_paths: Iterable[Path],
) -> list[DriftIssue]:
    inventory = _class_inventory(class_source_paths)
    issues: list[DriftIssue] = []

    for fence in _PYTHON_FENCE_RE.finditer(readme_text):
        code = textwrap.dedent(fence.group("code"))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Other README examples may intentionally be fragments.  Skipping
            # those is safer than treating prose formatting as an API failure.
            continue

        fence_start_line = _line_number(readme_text, fence.start("code"))
        constructor_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            imported_classes = [
                imported.name
                for imported in node.names
                if imported.name in {"Chat", "AsyncChat"}
            ]
            if node.module == "leanworks.rag" and imported_classes:
                issues.append(
                    DriftIssue(
                        code="rag-import-not-exported",
                        line=fence_start_line + node.lineno - 1,
                        message=(
                            "RAG example imports "
                            f"{', '.join(imported_classes)} from leanworks.rag, but "
                            "those classes are exported by leanworks.rag.chat."
                        ),
                    )
                )
                continue
            if node.module != "leanworks.rag.chat":
                continue
            for imported in node.names:
                if imported.name in {"Chat", "AsyncChat"}:
                    constructor_aliases[imported.asname or imported.name] = imported.name

        instances: dict[str, str] = {}
        for node in ast.walk(tree):
            value: ast.expr | None = None
            targets: Sequence[ast.expr] = ()
            if isinstance(node, ast.Assign):
                value = node.value
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets = (node.target,)
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
                continue
            class_name = constructor_aliases.get(value.func.id)
            if not class_name:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    instances[target.id] = class_name

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            class_name = constructor_aliases.get(node.func.id)
            if not class_name:
                continue
            signature = _resolve_method_signature(class_name, "__init__", inventory)
            if signature is not None:
                issues.extend(
                    _call_compatibility_issues(
                        node,
                        signature,
                        label=f"{class_name}()",
                        line=fence_start_line + node.lineno - 1,
                    )
                )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if not isinstance(owner, ast.Name) or owner.id not in instances:
                continue
            method_name = node.func.attr
            if method_name.startswith("_"):
                continue
            class_name = instances[owner.id]
            signature = _resolve_method_signature(class_name, method_name, inventory)
            line = fence_start_line + node.lineno - 1
            if signature is None:
                issues.append(
                    DriftIssue(
                        code="rag-method-not-found",
                        line=line,
                        message=(
                            f"RAG example calls {owner.id}.{method_name}(), but "
                            f"{class_name} has no public method named {method_name}."
                        ),
                    )
                )
                continue
            issues.extend(
                _call_compatibility_issues(
                    node,
                    signature,
                    label=f"{owner.id}.{method_name}()",
                    line=line,
                )
            )

    return issues


def _is_stale_pinecone_reference(line: str) -> bool:
    """Return whether a Pinecone mention presents it as current infrastructure."""

    if not re.search(r"\bpinecone\b", line, re.IGNORECASE):
        return False
    historical_before = re.search(
        r"\b(?:no|not|never|without|removed|retired|deprecated|former|legacy|historical)\b"
        r".{0,100}\bpinecone\b"
        r"|\bmigrat(?:ed|ing|ion)\s+(?:away\s+)?from\s+pinecone\b"
        r"|\breplaced\s+pinecone\s+with\b",
        line,
        re.IGNORECASE,
    )
    historical_after = re.search(
        r"\bpinecone\b.{0,100}\b"
        r"(?:(?:is|was|has\s+been)\s+)?"
        r"(?:no\s+longer|removed|retired|deprecated|replaced\s+by)\b",
        line,
        re.IGNORECASE,
    )
    return not (historical_before or historical_after)


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _source_default(value_node: ast.expr) -> object:
    try:
        return ast.literal_eval(value_node)
    except (ValueError, TypeError):
        pass

    if not isinstance(value_node, ast.Call):
        return _MISSING

    call_name = _call_name(value_node.func)
    if call_name in {"int", "float", "str"} and value_node.args:
        inner = _source_default(value_node.args[0])
        if inner is _MISSING:
            return _MISSING
        try:
            return {"int": int, "float": float, "str": str}[call_name](inner)
        except (TypeError, ValueError):
            return _MISSING

    if call_name == "getenv" and len(value_node.args) >= 2:
        return _source_default(value_node.args[1])

    return _MISSING


def _source_defaults(config_paths: Iterable[Path]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for path in config_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            name: str | None = None
            value_node: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    value_node = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                value_node = node.value
            if name not in DOCUMENTED_DEFAULT_KEYS or value_node is None:
                continue
            defaults[name] = _source_default(value_node)
    return defaults


def _documented_value(raw_value: str) -> object:
    value = raw_value.strip()
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value


def _same_value(documented: object, source: object) -> bool:
    if isinstance(documented, bool) or isinstance(source, bool):
        return documented is source
    if isinstance(documented, (int, float)) and isinstance(source, (int, float)):
        return float(documented) == float(source)
    return str(documented) == str(source)


def _default_issues(
    readme_text: str,
    config_paths: Iterable[Path],
) -> list[DriftIssue]:
    defaults = _source_defaults(config_paths)
    issues: list[DriftIssue] = []
    for line_number, line in enumerate(readme_text.splitlines(), start=1):
        match = _DOC_ASSIGNMENT_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name not in DOCUMENTED_DEFAULT_KEYS:
            continue
        if name not in defaults:
            issues.append(
                DriftIssue(
                    code="default-not-found",
                    line=line_number,
                    message=(
                        f"README documents {name}, but no assignment for that "
                        "allowlisted default was found in the checked source files."
                    ),
                )
            )
            continue
        if defaults[name] is _MISSING:
            issues.append(
                DriftIssue(
                    code="default-not-found",
                    line=line_number,
                    message=(
                        f"README documents {name}, but its source assignment could "
                        "not be resolved to a static default."
                    ),
                )
            )
            continue
        documented = _documented_value(match.group("value"))
        source = defaults[name]
        if _same_value(documented, source):
            continue
        issues.append(
            DriftIssue(
                code="default-mismatch",
                line=line_number,
                message=(
                    f"README documents {name}={documented!s}, but the source "
                    f"default is {source!s}."
                ),
            )
        )
    return issues


def check_readme(
    readme_path: Path,
    *,
    route_paths: Iterable[Path],
    class_source_paths: Iterable[Path],
    config_paths: Iterable[Path],
) -> list[DriftIssue]:
    """Return all drift issues found for one README and source contract."""

    readme_text = readme_path.read_text(encoding="utf-8")
    issues: list[DriftIssue] = []

    for line_number, line in enumerate(readme_text.splitlines(), start=1):
        if _is_stale_pinecone_reference(line):
            issues.append(
                DriftIssue(
                    code="retired-pinecone-reference",
                    line=line_number,
                    message="README describes retired Pinecone infrastructure; Leanworks uses GCP Vector Search.",
                )
            )

    actual_routes = _source_routes(route_paths)
    actual_paths = {path for _, path in actual_routes}
    seen_documented_routes: set[tuple[str | None, str]] = set()
    for route in _documented_routes(readme_text):
        route_key = (route.method, route.path)
        if route_key in seen_documented_routes:
            continue
        seen_documented_routes.add(route_key)
        if route.path not in actual_paths:
            issues.append(
                DriftIssue(
                    code="route-not-found",
                    line=route.line,
                    message=(
                        f"README documents API route {route.path}, but no matching "
                        "route decorator exists."
                    ),
                )
            )
            continue
        if route.method is not None and (route.method, route.path) not in actual_routes:
            actual_methods = sorted(
                method for method, path in actual_routes if path == route.path
            )
            issues.append(
                DriftIssue(
                    code="route-method-mismatch",
                    line=route.line,
                    message=(
                        f"README documents {route.method} {route.path}, but source "
                        f"decorators register {', '.join(actual_methods)}."
                    ),
                )
            )

    issues.extend(_rag_example_method_issues(readme_text, class_source_paths))
    issues.extend(_default_issues(readme_text, config_paths))
    return sorted(issues, key=lambda issue: (issue.line or 0, issue.code, issue.message))


def check_repository(root: Path = REPOSITORY_ROOT, readme_path: Path | None = None) -> list[DriftIssue]:
    """Check the standard Leanworks repository layout."""

    root = root.resolve()
    readme_path = (readme_path or root / "README.md").resolve()
    route_paths = sorted((root / "app" / "api").rglob("*.py"))
    class_source_paths = [
        root / "leanworks" / "rag" / "chat.py",
        root / "leanworks" / "rag" / "filters.py",
        root / "leanworks" / "rag" / "query.py",
        root / "leanworks" / "agent" / "core" / "memory.py",
    ]
    config_paths = [
        root / "leanworks" / "setting.py",
        root / "leanworks" / "rag" / "vectordb_gcp.py",
    ]
    return check_readme(
        readme_path,
        route_paths=route_paths,
        class_source_paths=class_source_paths,
        config_paths=config_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        help="README path override (useful while reviewing generated documentation).",
    )
    args = parser.parse_args(argv)

    readme_path = args.readme or args.root / "README.md"
    try:
        issues = check_repository(args.root, readme_path)
    except (OSError, SyntaxError) as error:
        print(f"README drift check could not run: {error}", file=sys.stderr)
        return 2

    if not issues:
        print("README drift check passed.")
        return 0

    print(f"README drift check failed with {len(issues)} issue(s):", file=sys.stderr)
    for issue in issues:
        location = f"{readme_path}:{issue.line}" if issue.line else str(readme_path)
        print(f"{location}: [{issue.code}] {issue.message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
