"""Tests for the lightweight README/source drift checker."""

from pathlib import Path

from scripts.check_readme_drift import check_readme, check_repository


BASE_README = """\
# Leanworks

### API Endpoints

- **`POST /api/ask`** - Main chat endpoint

### RAG usage

```python
from leanworks.rag.chat import Chat

rag_chat = Chat(vectordb_client, model_client=model_client)
response = rag_chat.get_response("Where is the plan?")
```

```bash
GENERATION_MODEL=claude-current
GCP_VECTOR_SEARCH_COLLECTION_TEXT=leanworks-multimodal
RETRIEVE_TOP_K=20
RERANK_TOP_K=8
```
"""


ROUTE_SOURCE = """\
@app.route("/api/ask", methods=["POST"])
def ask():
    pass

@app.get("/api/docs/<string:doc_id>")
def get_doc(doc_id):
    pass
"""

RAG_CLASS_SOURCE = """\
class Chat:
    def __init__(self, vectordb_client, *, model_client=None):
        pass

    def get_response(self, query, *, top_k=20):
        pass

class AsyncChat(Chat):
    async def async_get_response(self, query, *, top_k=20):
        pass
"""

SETTINGS_SOURCE = """\
import os

GENERATION_MODEL = "claude-current"
RETRIEVE_TOP_K = 20
RERANK_TOP_K = 8
GCP_VECTOR_SEARCH_COLLECTION_TEXT = os.getenv(
    "GCP_VECTOR_SEARCH_COLLECTION_TEXT", "leanworks-multimodal"
)
"""


def _write_contract(
    tmp_path: Path,
    readme_text: str = BASE_README,
    *,
    route_source: str = ROUTE_SOURCE,
    rag_class_source: str = RAG_CLASS_SOURCE,
    settings_source: str = SETTINGS_SOURCE,
):
    readme = tmp_path / "README.md"
    routes = tmp_path / "routes.py"
    rag_classes = tmp_path / "chat.py"
    settings = tmp_path / "settings.py"

    readme.write_text(readme_text, encoding="utf-8")
    routes.write_text(route_source, encoding="utf-8")
    rag_classes.write_text(rag_class_source, encoding="utf-8")
    settings.write_text(settings_source, encoding="utf-8")
    return readme, routes, rag_classes, settings


def _check(tmp_path: Path, readme_text: str = BASE_README, **source_overrides):
    readme, routes, rag_classes, settings = _write_contract(
        tmp_path, readme_text, **source_overrides
    )
    return check_readme(
        readme,
        route_paths=[routes],
        class_source_paths=[rag_classes],
        config_paths=[settings],
    )


def test_clean_readme_matches_source_contract(tmp_path):
    assert _check(tmp_path) == []


def test_retired_pinecone_reference_is_reported(tmp_path):
    issues = _check(tmp_path, BASE_README + "\nThe active backend is Pinecone hybrid search.\n")

    assert [issue.code for issue in issues] == ["retired-pinecone-reference"]


def test_explicitly_retired_pinecone_reference_is_allowed(tmp_path):
    readme = BASE_README + "\nThere is no Pinecone fallback; GCP Vector Search is the only backend.\n"

    assert _check(tmp_path, readme) == []


def test_clearly_historical_pinecone_wording_is_allowed(tmp_path):
    historical_claims = (
        "Pinecone was replaced by GCP Vector Search.",
        "Leanworks migrated from Pinecone to GCP Vector Search.",
        "The former Pinecone backend is retained only in migration notes.",
    )

    for claim in historical_claims:
        assert _check(tmp_path, BASE_README + f"\n{claim}\n") == [], claim


def test_nonexistent_documented_route_is_reported(tmp_path):
    readme = BASE_README.replace(
        "- **`POST /api/ask`** - Main chat endpoint",
        (
            "- **`POST /api/ask`** - Main chat endpoint\n"
            "- **`POST /api/ask-stream`** - Streaming endpoint"
        ),
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["route-not-found"]
    assert "/api/ask-stream" in issues[0].message


def test_documented_http_method_must_match_route_decorator(tmp_path):
    readme = BASE_README.replace("POST /api/ask", "GET /api/ask")

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["route-method-mismatch"]
    assert "GET /api/ask" in issues[0].message
    assert "POST" in issues[0].message


def test_session_manager_endpoint_section_is_skipped_then_main_scan_resumes(tmp_path):
    readme = BASE_README.replace(
        "### RAG usage",
        (
            "#### Session Manager API Endpoints\n\n"
            "- **`POST /api/sessions/archive`** - Session-manager-only route\n\n"
            "#### Additional main-app APIs\n\n"
            "- **`POST /api/main-missing`** - Main-app handler\n\n"
            "### RAG usage"
        ),
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["route-not-found"]
    assert "/api/main-missing" in issues[0].message
    assert "/api/sessions/archive" not in issues[0].message


def test_explicitly_absent_route_is_not_treated_as_documented_endpoint(tmp_path):
    readme = BASE_README.replace(
        "- **`POST /api/ask`** - Main chat endpoint",
        (
            "- **`POST /api/ask`** - Main chat endpoint\n"
            "- Search is a tool; this service does not expose a direct `/search` route"
        ),
    )

    assert _check(tmp_path, readme) == []


def test_negated_route_clause_does_not_hide_positive_clause(tmp_path):
    readme = BASE_README.replace(
        "- **`POST /api/ask`** - Main chat endpoint",
        (
            "- **`POST /api/missing`** is planned, but `/search` is not exposed "
            "as an API route"
        ),
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["route-not-found"]
    assert "/api/missing" in issues[0].message
    assert "/search" not in issues[0].message


def test_route_parameter_syntax_is_normalized(tmp_path):
    readme = BASE_README.replace(
        "- **`POST /api/ask`** - Main chat endpoint",
        (
            "- **`POST /api/ask`** - Main chat endpoint\n"
            "- **`GET /api/docs/{doc_id}`** - Document endpoint"
        ),
    )

    assert _check(tmp_path, readme) == []


def test_invalid_rag_example_method_is_reported(tmp_path):
    readme = BASE_README.replace("rag_chat.get_response", "rag_chat.generate_response")

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["rag-method-not-found"]
    assert "generate_response" in issues[0].message


def test_constructor_unknown_keyword_is_reported(tmp_path):
    readme = BASE_README.replace("model_client=model_client", "modle_client=model_client")

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["rag-call-unknown-keyword"]
    assert "Chat()" in issues[0].message
    assert "modle_client" in issues[0].message


def test_constructor_too_many_positionals_are_reported(tmp_path):
    readme = BASE_README.replace(
        "Chat(vectordb_client, model_client=model_client)",
        "Chat(vectordb_client, firestore_client, org_slug)",
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["rag-call-too-many-positionals"]
    assert "Chat()" in issues[0].message


def test_inherited_method_signature_is_checked(tmp_path):
    readme = BASE_README.replace(
        "from leanworks.rag.chat import Chat",
        "from leanworks.rag.chat import AsyncChat",
    ).replace("Chat(vectordb_client", "AsyncChat(vectordb_client").replace(
        'get_response("Where is the plan?")',
        'get_response("Where is the plan?", topp_k=5)',
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["rag-call-unknown-keyword"]
    assert "rag_chat.get_response()" in issues[0].message
    assert "topp_k" in issues[0].message


def test_method_too_many_positionals_are_reported(tmp_path):
    readme = BASE_README.replace(
        'get_response("Where is the plan?")',
        'get_response("Where is the plan?", 5)',
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["rag-call-too-many-positionals"]
    assert "rag_chat.get_response()" in issues[0].message


def test_unexported_rag_package_import_is_reported(tmp_path):
    readme = BASE_README.replace(
        "from leanworks.rag.chat import Chat",
        "from leanworks.rag import Chat",
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["rag-import-not-exported"]


def test_documented_default_mismatch_is_reported(tmp_path):
    readme = BASE_README.replace("RETRIEVE_TOP_K=20", "RETRIEVE_TOP_K=200")

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["default-mismatch"]
    assert "RETRIEVE_TOP_K=200" in issues[0].message
    assert "source default is 20" in issues[0].message


def test_environment_backed_source_default_is_compared(tmp_path):
    readme = BASE_README.replace(
        "GCP_VECTOR_SEARCH_COLLECTION_TEXT=leanworks-multimodal",
        "GCP_VECTOR_SEARCH_COLLECTION_TEXT=leanworks-text",
    )

    issues = _check(tmp_path, readme)

    assert [issue.code for issue in issues] == ["default-mismatch"]
    assert "leanworks-multimodal" in issues[0].message


def test_missing_allowlisted_source_default_is_reported(tmp_path):
    settings = SETTINGS_SOURCE.replace("RETRIEVE_TOP_K = 20\n", "")

    issues = _check(tmp_path, settings_source=settings)

    assert [issue.code for issue in issues] == ["default-not-found"]
    assert "RETRIEVE_TOP_K" in issues[0].message


def test_unresolvable_allowlisted_source_default_is_reported(tmp_path):
    settings = SETTINGS_SOURCE.replace("RETRIEVE_TOP_K = 20", "RETRIEVE_TOP_K = default_top_k()")

    issues = _check(tmp_path, settings_source=settings)

    assert [issue.code for issue in issues] == ["default-not-found"]
    assert "could not be resolved" in issues[0].message


def test_repository_readme_matches_source_contract():
    issues = check_repository()

    assert issues == [], "\n".join(issue.message for issue in issues)
