# Technical reference

This page explains the main system boundaries and implementation paths for maintainers. Start with the [project overview](../README.md) for the product story.

## System boundary

This repository provides the LeanWorks agent runtime and ask API. It owns conversation orchestration, tool selection and execution, knowledge retrieval, response streaming, and agent memory.

The surrounding LeanWorks platform owns the end-user interface and the general document-ingestion pipeline. Document uploads initiated here are handed to `leanworks-hub` for asynchronous processing.

```text
Client
  ↓
Quart API
  ↓
ChatAgent
  ├─ organizational search
  ├─ internal management tools
  ├─ connected external tools
  └─ isolated workspace execution
  ↓
Answer, sources, and tool events
```

## Main request flow

1. `POST /api/ask` validates the caller and request context.
2. `ChatAgent` loads recent conversation, memory, and working context.
3. The model decides whether it needs organizational search or another tool.
4. `ToolUse` exposes the internal tools and the integrations enabled for the organization.
5. Tool results are normalized and returned to the model for the next reasoning step.
6. The final answer is returned as JSON or streamed as Server-Sent Events.
7. Memory state is updated. The frontend remains the source of truth for persisted chat messages.

## Major components

### Agent runtime

The code in `leanworks/agent/` manages the conversational tool loop:

- `core/chat.py` coordinates model calls, tools, streaming, and response assembly.
- `core/conversation.py` manages the active multi-turn exchange.
- `core/memory.py` maintains rolling summaries, recent turns, profiles, and working context.
- `tools/toolkit.py` initializes internal and external tools only when they are needed.
- `tools/tool_response_handler.py` converts tool output into model-ready results and API events.

Internal tools cover search, projects, tasks, users, chat, documents, working context, and agent triggering. Connected tools add services such as GitHub, Linear, Jira, ClickUp, Notion, Outlook, Slack, Google Drive, OneDrive, Workday, BigQuery, and Cloud Storage.

### Knowledge retrieval

In the production path, knowledge retrieval is a tool named `search_documents`; it is not a separate public API route. The agent decides when retrieval is useful and then writes the final answer from the returned passages.

```text
Original query and three rewrites
  ↓
Dense and native text search
  ↓
Rank fusion, organization filters, merge, and deduplication
  ↓
Optional reranking
  ↓
Focused span selection
  ↓
Passages for the model + source links for the API response
```

Search operates against GCP Vector Search and scopes results by `org_slug`. Documents, code, and captured tool responses use separate collections. The production tool searches one scope at a time and applies explicit source, tool-name, and date filters supplied by the model.

Reranking is skipped for very small or already high-confidence result sets. When span selection runs, it uses overlapping windows and a BM25 prefilter before choosing the strongest passages.

### Context and memory

Conversation state and retrieved knowledge are separate concerns:

- Conversation management tracks the active model-and-tool exchange.
- Memory provides summaries, recent turns, user or organization context, and the current work focus.
- Knowledge retrieval supplies evidence for the current request.
- Source links are collected separately from the passage text and returned in `data_sources`.

### Document upload

The `upload_doc` agent tool validates a workspace file and sends it to the `leanworks-hub` upload API. The hub returns initial metadata and continues extraction and indexing asynchronously. This repository includes the query client and low-level vector upsert helpers, but does not own the general ingestion pipeline.

## Standalone retrieval API

`leanworks.rag.chat.Chat` and `AsyncChat` expose direct retrieval-and-generation methods for callers that do not need the broader agent tool loop. They are separate from the `/api/ask` implementation and do not emit its tool-event stream.

```python
from leanworks.rag.chat import Chat

rag_chat = Chat(
    vectordb_client=vectordb,
    firestore_client=firestore_client,
    org_slug="my-org",
    model_client=model_client,
)

response = rag_chat.get_response(
    "How do we handle user authentication?",
    top_k=20,
    rerank_top_k=8,
)

print(response["content"])
print(response["data_sources"])
```

Pass `user_id` and `session_id` to enable the standalone class's legacy memory integration. The production agent manages memory independently.

## Selected defaults

These values are Python constants in `leanworks/setting.py`, not environment-variable overrides:

```python
GENERATION_MODEL = "claude-haiku-4-5-20251001"
RERANK_MODEL = "claude-3-haiku-20240307"
RETRIEVE_TOP_K = 20
RERANK_TOP_K = 8
ALPHA = 0.7
RECENCY_WEIGHT = 0.6
EMBEDDING_REQUESTS_PER_MINUTE = 150
EMBEDDING_MODEL = "text-embedding-004"
```

With service-account credentials, `GoogleEmbedding` currently uses Vertex AI's `text-embedding-005`; the constant above is the Google GenAI API-key fallback.

## Repository map

| Path | Responsibility |
|---|---|
| `app/api/` | HTTP routes and specialized request handlers |
| `app/auth/` | Incoming request authentication |
| `app/services/` | Database, storage, and client setup |
| `leanworks/agent/` | Agent loop, memory, tools, and workspace execution |
| `leanworks/rag/` | Retrieval, reranking, span selection, and standalone RAG |
| `deploy/` | Container, Kubernetes, and build configuration |

See the [API reference](api.md), [development guide](development.md), and [deployment guide](../deploy/README.md) for operational details.
