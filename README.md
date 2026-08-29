# LeanWorks

A comprehensive AI agent framework combining intelligent task automation with Retrieval Augmented Generation (RAG) capabilities. LeanWorks provides a clean, API-based architecture that bridges AI agents with enterprise tools and knowledge management systems.

## Installation

From GitHub:
```bash
pip install git+https://github.com/leanworksai/leanworks.git
```

For development:
```bash
git clone https://github.com/leanworksai/leanworks.git
cd leanworks
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e . --no-deps
```

## Architecture Overview

LeanWorks is built on a modular, layered architecture with three main components:

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (Quart)                       │
│             POST /api/ask (JSON or SSE stream)              │
└────────┬────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │           Agent Core (ChatAgent)                      │
    │  ┌─────────────────────────────────────────────────┐ │
    │  │  • Conversation Management                      │ │
    │  │  • Tool Orchestration & Registry               │ │
    │  │  • Response Handling & Verification            │ │
    │  │  • Memory & Context Management                 │ │
    │  └─────────────────────────────────────────────────┘ │
    └────┬──────────────────┬──────────────────┬───────────┘
         │                  │                  │
    ┌────▼──────┐  ┌────────▼──────┐  ┌───────▼───────────┐
    │ SearchTool│  │  Agent Tools  │  │  Cloud Services   │
    │ GCP Vector│  │   & Integr.   │  │  & Infrastructure │
    └───────────┘  └───────────────┘  └───────────────────┘
```

### 1. **Agent Core** (`leanworks/agent/`)

The intelligent orchestration layer that manages conversations and tool execution.

#### Key Components:

**ChatAgent** (`chat.py`)
- Main conversation interface with Claude
- Manages multi-turn conversations with context awareness
- Handles tool invocation, verification, and response processing
- Loads Firestore-backed message and memory state
- Supports streaming and async operations

**Tool Registry & Orchestration** (`tool_registry.py`, `toolkit.py`)
- **ToolRegistry**: Dynamic registration and management of available tools
- **ToolUse**: Initialization and lazy-loading of tool connections
- Smart tool filtering based on workspace configuration
- Coordinates communication between agent and external services

**Response Handling** (`tool_response_handler.py`)
- Processes tool execution responses
- Formats results for Claude consumption
- Handles errors and edge cases gracefully
- Supports multiple response types (structured data, files, streaming)

**Conversation & Memory Management** (`conversation.py`, `memory.py`)
- **ConversationManager**: Tracks the active multi-turn tool/message conversation and loads prior messages
- **MemoryManager**: Persists rolling summaries, recent turns, profiles, and working context in Firestore
- Retrieves conversation history for context window
- Manages session-based conversation isolation

**Working Context** (`working_context.py`, `working_context_tool.py`)
- Maintains user's current context (active workspace, task focus)
- Provides context-aware tool execution
- Enables smarter tool recommendations

#### Available Integrations:

| Tool | Purpose | Category |
|------|---------|----------|
| **Project Management** | Query/manage projects via API | Management |
| **Task Management** | Query/create/update tasks | Management |
| **User Management** | Query organization users | Management |
| **Chat Management** | Query chat messages & threads | Management |
| **Document Management** | Manage documents & attachments | Management |
| **Search Tool** | Agent-invoked hybrid retrieval via GCP Vector Search | Search |
| **GitHub Integration** | Query repos, issues, PRs | External |
| **Linear Integration** | Query/manage Linear issues | External |
| **Notion Integration** | Query Notion databases | External |
| **Atlassian Integration** | Jira issue and user access | External |
| **ClickUp Integration** | ClickUp workspace queries | External |
| **Outlook Integration** | Calendar meetings and availability | External |
| **Cloud Storage** | GCS file operations | Infrastructure |
| **Bash Backend** | Execute shell commands | Execution |
| **Firestore** | Conversation and memory persistence | Infrastructure |
| **RAG Storage** | Low-level helper for unstructured tool-response vectors | Search |

### 2. **RAG Module** (`leanworks/rag/`)

Retrieval Augmented Generation components for hybrid search, reranking, span selection, and optional standalone answer generation.

In the main application path, RAG is exposed to `ChatAgent` as the `search_documents` tool. The main Claude model decides when to call it, receives the formatted passages as a tool result, and then writes the answer. Source links are collected separately for the API response's `data_sources` sidecar; they are not embedded in the tool-result text sent to Claude. Conversation memory is managed separately by `ChatAgent` and injected into its main prompt.

`leanworks.rag.chat.Chat` and `AsyncChat` also provide standalone `get_response`/`async_get_response` methods. They are useful for direct RAG calls, but they are not the implementation behind `/api/ask`, and they do not provide the API's SSE tool-event stream.

#### Architecture:

```
User query
    ↓
ChatAgent optionally calls search_documents
    ↓
Original query + 3 LLM rewrites
    ↓
GCP Vector Search: dense vector search + native text search
    ↓
Weighted reciprocal-rank fusion, org filters, merge/dedup
    ↓
LLM or BGE reranker (default: LLM)
    ↓
Sliding-window span selection with BM25 prefilter
    ↓
Formatted passages returned as a tool result; source links collected separately
    ↓
Main ChatAgent model generates the final answer
```

#### Key Components:

**Production Search Tool** (`leanworks/agent/tools/internal/search.py`)
- Registers as `search_documents` in the agent toolkit
- Searches one scope at a time: `docs`, `codes`, or `tool_responses`
- Applies explicit data-source, tool-name, and date filters supplied as tool arguments
- Returns formatted passages to the model and exposes source metadata separately to the agent response handler; it does not generate the final answer

**Standalone RAG Chat** (`chat.py`)
- Provides `Chat.get_response()` and `AsyncChat.async_get_response()`
- Can perform retrieval, postprocessing, prompt construction, and direct model generation
- Is separate from the `ChatAgent`/`search_documents` production path

**Vector Database** (`vectordb_client.py`, `vectordb_gcp.py`)
- Uses the `google-cloud-vectorsearch` client; GCP is the only configured backend
- Combines dense search with GCP native text search using weighted reciprocal-rank fusion
- Applies `org_slug` filtering for tenant isolation
- Routes normal documents, code, and tool responses to separate collections
- Supports 512-token chunks with 128-token overlap for local upsert helpers

**Query Processing**:
- **Query Rewriter**: Generates three diverse query rewrites by default
- **Filter Extractor**: Utility for natural-language time extraction; the production `SearchTool` currently uses explicit filter arguments instead
- **Data Source Formatter**: Formats retrieved documents for context

**Reranking** (`reranker/`)
- **LLM Reranker**: Uses Claude to score relevance
- **BGE Reranker**: Optional ONNX-based reranker; its `onnxruntime`, `transformers`, and `optimum` dependencies are not included in the shipped requirements
- **Factory Pattern**: Easy switching between rerankers

**Span Selection** (`span_selection/`)
- Uses overlapping sliding-window candidates by default (approximately 96 tokens, stride 48)
- BM25-prefilters candidates before LLM or BGE scoring
- Keeps up to four spans per document and 18 spans globally with the current `Chat` configuration

**Configuration** (`leanworks/setting.py`)
- `RETRIEVE_TOP_K`: Standalone `Chat` retrieval default (20)
- `RERANK_TOP_K`: Production `SearchTool` retrieval and reranking cap (8)
- `RECENCY_WEIGHT`: Balance between relevance and recency
- `RERANKER_TYPE` and `SPAN_SELECTION_TYPE`: Select LLM or BGE implementations; the default `llm` path works with the shipped dependencies
- These RAG/model values are Python constants, not environment-variable overrides

### 3. **API & Services Layer** (`app/`)

Quart-based REST API with cloud infrastructure integration.

#### API Endpoints:

- **`POST /api/ask`** - Main ChatAgent interface; returns JSON or Server-Sent Events when `stream=true`
- **`GET /api/verify`** - API-key verification
- **`POST /api/generate-task`**, **`POST /api/doc-summary`**, and **`POST /api/messages/generate-response`** - Specialized agent endpoints
- **`POST /api/plans/generate-resource-plan`** and **`POST /api/plans/generate-insights`** - Plan analysis endpoints
- **`POST /api/lean-route`** - Event-driven agent routing
- **`POST /api/cache/clear`** - Clear application caches
- **`GET /`** - Liveness/readiness response
- Knowledge search is an agent tool named `search_documents`; this service does not expose a direct `/search` route
- Document upload is an agent tool named `upload_doc`, which delegates to leanworks-hub's `POST /api/docs/upload`; this service does not expose a direct `/upload` route
- **Authentication** - API key validation middleware

#### Services:

**Database** (`app/services/database.py`)
- PostgreSQL client for shared user database
- Firestore client for org-specific data
- Query helpers for common operations
- Multi-tenant data isolation

**Cloud Storage** (`app/services/storage.py`)
- Google Cloud Storage integration
- File upload/download management
- Secure credential handling

**Anthropic Files** (`app/services/anthropic_files.py`)
- Client wrapper for Claude Files API integration
- File lifecycle management
- Storage quota tracking
- Separate from the leanworks-hub document-ingestion path

**Client Management** (`app/services/client.py`)
- Lazy initialization of expensive clients
- Connection pooling and caching
- Async client setup

**Authentication** (`app/auth/middleware.py`)
- API key verification
- Multi-tenant request validation

## Core Workflows

### Chat Workflow

```
1. User sends a message to POST /api/ask
   ↓
2. API initializes ChatAgent with user context
   ↓
3. Agent retrieves conversation history from Firestore
   ↓
4. Agent prepares system prompt + tools + context
   ↓
5. Claude processes user query and decides which tools to call
   ↓
6. Agent executes tool calls:
   - Validates tool permissions
   - Executes tool logic
   - Formats responses
   ↓
7. Agent passes tool results back to Claude
   ↓
8. Claude generates final response
   ↓
9. Memory state is updated; the frontend remains the source of truth for persisted chat messages
   ↓
10. Response streamed/returned to client
```

### Agentic RAG Workflow

```
1. User submits a query to POST /api/ask
   ↓
2. ChatAgent decides whether search_documents is needed
   ↓
3. SearchTool generates 3 rewrites and keeps the original query
   ↓
4. Search the selected collection scope for each query:
   - docs: leanworks-multimodal
   - codes: leanworks-codes
   - tool_responses: leanworks-tool-responses
   - combine dense and native text search with weighted RRF
   - enforce org_slug and explicit metadata filters
   ↓
5. Merge by chunk ID, deduplicate, sort, and cap at 8 candidates
   ↓
6. LLM reranker scores relevance and recency when reranking is needed
   ↓
7. Span selection creates overlapping windows, BM25-prefilters them,
   and retains the best passages
   ↓
8. Return formatted passages as a tool result and collect source links separately
   ↓
9. ChatAgent sends the tool result, conversation, and separately managed
   memory context to the main Claude model
   ↓
10. Return the final answer with a separate data_sources list
```

The standalone `Chat.get_response()` path follows the same retrieval and postprocessing primitives, but starts with `RETRIEVE_TOP_K = 20`, builds its own context prompt, and calls the generation model directly.

### Document Upload and Knowledge-Base Ingestion

```
1. ChatAgent invokes upload_doc with a workspace file path
   ↓
2. DocManagementTool validates the supported extension and resolves the file path
   ↓
3. POST the file as multipart/form-data to leanworks-hub /api/docs/upload
   ↓
4. Return document ID, type, initial processing status, and creation metadata
   ↓
5. leanworks-hub continues document processing asynchronously
```

The extraction and general knowledge-base indexing pipeline is owned by leanworks-hub and is outside this repository. This repository contains the GCP Vector Search query client and low-level upsert helpers. The in-process `RAGStorageTool` is specifically for unstructured tool responses; the current large-response path saves responses to workspace files and does not invoke that indexing helper by default.

## Configuration & Deployment

### Environment Variables

```bash
# Vector Database (Vertex AI Vector Search)
GCP_PROJECT_ID=xxx
GCP_VECTOR_SEARCH_LOCATION=us-central1
GCP_VECTOR_SEARCH_COLLECTION_TEXT=leanworks-multimodal
GCP_VECTOR_SEARCH_COLLECTION_CODES=leanworks-codes
GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES=leanworks-tool-responses
GCP_VECTOR_SEARCH_BATCH_SIZE=100
GCP_VECTOR_SEARCH_REQUEST_TIMEOUT=60

# Cloud Services
# Production override; local/dev always use leanworks-dev.
FIRESTORE_DATABASE_NAME=xxx
GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp_credential.json
# Local/dev will prefer gcp_credential_dev.json when present

# API Configuration
# Optional outbound leanworks-hub overrides.
LEANWORKS_HUB_URL=https://hub.leanworks.ai
LEANWORKS_API_KEY=xxx
```

Without a `LEANWORKS_HUB_URL` override, `get_hub_url()` uses
`http://localhost:3001` locally, `https://dev.leanworks.ai` in dev,
`https://hub.leanworks.ai` in production, and
`http://leanworks-hub-service` inside Kubernetes. `LEANWORKS_API_KEY` and
`LEANWORKS_BEARER_TOKEN` authenticate outbound hub calls. Protected incoming
routes such as `/api/ask` are validated separately by `app/auth/middleware.py`,
using a Firebase Bearer token or an `X-API-Key` matched against Secret Manager.

GCP Vector Search is always selected by `create_vectordb_client()`; there is no `USE_GCP_VECTOR_SEARCH` feature flag or Pinecone fallback.

The following defaults are Python constants in `leanworks/setting.py`, not environment variables:

```python
GENERATION_MODEL = "claude-haiku-4-5-20251001"
RERANK_MODEL = "claude-3-haiku-20240307"
RETRIEVE_TOP_K = 20       # standalone Chat default
RERANK_TOP_K = 8          # SearchTool retrieval/reranking cap
ALPHA = 0.7               # dense-search RRF weight
RECENCY_WEIGHT = 0.6
EMBEDDING_REQUESTS_PER_MINUTE = 150
EMBEDDING_MODEL = "text-embedding-004"  # Google GenAI API-key fallback only
```

With service-account credentials, `GoogleEmbedding` currently loads Vertex AI's
`text-embedding-005` instead of the fallback constant above.

### Deployment

**Docker** (`deploy/Dockerfile`)
- Production image with all dependencies
- Multi-stage build for optimization

**Kubernetes** (`deploy/`)
- Pod configuration in `deployment.yaml`
- Service routing in `service.yaml` and external routing in `consolidated_ingress.yaml`
- Cloud storage integration

**GCP Cloud Build** (`deploy/cloudbuild.yaml`)
- Builds and publishes the Bash session and ask-api container images
- Does not currently run tests or deploy Kubernetes resources

## Key Features

- **Multi-Tool Orchestration**: 15+ integrated tools for enterprise services
- **Intelligent RAG**: Hybrid search with reranking and span selection
- **Async Architecture**: Non-blocking operations for scalability
- **Multi-Tenant**: Namespace and organization-based data isolation
- **Conversation Memory**: Persistent session management with Firestore
- **File Management**: Hub-backed document upload plus Claude file-reference support
- **Server-Sent Events Streaming**: Real-time tool execution and response streaming
- **Cloud Native**: GCP-ready with Kubernetes support
- **Embedding Rate Limiting**: Client-side pacing and retry handling for embedding requests
- **Error Resilience**: Comprehensive error handling and recovery

## Quick Start

### Using Domain-Specific Management Tools

```python
from leanworks.agent.tools.toolkit import ToolUse
from google.cloud import firestore, secretmanager
from anthropic import Anthropic

# Initialize clients
firestore_client = firestore.Client()
secret_manager = secretmanager.SecretManagerServiceClient()
model_client = Anthropic()

# Initialize toolkit
tool_use = ToolUse(
    org_slug='myorg.ai',
    user_id='user@example.com',
    firestore_client=firestore_client,
    secret_manager_client=secret_manager,
    model_client=model_client,
    tools=['task_management', 'project_management', 'search']
)

# Use the unified project-management API client
project_tool = tool_use.project_management_tool
tasks = project_tool.execute_sql_query(
    "SELECT id, title, status FROM tasks WHERE status = $1 LIMIT 10",
    params=['completed'],
)
new_task = project_tool.create_task(title='New Task', priority='high')
```

### Using ChatAgent

```python
from leanworks.agent.core.chat import ChatAgent

# Initialize agent
agent = ChatAgent(
    firestore_client=firestore_client,
    secret_manager_client=secret_manager,
    model_client=model_client,
    user_id='user@example.com',
    org_slug='myorg.ai',
    session_id='conv-123',
    clear_conversation=False,
    tools=['search', 'task_management', 'project_management']
)

# Chat with the agent. Claude can call search_documents when retrieval is useful.
response = agent.process_message("What tasks are due this week?")
print(response)
```

### Using Standalone RAG Chat

The main `/api/ask` path above uses `search_documents` as an agent tool. For a direct RAG call that performs retrieval and generation without the broader tool loop, use `Chat.get_response()`:

```python
from leanworks.rag.chat import Chat
from leanworks.rag.vectordb_client import create_vectordb_client
from leanworks.rag.embedding import GoogleEmbedding

# Initialize RAG (uses GCP Vector Search)
embedding_client = GoogleEmbedding(gcp_credential_path="/path/to/credentials.json")
vectordb = create_vectordb_client(
    embedding_model_client=embedding_client,
    gcp_credential_path="/path/to/credentials.json",
)
rag_chat = Chat(
    vectordb_client=vectordb,
    firestore_client=firestore_client,
    org_slug='myorg.ai',
    model_client=model_client,
)

# Query the knowledge base and generate a direct answer
response = rag_chat.get_response(
    "How do we handle user authentication?",
    top_k=20,
    rerank_top_k=8,
)
print(response["content"])
print(response["data_sources"])
```

Omitting `user_id` and `session_id` disables the standalone class's legacy memory integration. The production `ChatAgent` manages conversation memory separately.

### Using Streaming API

The `/api/ask` endpoint supports Server-Sent Events (SSE) streaming. Enable streaming by adding `stream=true` to your request:

```python
import requests
import json

response = requests.post(
    'http://localhost:8082/api/ask',
    json={
        "user_id": "user@example.com",
        "org_slug": "my-org",
        "query": "What projects do I have?",
        "stream": True  # Enable streaming
    },
    headers={"X-API-Key": "your-api-key"},
    stream=True
)

# Process Server-Sent Events
for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        event = json.loads(line[6:].decode('utf-8'))

        if event['type'] == 'tool_start':
            print(f"🔧 {event['tool_name']}: {event['description']}")
        elif event['type'] == 'tool_end':
            print(f"✅ {event['tool_name']}: {event['summary']}")
        elif event['type'] == 'text_delta':
            print(event['text'], end='', flush=True)
        elif event['type'] == 'done':
            print(f"\n✨ Complete ({len(event['data_sources'])} sources)")
```

**Streaming Features:**
- `tool_start`: Shows which tool is executing
- `tool_end`: Shows tool completion with summary
- `text_delta`: Response text streamed incrementally
- `done`: Stream completion with data sources
- `error`: Error handling with diagnostics

The final `done` event contains the collected `data_sources` list.

## Requirements

- Python 3.10 or higher
- Google Cloud Platform account with Firestore, Secret Manager, Cloud Storage, Vertex AI, and Vector Search access
- Anthropic API key (Claude models)
- PostgreSQL/Cloud SQL for the API and management tools

## Testing Streaming

To smoke-test the streaming API with your local setup:

```bash
curl -N http://localhost:8082/api/ask \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-api-key' \
  -d '{
    "user_id": "user@example.com",
    "org_slug": "my-org",
    "session_id": "smoke-test",
    "query": "What projects do I have?",
    "stream": true
  }'
```

## Local Development Setup

### Prerequisites
- Python 3.10+
- `gcp_credential_dev.json` in project root (dev environment credentials)
- Cloud SQL Auth Proxy for database-backed API operations (startup auto-starts it when installed)

### Quick Start

1. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt -r requirements-dev.txt
   python -m pip install -e . --no-deps
   ```

2. **Configure local services:**
   ```bash
   export ENVIRONMENT=local
   export DB_HOST=127.0.0.1
   # Optional when leanworks-hub is not running on localhost:3001:
   export LEANWORKS_HUB_URL=https://dev.leanworks.ai
   ```

   Put the dev service-account file at `gcp_credential_dev.json`. The API also
   expects access to Secret Manager and PostgreSQL. When Cloud SQL Proxy is
   installed, startup attempts to launch it if port 5432 is not already open.

3. **Start the application:**
   ```bash
   python run.py
   ```

The local development server listens on `http://localhost:8082`.

### Tests

Run the project test suite in its managed virtual environment:

```bash
./scripts/run_tests.sh
```

Run only the README/source contract checks:

```bash
python -m pytest tests/test_readme_drift.py
python scripts/check_readme_drift.py
```

### Troubleshooting

- **Credentials not found**: Ensure `gcp_credential_dev.json` exists in project root
- **Database connection fails**: Check Cloud SQL Proxy is running on port 5432
- **Hub connection fails**: Run leanworks-hub on port 3001 or set `LEANWORKS_HUB_URL`

## Dependencies Overview

- **LLM**: `anthropic`, `google-genai`, `openai` - Multiple LLM provider support
- **Vector DB**: `google-cloud-vectorsearch` - GCP hybrid search backend
- **Cloud**: `google-cloud-storage`, `google-cloud-firestore`, `google-cloud-secret-manager`
- **APIs**: `requests`, `google-api-python-client`, `msal` (Microsoft auth)
- **Web**: `flask`, `quart`, `gunicorn`, `hypercorn` - API server
- **Database**: `psycopg2-binary`, `duckdb` - SQL backends
- **ML**: `tiktoken`, `numpy` - Token counting and math
- **Testing**: `pytest`, `pytest-asyncio` - Test framework
