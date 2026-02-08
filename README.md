# LeanWorks

A comprehensive AI agent framework combining intelligent task automation with Retrieval Augmented Generation (RAG) capabilities. LeanWorks provides a clean, API-based architecture that bridges AI agents with enterprise tools and knowledge management systems.

## Installation

From GitHub:
```bash
pip install git+https://github.com/yourusername/leanworks.git
```

For development:
```bash
git clone https://github.com/yourusername/leanworks.git
cd leanworks
pip install -e .
```

## Architecture Overview

LeanWorks is built on a modular, layered architecture with three main components:

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (Quart/Flask)                   │
│              /chat, /upload, /search endpoints               │
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
    │   RAG     │  │  Agent Tools  │  │  Cloud Services   │
    │  Module   │  │   & Integr.   │  │  & Infrastructure │
    └───────────┘  └───────────────┘  └───────────────────┘
```

### 1. **Agent Core** (`leanworks/agent/`)

The intelligent orchestration layer that manages conversations and tool execution.

#### Key Components:

**ChatAgent** (`chat.py`)
- Main conversation interface with Claude
- Manages multi-turn conversations with context awareness
- Handles tool invocation, verification, and response processing
- Integrates with Firestore for conversation persistence
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
- **ConversationManager**: Tracks multi-turn conversations in Firestore
- **MemoryManager**: Persistent memory of interactions and context
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
| **Search Tool** | Semantic search via Firestore | Search |
| **GitHub Integration** | Query repos, issues, PRs | External |
| **Linear Integration** | Query/manage Linear issues | External |
| **Notion Integration** | Query Notion databases | External |
| **Atlassian Integration** | Jira & Confluence access | External |
| **ClickUp Integration** | ClickUp workspace queries | External |
| **Outlook Integration** | Calendar & email events | External |
| **Cloud Storage** | GCS file operations | Infrastructure |
| **Bash Backend** | Execute shell commands | Execution |
| **Firestore** | Direct database queries | Database |
| **RAG Storage** | Knowledge base retrieval | Search |

### 2. **RAG Module** (`leanworks/rag/`)

Retrieval Augmented Generation system for semantic search and knowledge extraction.

#### Architecture:

```
User Query
    ↓
Query Rewriter (Multi-query expansion)
    ↓
Pinecone Hybrid Search (BM25 + Dense embeddings)
    ↓
Reranker (BGE or LLM-based)
    ↓
Span Selection (Extract relevant sentences)
    ↓
Memory Integration (Add conversation context)
    ↓
Response Generation (Claude)
```

#### Key Components:

**Chat** (`chat.py`)
- Main RAG chat interface
- Combines Pinecone retrieval with generation
- Manages conversation context and memory
- Handles streaming responses

**Vector Database** (`vectordb.py`)
- Pinecone hybrid search client
- Supports BM25 + semantic vector search
- Namespace-based organization isolation
- Efficient batch operations

**Query Processing**:
- **Query Rewriter**: Generates diverse query rewrites for better recall
- **Filter Extractor**: Extracts search filters from natural language
- **Data Source Formatter**: Formats retrieved documents for context

**Reranking** (`reranker/`)
- **LLM Reranker**: Uses Claude to score relevance
- **BGE Reranker**: Fast ONNX-optimized reranker
- **Factory Pattern**: Easy switching between rerankers

**Span Selection** (`span_selection/`)
- Extracts relevant sentences/spans from documents
- Hybrid scoring (BM25 + embeddings)
- Context window management for optimal token usage

**Configuration** (`settings.py` in root)
- `RETRIEVE_TOP_K`: Initial retrieval count (20)
- `RERANK_TOP_K`: Final reranked results (8)
- `RECENCY_WEIGHT`: Balance between relevance and recency
- `USE_HYBRID_SPAN_SELECTION`: BM25 + semantic hybrid mode
- Embedding rate limiting and batch optimization

### 3. **API & Services Layer** (`app/`)

Flask/Quart-based REST API with cloud infrastructure integration.

#### API Endpoints:

- **`/api/ask`** - Main chat interface (supports streaming via `stream=true` parameter)
- **`/api/ask-stream`** - Server-Sent Events streaming responses with tool execution tracking
- **`/search`** - Semantic search over knowledge base
- **`/upload`** - File upload to Claude Files API + Pinecone indexing
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
- Claude Files API integration
- File lifecycle management
- Storage quota tracking

**Client Management** (`app/services/client.py`)
- Lazy initialization of expensive clients
- Connection pooling and caching
- Async client setup

**Authentication** (`app/auth/middleware.py`)
- API key verification
- Multi-tenant request validation
- Rate limiting support

## Core Workflows

### Chat Workflow

```
1. User sends message to /chat endpoint
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
9. Response saved to Firestore conversation history
   ↓
10. Response streamed/returned to client
```

### RAG Workflow

```
1. User submits query to /search or RAG chat
   ↓
2. Query Rewriter generates 3-5 diverse query variations
   ↓
3. For each variation:
   - Pinecone hybrid search (BM25 + embeddings)
   - Retrieve top 20 candidates
   - Merge and deduplicate results
   ↓
4. Reranker scores all candidates with relevance model
   ↓
5. Top-8 documents pass to Span Selection
   ↓
6. Extract 3-4 relevant sentences per document with context
   ↓
7. Merge with conversation memory for additional context
   ↓
8. Format final context prompt
   ↓
9. Claude generates response with full context
   ↓
10. Response with citations returned to user
```

### File Upload & Indexing

```
1. User uploads file via /upload endpoint
   ↓
2. Validate file size/type (MAX_FILE_SIZE_MB = 500MB)
   ↓
3. Upload to Claude Files API
   ↓
4. Create file metadata record in Firestore
   ↓
5. Extract content and chunk for Vector Search
   ↓
6. Generate embeddings (rate-limited to 150/min)
   ↓
7. Store vectors in Vertex AI Vector Search with org filters
   ↓
8. Return file ID to user
```

## Configuration & Deployment

### Environment Variables

```bash
# LLM Configuration
GENERATION_MODEL=claude-haiku-4-5-20251001
RERANK_MODEL=claude-3-haiku-20240307

# Vector Database (Vertex AI Vector Search)
USE_GCP_VECTOR_SEARCH=true
GCP_VECTOR_SEARCH_LOCATION=us-central1
GCP_VECTOR_SEARCH_COLLECTION_TEXT=leanworks-text

# Cloud Services
GCP_PROJECT_ID=xxx
FIRESTORE_DATABASE_NAME=xxx
GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp_credential.json
# Local/dev will prefer gcp_credential_dev.json when present

# API Configuration
LEANWORKS_HUB_URL=https://hub.leanworks.ai
# For dev: https://dev.leanworks.ai
# For local without hub running: set LEANWORKS_HUB_URL to dev hub or run leanworks-hub locally.
LEANWORKS_API_KEY=xxx

# RAG Parameters
RETRIEVE_TOP_K=20
RERANK_TOP_K=8
RECENCY_WEIGHT=0.6
```

### Deployment

**Docker** (`deploy/Dockerfile`)
- Production image with all dependencies
- Multi-stage build for optimization

**Kubernetes** (`deploy/deployment.yaml`)
- Scalable pod configuration
- Service exposure via ingress
- Cloud storage integration

**GCP Cloud Build** (`deploy/cloudbuild.yaml`)
- Automated build and deploy pipeline
- Testing and validation stages

## Key Features

- **Multi-Tool Orchestration**: 15+ integrated tools for enterprise services
- **Intelligent RAG**: Hybrid search with reranking and span selection
- **Async Architecture**: Non-blocking operations for scalability
- **Multi-Tenant**: Namespace and organization-based data isolation
- **Conversation Memory**: Persistent session management with Firestore
- **File Management**: Claude Files API integration with lifecycle tracking
- **Server-Sent Events Streaming**: Real-time tool execution and response streaming
- **Cloud Native**: GCP-ready with Kubernetes support
- **Rate Limiting**: Embedding and API rate limiting built-in
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

# Use task management tool
task_tool = tool_use.task_management_tool
tasks = task_tool.query_tasks(status='completed', limit=10)
new_task = task_tool.create_task(title='New Task', priority='high')
```

### Using ChatAgent

```python
from leanworks.agent.chat import ChatAgent

# Initialize agent
agent = ChatAgent(
    firestore_client=firestore_client,
    secret_manager_client=secret_manager,
    model_client=model_client,
    user_id='user@example.com',
    org_slug='myorg.ai',
    session_id='conv-123',
    tools=['search', 'task_management', 'project_management']
)

# Chat with agent
response = await agent.chat(
    user_message="What tasks are due this week?",
    additional_context="Today is Monday"
)
print(response)
```

### Using RAG Chat

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
    user_id='user@example.com',
    session_id='session-123'
)

# Query knowledge base
context = rag_chat.retrieve_context("How do we handle user authentication?")
response = rag_chat.generate_response(
    user_query="How do we handle user authentication?",
    context=context
)
```

### Using Streaming API

The `/api/ask` endpoint supports Server-Sent Events (SSE) streaming. Enable streaming by adding `stream=true` to your request:

```python
import requests
import json

response = requests.post(
    'http://localhost:8000/api/ask',
    json={
        "user_id": "user@example.com",
        "org_slug": "my-org",
        "query": "What projects do I have?",
        "stream": true  # Enable streaming
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

See [STREAMING.md](STREAMING.md) for complete documentation and [STREAMING_QUICKSTART.md](STREAMING_QUICKSTART.md) for examples.

## Requirements

- Python 3.10 or higher
- Google Cloud Platform account (Firestore, Secret Manager, Cloud Storage)
- Pinecone account for vector database
- Anthropic API key (Claude models)
- PostgreSQL for shared user database (optional)

## Testing Streaming

To test the streaming API with your local setup:

```bash
# Using the helper script (fetches API key from Secret Manager)
python3 run_streaming_test.py \
  --secret-name api-key \
  --org-slug your-org \
  --user-id your@email.com \
  --url http://localhost:8081 \
  --query "Your question here"

# Or use the test script directly
python3 test_streaming.py \
  --api-key your-api-key \
  --org-slug your-org \
  --user-id your@email.com \
  --query "Your question here"
```

See [TEST_RESULTS.md](TEST_RESULTS.md) for example test runs and [STREAMING_QUICKSTART.md](STREAMING_QUICKSTART.md) for client implementation examples.

## Local Development Setup

### Prerequisites
- Python 3.10+
- `gcp_credential_dev.json` in project root (dev environment credentials)
- Cloud SQL Proxy installed (optional, will auto-start if available)

### Quick Start

1. **Set up environment variables:**
   ```bash
   source scripts/setup-local.sh
   # Or manually:
   export ENVIRONMENT=local
   export DB_HOST=127.0.0.1
   ```

2. **Start the application:**
   ```bash
   python run.py
   ```

### Environment Variables

Copy `.env.local.example` to `.env.local` and adjust as needed:

```bash
cp .env.local.example .env.local
```

### Troubleshooting

- **Credentials not found**: Ensure `gcp_credential_dev.json` exists in project root
- **Database connection fails**: Check Cloud SQL Proxy is running on port 5432
- **Hub connection fails**: Adjust `LEANWORKS_HUB_URL` if running hub locally

## Dependencies Overview

- **LLM**: `anthropic`, `google-genai`, `openai` - Multiple LLM provider support
- **Vector DB**: `vertex-ai-vector-search` - Semantic search backend
- **Cloud**: `google-cloud-storage`, `google-cloud-firestore`, `google-cloud-secret-manager`
- **APIs**: `requests`, `google-api-python-client`, `msal` (Microsoft auth)
- **Web**: `flask`, `quart`, `gunicorn`, `hypercorn` - API server
- **Database**: `psycopg2-binary`, `duckdb` - SQL backends
- **ML**: `tiktoken`, `numpy` - Token counting and math
- **Testing**: `pytest`, `pytest-asyncio` - Test framework

## License

[Your chosen license]
