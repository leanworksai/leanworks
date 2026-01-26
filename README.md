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

- **`/chat`** - Primary chat interface (streaming responses)
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
5. Extract content and chunk for Pinecone
   ↓
6. Generate embeddings (rate-limited to 150/min)
   ↓
7. Store vectors in Pinecone with org namespace
   ↓
8. Return file ID to user
```

## Configuration & Deployment

### Environment Variables

```bash
# LLM Configuration
GENERATION_MODEL=claude-haiku-4-5-20251001
RERANK_MODEL=claude-3-haiku-20240307

# Vector Database
PINECONE_API_KEY=xxx
PINECONE_ENVIRONMENT=xxx

# Cloud Services
GCP_PROJECT_ID=xxx
FIRESTORE_DATABASE_ID=xxx
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# API Configuration
LEANWORKS_HUB_URL=https://api.leanworks.ai
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

**GCP Cloud Build** (`cloudbuild.yaml`)
- Automated build and deploy pipeline
- Testing and validation stages

## Key Features

- **Multi-Tool Orchestration**: 15+ integrated tools for enterprise services
- **Intelligent RAG**: Hybrid search with reranking and span selection
- **Async Architecture**: Non-blocking operations for scalability
- **Multi-Tenant**: Namespace and organization-based data isolation
- **Conversation Memory**: Persistent session management with Firestore
- **File Management**: Claude Files API integration with lifecycle tracking
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
from leanworks.rag.vectordb import PineconeHybridIndex

# Initialize RAG
vectordb = PineconeHybridIndex(index_name='myindex')
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

## Requirements

- Python 3.10 or higher
- Google Cloud Platform account (Firestore, Secret Manager, Cloud Storage)
- Pinecone account for vector database
- Anthropic API key (Claude models)
- PostgreSQL for shared user database (optional)

## Dependencies Overview

- **LLM**: `anthropic`, `google-genai`, `openai` - Multiple LLM provider support
- **Vector DB**: `pinecone` - Semantic search backend
- **Cloud**: `google-cloud-storage`, `google-cloud-firestore`, `google-cloud-secret-manager`
- **APIs**: `requests`, `google-api-python-client`, `msal` (Microsoft auth)
- **Web**: `flask`, `quart`, `gunicorn`, `hypercorn` - API server
- **Database**: `psycopg2-binary`, `duckdb` - SQL backends
- **ML**: `tiktoken`, `numpy` - Token counting and math
- **Testing**: `pytest`, `pytest-asyncio` - Test framework

## License

[Your chosen license]