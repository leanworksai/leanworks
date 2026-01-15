# LeanWorks

Internal Python package for LeanWorks AI.

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

## Features

- **Agent Tools**: Domain-specific management tools (tasks, projects, events, users, chats)
- **RAG**: Retrieval Augmented Generation for semantic search
- **Cloud Integration**: Storage and secret management
- **API-Based Architecture**: Clean separation between agent and backend

## Quick Start

### Using Domain-Specific Management Tools

The agent uses domain-specific tools (not database-specific tools) for better clarity:

```python
from leanworks.agent.tools.toolkit import ToolUse

# Initialize toolkit
tool_use = ToolUse(
    org_slug='myorg.ai',
    user_id='user@example.com',
    tools=['task_management', 'project_management']
)

# Use task management
task_tool = tool_use.task_management_tool
tasks = task_tool.query_tasks(status='completed', limit=10)
new_task = task_tool.create_task(title='New Task', priority='high')
```

### Configuration

Set environment variables:

```bash
# Local development
export LEANWORKS_HUB_URL=http://localhost:3001
export LEANWORKS_API_KEY=your_dev_api_key

# Production
export LEANWORKS_HUB_URL=https://api.leanworks.ai
export LEANWORKS_BEARER_TOKEN=your_firebase_token
```

## Available Tools

| Tool | Purpose |
|------|---------|
| `TaskManagementTool` | Manage tasks (query, create, update) |
| `ProjectManagementTool` | Query projects |
| `EventManagementTool` | Query calendar events |
| `UserManagementTool` | Query organization users |
| `ChatManagementTool` | Query chat messages |
| `DocumentManagementTool` | Manage documents |

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Agent Tools Documentation](docs/agent-tools.md)** - Complete tool reference
- **[API Setup Guide](docs/API_SETUP.md)** - Configuration and setup
- **[Examples](examples/domain_tools_usage.py)** - Usage examples

## RAG Usage

```python
from leanworks import rag
from leanworks.rag import Chat

# Basic RAG example
retriever = rag.get_retriever()
results = retriever.query("your query here")
```

## Requirements

- Python 3.10 or higher
- leanworks-hub server running (for API-based tools)
- Valid authentication (API key or bearer token)

## License

[Your chosen license] 