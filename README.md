# LeanWorks

**An AI teammate that brings company knowledge and work tools together.**

[![Watch the LeanWorks product demo](https://img.youtube.com/vi/R19LnjhraIM/hqdefault.jpg)](https://youtu.be/R19LnjhraIM)

LeanWorks helps teams find the context they need, understand what is happening, and move work forward without searching across disconnected systems. People ask in plain language; LeanWorks gathers relevant information, uses the tools available to that organization, and returns a clear answer or completes the requested action.

## What LeanWorks does

LeanWorks turns scattered organizational context into useful answers and action:

- **Finds the right information.** It searches company documents, code, and prior tool results, then returns focused passages and source links.
- **Connects knowledge to live work.** It can combine stored knowledge with current projects, tasks, issues, files, meetings, and conversations.
- **Takes action where work happens.** It can create or update work through connected services instead of stopping at a recommendation.
- **Keeps the conversation coherent.** It carries forward recent messages, working context, and relevant memory across a session.

The result is one place to ask questions such as:

- “What is putting the launch at risk?”
- “Summarize the latest decisions and show me the sources.”
- “Which tasks are overdue, and who owns them?”
- “Create follow-up issues for the gaps we found.”

## How it works

```text
Ask → gather relevant context → use connected tools → answer or act
```

1. **Understand the request.** LeanWorks identifies the user’s intent and the context needed to help.
2. **Gather evidence.** It searches organizational knowledge and, when useful, queries connected systems for current information.
3. **Use the right tools.** The agent selects only the capabilities available for that organization and carries out the requested work.
4. **Return a useful result.** It responds with a concise answer, relevant sources, and the outcome of any actions it performed.

For example, when someone asks what is blocking a launch, LeanWorks can combine project records, issue status, meeting context, and internal documents into one grounded summary. If asked, it can then create or update the follow-up work in the connected project system.

## Core capabilities

| Capability | What it means for teams |
|---|---|
| Organizational search | Ask questions across documents, code, and captured tool output without knowing where the information lives. |
| Work coordination | Review projects, tasks, people, conversations, and events from one conversational interface. |
| Connected actions | Create and update work in supported external systems. |
| Context and memory | Keep recent discussion, cited material, and the active work focus available throughout a conversation. |
| Streaming responses | See answers and tool activity as the work happens. |
| Organization isolation | Scope knowledge retrieval and application data to the requesting organization. |

## Connected systems

LeanWorks supports internal project, task, user, chat, document, and search capabilities. External connections can extend the agent to:

- **Project and engineering work:** GitHub, Linear, Jira, ClickUp
- **Knowledge and files:** Notion, Google Drive, OneDrive, Google Cloud Storage
- **Communication and scheduling:** Slack, Outlook
- **People and data:** Workday, BigQuery

The tools available in a conversation depend on the integrations configured for that organization.

## What this repository provides

This repository contains the LeanWorks agent runtime and its API service. It receives requests, manages conversational context, chooses and runs tools, retrieves relevant knowledge, and streams the resulting answer back to the caller.

General document ingestion and the end-user interface are owned by the broader LeanWorks platform. This service connects to those platform capabilities rather than duplicating them.

## Developer quick start

Install the package directly from GitHub:

```bash
pip install git+https://github.com/leanworksai/leanworks.git
```

Running the API locally also requires organization credentials and access to the configured data services. Follow the development guide for a complete setup.

## Documentation

- [Technical reference](docs/technical-reference.md) — system boundaries, major components, retrieval, and memory
- [API reference](docs/api.md) — endpoints, authentication, requests, and streaming events
- [Development guide](docs/development.md) — local setup, configuration, tests, and troubleshooting
- [Deployment guide](deploy/README.md) — container and Kubernetes deployment
