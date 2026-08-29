# API reference

The LeanWorks API exposes the agent runtime to the rest of the platform. The primary interface is `POST /api/ask`, which can return one JSON response or a Server-Sent Events stream.

## Authentication

Protected routes accept either a Firebase bearer token or an `X-API-Key` that matches the organization API key stored in Secret Manager.

```http
Authorization: Bearer <firebase-token>
```

or

```http
X-API-Key: <organization-api-key>
```

## Main agent request

### `POST /api/ask`

Required JSON fields:

| Field | Purpose |
|---|---|
| `user_id` | Identifies the requesting user. |
| `org_slug` | Scopes data, tools, and credentials to an organization. |
| `query` | Contains the user's request. |

Common optional fields include `session_id`, `cited_context`, `tools`, `images`, and `stream`. When `stream` is `false` or omitted, the endpoint returns one JSON response. When it is `true`, the endpoint returns Server-Sent Events.

```bash
curl -N http://localhost:8082/api/ask \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-api-key' \
  -d '{
    "user_id": "user@example.com",
    "org_slug": "my-org",
    "session_id": "example-session",
    "query": "What is blocking the launch?",
    "stream": true
  }'
```

## Streaming events

The streaming response uses `data:` frames containing JSON objects. Clients should handle these event types:

| Event | Meaning |
|---|---|
| `tool_start` | A tool invocation has started. |
| `tool_end` | A tool invocation has completed. |
| `text_delta` | The next piece of answer text is available. |
| `done` | The response is complete; this event includes `data_sources`. |
| `error` | Processing failed and diagnostic information is available. |

Clients should treat unknown event types as forward-compatible additions rather than fatal errors.

## API endpoints

- **`GET /`** — liveness and readiness response
- **`GET /api/verify`** — verify the supplied API credential
- **`POST /api/ask`** — run the main agent request, with optional streaming
- **`POST /api/generate-task`** — generate a task from supplied context
- **`POST /api/doc-summary`** — generate a document summary
- **`POST /api/messages/generate-response`** — generate a response for a message workflow
- **`POST /api/plans/generate-resource-plan`** — generate a resource allocation plan
- **`POST /api/plans/generate-insights`** — generate plan insights
- **`POST /api/lean-route`** — select agents for a platform event
- **`POST /api/cache/clear`** — clear application caches

Knowledge search is an agent tool named `search_documents`; this service does not expose a direct `/search` route. Document upload is an agent tool named `upload_doc`, which delegates to the leanworks-hub `POST /api/docs/upload` API; this service does not expose a direct `/upload` route.

## Service boundaries

The API handles agent execution and returns tool activity, answer text, and sources. Document ingestion continues asynchronously in leanworks-hub, while the frontend remains responsible for persisted chat messages.

See the [technical reference](technical-reference.md) for the internal request flow.
