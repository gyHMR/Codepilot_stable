# Codepilot Web Interface Design

## 1. Goal

Add a complete local Web workspace to Codepilot. Running `codepilot web` starts a FastAPI server and serves a React + TypeScript + Vite frontend. Users can create, resume, switch, and delete sessions; chat with the agent through streamed responses; inspect tool activity; approve or deny protected actions; cancel active runs; and recover the UI after refreshes or short connection interruptions.

The first release is local-only and uses one fixed workspace selected from the process working directory or `--workspace`. The design must leave clear extension points for authenticated remote deployment without implementing accounts, authorization, or public hosting now.

## 2. Scope

### Included

- A `codepilot web` command with configurable `--host`, `--port`, and `--workspace`.
- FastAPI REST endpoints and Server-Sent Events (SSE).
- A separately developed React + TypeScript + Vite frontend.
- Production hosting of the built frontend through FastAPI.
- Multiple Codepilot sessions within one fixed workspace.
- Historical session listing, opening, switching, and deletion.
- Streamed assistant and runtime activity display.
- Tool approval and denial, with an optional reason.
- Active-run cancellation.
- Refresh recovery, SSE reconnection, event replay, and authoritative state resynchronization.
- Safe Markdown rendering, code blocks, syntax highlighting, and copying code.
- Existing runtime commands, including planning and mode-related commands.

### Excluded from the first release

- User accounts, authentication, and multi-user authorization.
- Public Internet deployment and HTTPS termination.
- Arbitrary per-session workspace selection.
- File browsers, Git visualizations, embedded terminals, image uploads, and collaborative editing.
- Session-wide or permanent tool approval rules.
- A new permanent event database separate from the existing session persistence system.

## 3. Architectural Boundaries

Create `src/codepilot/interfaces/web/` as an interface adapter parallel to the CLI and DingTalk interfaces. The Web interface converts HTTP and SSE traffic into existing runtime intents and actions. It must not call model providers, tools, session internals, or core agent-loop functions directly.

The dependency flow is:

```text
React frontend
    -> FastAPI Web interface
    -> RuntimeGateway
    -> sessions / core / tools / llm
```

`RuntimeGateway` remains the shared application boundary. Web message submission maps to `PromptSubmitted`, runtime commands map to `CommandSubmitted`, approvals map to `ApprovalDecided`, and cancellation maps to `RunCancelled`. The Web adapter consumes the same `RuntimeFrame` stream already used by other interfaces.

Suggested backend responsibilities:

- `main.py`: parse Web command arguments and start Uvicorn.
- `app.py`: build the FastAPI application, register lifespan handling, routes, and static frontend hosting.
- `service.py`: coordinate Web session use cases, opened-session indexing, run tasks, and runtime access.
- `schemas.py`: define stable HTTP request and response models.
- `events.py`: define the Web event envelope and convert runtime frames into frontend events.
- `routes/`: keep session, action, event, and application routes focused.

FastAPI and Uvicorn should be exposed through an optional Python dependency group named `web`, so CLI-only installations do not require the Web stack. Node.js is required only for frontend development and production asset builds. Built assets are included in the Python distribution and hosted by FastAPI.

## 4. Process and Workspace Model

One Web server process owns one `RuntimeGateway` and one resolved workspace root. The workspace defaults to the directory in which `codepilot web` is launched and may be overridden with `--workspace`. It is normalized to an absolute path at startup.

Clients cannot submit workspace paths through the API. Every newly created or resumed session is forced to use the server's workspace. This prevents the browser from turning a local-only interface into an unrestricted filesystem selector.

The default bind address is `127.0.0.1`. Binding to a non-loopback address requires an explicit option and produces a prominent warning that the service has no authentication. The application structure must allow authentication middleware and session-ownership checks to be inserted later.

## 5. Session Management

The session list shows persisted sessions belonging to the fixed workspace, including the session ID, display title, creation and update times when available, current mode, and recent run status.

Supported operations are:

- Create a session.
- List historical sessions for the workspace.
- Open or resume a persisted session on demand.
- Switch sessions using a route such as `/sessions/:sessionId`.
- Delete a session record after confirmation without modifying workspace files.

Sessions are opened lazily rather than all being loaded at server startup. The Web service maintains an in-process index of sessions already opened in `RuntimeGateway`, preventing duplicate opens in one process.

The initial display title is derived from a normalized, truncated first user message. The representation leaves room for a future model-generated title, but no model title request is made in the first release.

A browser refresh restores the selected session from the URL and reloads authoritative persisted messages and runtime status. Visiting the root route selects the most recently used session when one exists, otherwise it presents the new-session state.

## 6. Chat and Runtime Actions

A session permits at most one active run. While a run is active, the composer does not submit another message or command, but cancellation and pending approval decisions remain available.

The UI represents:

- User and assistant messages.
- Incremental assistant output when exposed by progress frames.
- Thinking and progress summaries that are safe for user display.
- Tool calls and tool results.
- Approval requests.
- Paused, completed, cancelled, and failed run states.

Markdown output is sanitized and raw arbitrary HTML is not rendered. Code blocks support syntax highlighting and copying.

Commands use a dedicated command endpoint but retain existing `CommandSubmitted` behavior. The frontend may offer command completion later, but it must not duplicate command parsing or command semantics.

## 7. REST API

The initial API surface is:

```text
GET    /api/health
GET    /api/config
GET    /api/sessions
POST   /api/sessions
GET    /api/sessions/{session_id}
DELETE /api/sessions/{session_id}

GET    /api/sessions/{session_id}/messages
POST   /api/sessions/{session_id}/messages
POST   /api/sessions/{session_id}/commands
POST   /api/sessions/{session_id}/approvals/{approval_id}
POST   /api/sessions/{session_id}/cancel

GET    /api/sessions/{session_id}/events
```

The configuration response exposes only frontend-safe values such as workspace display path, model ID, permission mode, current mode, and feature flags. It never exposes API keys, provider credentials, environment variables, or unredacted internal configuration.

Message and command submission validates the session and active-run state, starts one background consumer of `RuntimeGateway.dispatch()`, and returns an accepted response containing the session ID and run correlation data available at that time. It does not hold the HTTP request open for the complete agent run.

Approval requests validate that the approval belongs to the target session and is still pending. The decision supports `approve` or `deny` and an optional reason. Cancellation is idempotent from the client's perspective: repeated cancellation requests do not start new work and return the latest known cancellation state.

Deleting an active session is rejected until its run is cancelled or completed. This avoids ambiguous cleanup and lost runtime events.

## 8. SSE Event Protocol

Each event uses a stable JSON envelope:

```json
{
  "event_id": "unique-event-id",
  "session_id": "session-id",
  "run_id": "optional-run-id",
  "type": "progress",
  "sequence": 12,
  "timestamp": "ISO-8601 timestamp",
  "data": {}
}
```

Initial event types include:

- `connected`
- `progress`
- `message_delta`
- `tool_activity`
- `approval_required`
- `run_paused`
- `run_finished`
- `command_finished`
- `cancelled`
- `failed`
- `sync_required`

The exact `data` payload is defined per event type and remains JSON serializable. Internal dataclasses and exception objects are not exposed directly.

The Web service owns one bounded in-memory event buffer per opened session. This buffer supports brief connection loss but is not a permanent source of truth. A reconnecting browser supplies `Last-Event-ID`; the server replays later buffered events in sequence. If the requested event is no longer buffered, it emits `sync_required`, and the frontend reloads session details, messages, active-run status, and pending approvals through REST.

The server sends periodic SSE keepalive comments. A disconnected SSE subscriber does not cancel an active run. Multiple subscribers may observe one session, but only one background dispatch consumer executes the run and broadcasts converted events to all subscribers.

Events are deduplicated in the frontend by `event_id`, and sequence gaps trigger authoritative resynchronization. After terminal events, the frontend invalidates REST queries for messages and session status so persisted state replaces temporary streamed state.

## 9. Approval, Cancellation, and Failure Behavior

Approval cards show the tool name, sanitized arguments, risk or policy explanation, approval ID, and optional contextual detail already provided by the runtime. Users can approve or deny and may add a reason. Refreshing the page reloads pending approvals from `RuntimeGateway.describe()` or the existing persisted session state.

Cancellation changes the UI immediately to a `cancelling` state but does not declare success until a cancellation or terminal event arrives. Cancellation maps only to `RunCancelled`; the Web layer does not directly cancel arbitrary internal tasks.

API errors use stable codes and suitable HTTP status codes. At minimum, the Web interface distinguishes:

- Invalid request payloads.
- Session not found.
- Session workspace mismatch.
- Active-run conflicts.
- Approval not found or no longer pending.
- Runtime, model, and tool failures.
- Unexpected server failures.

User-facing messages are concise, while expandable technical detail may include safe error codes and sanitized context. Secret-bearing values and full environment dumps are never returned.

## 10. Frontend Structure and State

The desktop layout uses three collapsible areas:

- Left: create session, search historical sessions, show status, and delete sessions.
- Center: message timeline, streamed output, tool and approval cards, errors, and composer.
- Right: workspace, model, mode, permission mode, session ID, active run, and pending approvals.

On narrow screens, the side areas become drawers. The top bar shows SSE connection state and provides reconnect and cancel actions.

TanStack Query owns REST-backed server state. A dedicated SSE client and reducer own connection state, replay, deduplication, sequence validation, and temporary streaming state. React Router owns the selected session route. Redux is not introduced in the first release.

Frontend modules are separated into API client, SSE client, event reducer, session components, chat components, approval components, and application pages. Styling uses CSS Modules or a small token-based stylesheet rather than a large UI framework.

## 11. Application Lifecycle and Static Hosting

FastAPI lifespan startup creates the singleton `RuntimeGateway`, Web service, event hub, and session index. Shutdown stops Web-owned background tasks, closes subscribers, cancels unfinished Web dispatch tasks through runtime-supported behavior, and closes opened runtime sessions.

In development, Vite runs separately and proxies `/api` to FastAPI. In production, FastAPI serves fingerprinted assets and returns `index.html` for non-API application routes so browser refreshes on `/sessions/:sessionId` work. Missing production assets result in a clear startup or request error explaining that the frontend must be built; they do not produce an obscure server exception.

Production same-origin hosting does not enable CORS by default. Development CORS or proxy behavior is narrowly configured for the Vite origin only.

## 12. Testing Strategy

Backend unit tests cover:

- Every `RuntimeFrame` to Web event mapping.
- JSON serialization and sanitization.
- Bounded buffering, replay, event expiration, and `sync_required`.
- Subscriber broadcasting and cleanup.
- Opened-session indexing and workspace enforcement.
- Error-code mapping.

Backend API tests use FastAPI's test client and cover:

- Health and safe configuration responses.
- Session creation, listing, opening, and deletion.
- Message and command acceptance.
- Active-run conflicts.
- Approval and cancellation behavior.
- SSE connection and replay.
- Static assets and SPA fallback without intercepting `/api` failures.

Runtime integration tests use fake model and tool ports with a real `RuntimeGateway`. They verify event order, one dispatch per submitted action, approval pause and resume, cancellation, multiple SSE subscribers, and refresh-style state recovery without contacting real providers.

Frontend tests cover:

- REST client behavior and error mapping.
- SSE reconnection, replay, deduplication, and gap handling.
- Event reducer transitions.
- Session creation and switching.
- Message submission and streamed rendering.
- Approval, denial, cancellation, and historical recovery.

Release verification includes Python tests, TypeScript type checking, frontend unit tests, a Vite production build, `/api/health`, production static hosting, and a direct SPA route refresh.

## 13. Acceptance Criteria

The feature is accepted when:

1. `codepilot web` starts on `127.0.0.1` by default and accepts host, port, and workspace options.
2. The served React application can create, list, switch, resume, and delete multiple sessions in the fixed workspace.
3. Messages and commands run through `RuntimeGateway` and appear incrementally through SSE.
4. The UI displays runtime progress, tool activity, terminal status, and sanitized failures.
5. Pending tool approvals can be approved or denied after initial delivery or a page refresh.
6. Active runs can be cancelled without starting duplicate dispatch work.
7. Refreshes and brief SSE interruptions recover through replay or authoritative REST resynchronization.
8. Multiple subscribers never cause an agent action to execute more than once.
9. Production FastAPI hosting supports frontend assets and direct navigation to session routes.
10. API responses and frontend rendering do not expose credentials or execute untrusted HTML.

## 14. Future Extension Points

Remote deployment can later add an authentication middleware, secure cookies or bearer tokens, HTTPS-aware configuration, explicit allowed origins, per-user session ownership, rate limits, audit records, and server-side authorization for workspace and tool actions. These concerns should wrap the stable REST and SSE boundaries rather than require changes to core agent execution.

The fixed-workspace service may later be generalized into an authorized workspace registry. The first release deliberately avoids accepting arbitrary paths so that this extension starts from an explicit security policy rather than an unsafe default.
