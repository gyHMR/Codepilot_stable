# Codepilot Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local `codepilot web` workspace with multi-session management, React chat UI, SSE streaming, approvals, cancellation, and refresh recovery.

**Architecture:** A new FastAPI adapter under `codepilot.interfaces.web` translates REST requests into existing `RuntimeGateway` actions and broadcasts normalized runtime frames through per-session SSE event hubs. A React + TypeScript + Vite application consumes authoritative REST state plus replayable SSE events; FastAPI hosts its production build without changing core agent execution.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, Pydantic, pytest, React 18, TypeScript, Vite, React Router, TanStack Query, Vitest, Testing Library, EventSource/SSE.

## Global Constraints

- Preserve dependency direction: `protocols → llm/tools → core → sessions/observability → extensions → runtime → interfaces`.
- The Web interface may use `RuntimeGateway`, `SessionOpenIntent`, and runtime actions but must not call model, tool, core loop, or session-controller internals directly.
- Default bind address is `127.0.0.1`; non-loopback binding emits a no-authentication warning.
- One server process uses one absolute workspace; HTTP clients cannot select arbitrary filesystem paths.
- FastAPI and Uvicorn are optional dependencies under the `web` extra.
- Source text uses UTF-8 and LF; Python file operations involving Chinese text specify `encoding="utf-8"`.
- Use TDD for every production behavior: add a failing test, observe the expected failure, implement minimally, then rerun.
- Do not add or commit `benchmarks/` content.
- Do not disturb unrelated changes already present in the working tree.

---

## File Structure

### Backend

- `src/codepilot/interfaces/web/__init__.py`: public Web adapter exports.
- `src/codepilot/interfaces/web/main.py`: `codepilot web` argument handling and Uvicorn startup.
- `src/codepilot/interfaces/web/app.py`: FastAPI factory, lifespan, exception handling, API registration, and SPA hosting.
- `src/codepilot/interfaces/web/schemas.py`: Pydantic request/response DTOs and stable error payloads.
- `src/codepilot/interfaces/web/events.py`: Web event envelope, runtime-frame conversion, bounded replay buffer, and subscriber hub.
- `src/codepilot/interfaces/web/service.py`: fixed-workspace session coordination and one-dispatch-per-action background execution.
- `src/codepilot/interfaces/web/routes/health.py`: health and safe configuration endpoints.
- `src/codepilot/interfaces/web/routes/sessions.py`: session CRUD and message history.
- `src/codepilot/interfaces/web/routes/actions.py`: messages, commands, approvals, and cancellation.
- `src/codepilot/interfaces/web/routes/events.py`: per-session SSE endpoint.
- `src/codepilot/interfaces/cli/main.py`: register and dispatch the `web` subcommand.
- `pyproject.toml`: optional Web dependencies and packaged frontend assets.

### Frontend

- `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/index.html`: frontend toolchain.
- `web/src/main.tsx`, `web/src/App.tsx`: application bootstrap and routes.
- `web/src/api/types.ts`, `web/src/api/client.ts`: REST contracts and client.
- `web/src/events/client.ts`, `web/src/events/reducer.ts`: SSE lifecycle and streamed state.
- `web/src/state/queryClient.ts`: TanStack Query configuration.
- `web/src/components/SessionSidebar.tsx`: create, search, switch, delete.
- `web/src/components/ChatTimeline.tsx`: persisted and streamed messages.
- `web/src/components/ApprovalCard.tsx`: approve/deny interaction.
- `web/src/components/Composer.tsx`: message/command submission and cancellation state.
- `web/src/components/SessionInspector.tsx`: workspace/model/mode/permission/run metadata.
- `web/src/pages/WorkspacePage.tsx`: route-level orchestration.
- `web/src/styles/*.css`: design tokens and responsive three-panel layout.

### Tests

- `test/test_web_events.py`: serialization, replay, and broadcast.
- `test/test_web_service.py`: session coordination and dispatch ownership.
- `test/test_web_api.py`: FastAPI routes, errors, SSE, and static hosting.
- `test/test_cli_web.py`: CLI parser and Web startup wiring.
- `web/src/**/*.test.ts(x)`: frontend unit and component tests.

---

### Task 1: Add the optional Web package boundary and CLI command

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/codepilot/interfaces/cli/main.py`
- Create: `src/codepilot/interfaces/web/__init__.py`
- Create: `src/codepilot/interfaces/web/main.py`
- Test: `test/test_cli_web.py`

**Interfaces:**
- Produces: `WebServerOptions(host: str, port: int, workspace: Path, reload: bool)`.
- Produces: `run_web_server(options: WebServerOptions) -> None`.
- Produces: `codepilot web --host HOST --port PORT --workspace PATH [--reload]`.

- [ ] **Step 1: Write failing parser tests**

```python
def test_web_subcommand_defaults_to_localhost_and_current_workspace():
    from codepilot.interfaces.cli.main import build_parser

    args = build_parser().parse_args(["web"])

    assert args.command == "web"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.workspace == "."
    assert args.reload is False


def test_web_subcommand_accepts_server_options():
    from codepilot.interfaces.cli.main import build_parser

    args = build_parser().parse_args(
        ["web", "--host", "0.0.0.0", "--port", "9000", "--workspace", "repo", "--reload"]
    )

    assert (args.host, args.port, args.workspace, args.reload) == (
        "0.0.0.0", 9000, "repo", True
    )
```

- [ ] **Step 2: Run the parser tests and observe the missing command failure**

Run: `pytest test/test_cli_web.py -q`

Expected: FAIL because `web` is not a registered subcommand.

- [ ] **Step 3: Add optional dependencies and the CLI branch**

Add to `pyproject.toml`:

```toml
web = [
  "fastapi>=0.115.0,<1",
  "uvicorn[standard]>=0.30.0,<1",
]
```

Register `web` in `build_parser()` and branch before `build_session_intent(args)` in `_run_from_args()` so Web startup does not open a CLI session:

```python
if args.command == "web":
    from codepilot.interfaces.web.main import WebServerOptions, run_web_server

    run_web_server(
        WebServerOptions(
            host=args.host,
            port=args.port,
            workspace=Path(args.workspace),
            reload=args.reload,
        )
    )
    return 0
```

Implement `WebServerOptions` as a frozen dataclass that resolves `workspace` to an absolute path and validates port range `1..65535`. `run_web_server()` imports Uvicorn lazily and calls it with the future factory `codepilot.interfaces.web.app:create_app_from_env`, passing host, port, reload, and factory mode.

- [ ] **Step 4: Add startup wiring tests**

Monkeypatch `codepilot.interfaces.web.main.run_web_server` and assert `_run_from_args()` passes an absolute workspace. Add a test that non-loopback `host` writes a warning containing `no authentication` to stderr.

- [ ] **Step 5: Run focused tests**

Run: `pytest test/test_cli_web.py test/test_cli_refactor.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/codepilot/interfaces/cli/main.py src/codepilot/interfaces/web test/test_cli_web.py
git commit -m "feat: add codepilot web command"
```

---

### Task 2: Define stable Web schemas and runtime event conversion

**Files:**
- Create: `src/codepilot/interfaces/web/schemas.py`
- Create: `src/codepilot/interfaces/web/events.py`
- Test: `test/test_web_events.py`

**Interfaces:**
- Produces: `WebEvent(event_id, session_id, run_id, type, sequence, timestamp, data)`.
- Produces: `runtime_frame_to_event(frame, *, session_id, sequence, event_id_factory, clock) -> WebEvent`.
- Produces: `EventHub(capacity: int = 256)` with `publish`, `subscribe`, `unsubscribe`, and `replay_after`.
- Produces: Pydantic DTOs `ApiError`, `MessageCreate`, `CommandCreate`, `ApprovalCreate`, `SessionCreate`, `AcceptedAction`.

- [ ] **Step 1: Write failing frame-conversion tests**

Create one parametrized test covering `ProgressFrame`, `ApprovalRequiredFrame`, `RunPausedFrame`, `RunFinishedFrame`, `CommandFinishedFrame`, `CancelledFrame`, and `FailedFrame`. Assert stable event types, session/run correlation, JSON-safe `data`, and no raw exception objects.

```python
event = runtime_frame_to_event(
    ProgressFrame(event={"type": "text_delta", "delta": "hi"}),
    session_id="s1",
    sequence=3,
    event_id_factory=lambda: "evt-3",
    clock=lambda: "2026-07-12T00:00:00Z",
)
assert event.model_dump() == {
    "event_id": "evt-3",
    "session_id": "s1",
    "run_id": None,
    "type": "message_delta",
    "sequence": 3,
    "timestamp": "2026-07-12T00:00:00Z",
    "data": {"type": "text_delta", "delta": "hi"},
}
```

- [ ] **Step 2: Run and observe import failure**

Run: `pytest test/test_web_events.py::test_runtime_frame_conversion -q`

Expected: FAIL because `events.py` does not exist.

- [ ] **Step 3: Implement DTOs and conversion**

Use explicit conversion functions rather than exposing `dataclasses.asdict()` recursively without filtering. Map progress events with `type == "text_delta"` to `message_delta`; other progress frames map to `progress` or `tool_activity` when the progress type begins with `tool_`. Derive run IDs from frame records when present.

- [ ] **Step 4: Write failing replay and broadcast tests**

Test that a capacity-2 hub retains only the last two events, replays strictly after a supplied event ID, returns a `ReplayResult(expired=True, events=())` for an evicted ID, and sends one published event to two independent subscriber queues.

- [ ] **Step 5: Implement the bounded `EventHub`**

Use `collections.deque(maxlen=capacity)`, a monotonic per-session sequence counter, and one bounded `asyncio.Queue[WebEvent]` per subscriber. If a slow subscriber queue is full, remove that subscriber and let its SSE generator terminate with `sync_required`; never block runtime dispatch on a browser.

- [ ] **Step 6: Run focused tests**

Run: `pytest test/test_web_events.py -q`

Expected: PASS with coverage of all runtime frame classes.

- [ ] **Step 7: Commit**

```bash
git add src/codepilot/interfaces/web/schemas.py src/codepilot/interfaces/web/events.py test/test_web_events.py
git commit -m "feat: define web event protocol"
```

---

### Task 3: Build the fixed-workspace Web service

**Files:**
- Create: `src/codepilot/interfaces/web/service.py`
- Modify: `src/codepilot/sessions/service.py`
- Modify: `src/codepilot/sessions/repository.py`
- Test: `test/test_web_service.py`
- Test: `test/test_sessions_persistence_v2.py`

**Interfaces:**
- Produces: `WebService(runtime: RuntimeGateway, workspace: Path, event_capacity: int = 256)`.
- Produces async methods: `list_sessions`, `create_session`, `get_session`, `delete_session`, `messages`, `submit_prompt`, `submit_command`, `decide_approval`, `cancel`, `shutdown`.
- Produces: `events_for(session_id) -> EventHub` and `ensure_open(session_id) -> str`.

- [ ] **Step 1: Write failing workspace and open-index tests**

Use a fake gateway recording `SessionOpenIntent`. Assert that `create_session()` and `ensure_open()` always use the service workspace, and two calls to `ensure_open("s1")` call `open_session()` once.

- [ ] **Step 2: Run and observe the missing service failure**

Run: `pytest test/test_web_service.py::test_sessions_use_fixed_workspace -q`

Expected: FAIL because `WebService` does not exist.

- [ ] **Step 3: Implement session projections using public runtime/session APIs**

Create `SessionSummary` and `SessionDetail` schema projections. Add `SessionService.list_sessions(*, workspace_root: str) -> tuple[SessionView, ...]` and `SessionRepository.list_session_ids() -> tuple[str, ...]`; the service loads each candidate through existing repository methods and filters by normalized workspace root. Add a focused persistence test. The Web interface must not read `.codepilot` files directly.

`ensure_open()` calls:

```python
self.runtime.open_session(
    SessionOpenIntent(workspace_dir=self.workspace, session_id=session_id)
)
```

and caches only the returned session ID. Validate that the described session workspace resolves to the fixed workspace; otherwise close it and raise `WebConflict("web.workspace_mismatch", ...)`.

- [ ] **Step 4: Write failing one-dispatch-owner tests**

Submit one prompt, assert the method returns immediately with `AcceptedAction`, and assert a second prompt during the active task raises `WebConflict("runtime.run_active")`. Verify two EventHub subscribers receive the same frames while the fake gateway's `dispatch()` is called once.

- [ ] **Step 5: Implement background dispatch ownership**

Maintain `_tasks: dict[str, asyncio.Task[None]]`. A private `_start_dispatch(session_id, action)` creates one task. The task consumes `runtime.dispatch()`, converts and publishes every frame, and removes itself in `finally`. Do not infer completion from SSE connection state.

- [ ] **Step 6: Write failing approval, cancellation, deletion, and shutdown tests**

Assert approvals map to `ApprovalDecided`, cancellation maps to `RunCancelled`, active-session deletion raises a conflict, idle deletion uses the public session service and closes an opened runtime session, and shutdown cancels/awaits owned tasks then calls `runtime.close_all()`.

- [ ] **Step 7: Implement the remaining service methods**

Validate pending approval IDs against `runtime.describe(session_id).pending_approvals`. Make cancellation idempotent by returning current status when a cancellation task is already in progress. Never directly cancel runtime internal tasks.

- [ ] **Step 8: Run focused and runtime tests**

Run: `pytest test/test_web_service.py test/test_runtime_gateway_v2.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/codepilot/interfaces/web/service.py src/codepilot/sessions test/test_web_service.py test/test_sessions_persistence_v2.py
git commit -m "feat: add web session service"
```

---

### Task 4: Expose FastAPI REST routes and stable errors

**Files:**
- Create: `src/codepilot/interfaces/web/app.py`
- Create: `src/codepilot/interfaces/web/routes/__init__.py`
- Create: `src/codepilot/interfaces/web/routes/health.py`
- Create: `src/codepilot/interfaces/web/routes/sessions.py`
- Create: `src/codepilot/interfaces/web/routes/actions.py`
- Test: `test/test_web_api.py`

**Interfaces:**
- Produces: `create_app(*, workspace: Path, runtime: RuntimeGateway | None = None, frontend_dir: Path | None = None) -> FastAPI`.
- Produces: JSON error shape `{"error": {"code": str, "message": str, "details": object | null}}`.

- [ ] **Step 1: Write failing health and config tests**

Assert `/api/health` returns `{"status": "ok"}` and `/api/config` returns only workspace, model, permission mode, current mode, and feature flags. Seed an environment API key and assert it is absent from the serialized response.

- [ ] **Step 2: Run and observe missing app failure**

Run: `pytest test/test_web_api.py::test_health_and_safe_config -q`

Expected: FAIL because `create_app` does not exist.

- [ ] **Step 3: Implement the app factory, lifespan, and error mapping**

Store `WebService` on `app.state.web_service`. Register handlers for schema validation, `KeyError`, `WebConflict`, and unexpected exceptions. Unexpected errors log server-side and return `web.internal_error` without traceback or environment data. Lifespan calls `service.shutdown()`.

- [ ] **Step 4: Write failing session route tests**

Cover list, create, detail, message history, delete, not found, workspace mismatch, and deleting an active session. Assert status codes `200/201/404/409` and stable error codes.

- [ ] **Step 5: Implement session routes**

Keep route functions thin: validate DTO, call one service method, return schema projection. Use `201` for session creation and `204` for successful deletion.

- [ ] **Step 6: Write failing action route tests**

Cover message `202`, command `202`, approval `202`, cancellation `202`, blank input `422`, missing approval `404`, and active-run conflict `409`.

- [ ] **Step 7: Implement action routes**

Map request schemas only to runtime actions through `WebService`; route modules must not import session controllers or model/tool ports.

- [ ] **Step 8: Run focused tests**

Run: `pytest test/test_web_api.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/codepilot/interfaces/web/app.py src/codepilot/interfaces/web/routes test/test_web_api.py
git commit -m "feat: expose web REST API"
```

---

### Task 5: Add SSE delivery, replay, keepalive, and resynchronization

**Files:**
- Create: `src/codepilot/interfaces/web/routes/events.py`
- Modify: `src/codepilot/interfaces/web/app.py`
- Modify: `test/test_web_api.py`
- Modify: `test/test_web_service.py`

**Interfaces:**
- Produces: `GET /api/sessions/{session_id}/events` with media type `text/event-stream`.
- Consumes: `Last-Event-ID` request header.
- Produces SSE events with `id`, `event`, and JSON `data` fields.

- [ ] **Step 1: Write a failing replay test**

Publish three events into a capacity-3 hub, connect with `Last-Event-ID` equal to the first ID, and assert the response emits only events two and three in order.

- [ ] **Step 2: Run and observe 404**

Run: `pytest test/test_web_api.py::test_sse_replays_after_last_event_id -q`

Expected: FAIL with HTTP 404.

- [ ] **Step 3: Implement the SSE generator**

Use `StreamingResponse`. Serialize each event as:

```text
id: <event_id>
event: <type>
data: <single-line JSON>

```

Replay buffered events before subscribing to live events. Use `asyncio.wait_for(queue.get(), timeout=15)` and emit `: keepalive\n\n` on timeout. Always unsubscribe in `finally`.

- [ ] **Step 4: Write failing expiration and disconnect tests**

Assert an evicted `Last-Event-ID` produces `sync_required`; closing the client removes the subscriber; disconnecting does not call cancel; and two clients see one background dispatch sequence.

- [ ] **Step 5: Implement expiration and cleanup behavior**

Emit a generated `sync_required` event with the current highest sequence and reason `event_history_expired`. Subscriber overflow uses reason `subscriber_overflow`.

- [ ] **Step 6: Run API and service tests**

Run: `pytest test/test_web_api.py test/test_web_service.py -q`

Expected: PASS without leaked asyncio-task warnings.

- [ ] **Step 7: Commit**

```bash
git add src/codepilot/interfaces/web/routes/events.py src/codepilot/interfaces/web/app.py test/test_web_api.py test/test_web_service.py
git commit -m "feat: stream replayable web events"
```

---

### Task 6: Scaffold the React application and typed clients

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/api/types.ts`
- Create: `web/src/api/client.ts`
- Create: `web/src/api/client.test.ts`
- Create: `web/src/state/queryClient.ts`

**Interfaces:**
- Produces: `api.listSessions`, `api.createSession`, `api.getSession`, `api.getMessages`, `api.deleteSession`, `api.submitMessage`, `api.submitCommand`, `api.decideApproval`, `api.cancelRun`.
- Produces typed `ApiError` with HTTP status, code, message, and details.

- [ ] **Step 1: Create the minimal Vite test toolchain**

Use React 18, React Router 6, TanStack Query 5, `react-markdown`, `rehype-sanitize`, Vitest, jsdom, and Testing Library. Add scripts `dev`, `build`, `typecheck`, and `test`.

- [ ] **Step 2: Write failing API client tests**

Mock `fetch` and assert correct methods/paths/bodies, `204` handling, and conversion of the stable backend error envelope into `ApiError`.

- [ ] **Step 3: Run and observe missing client failures**

Run from `web`: `npm test -- --run src/api/client.test.ts`

Expected: FAIL because client functions do not exist.

- [ ] **Step 4: Implement types and REST client**

Use one internal `request<T>()` helper, JSON headers only when a body exists, and `AbortSignal` support. Keep backend field names unchanged rather than adding a second camelCase protocol.

- [ ] **Step 5: Bootstrap routing and query provider**

Routes are `/` and `/sessions/:sessionId`. Root renders a session landing state until Task 8 adds selection behavior.

- [ ] **Step 6: Verify frontend foundation**

Run from `web`:

```bash
npm test -- --run
npm run typecheck
npm run build
```

Expected: all commands PASS and `web/dist/index.html` exists.

- [ ] **Step 7: Commit**

```bash
git add web
git commit -m "feat: scaffold web frontend"
```

---

### Task 7: Implement SSE client and deterministic event state

**Files:**
- Create: `web/src/events/reducer.ts`
- Create: `web/src/events/reducer.test.ts`
- Create: `web/src/events/client.ts`
- Create: `web/src/events/client.test.ts`

**Interfaces:**
- Produces: `SessionEventState` with connection, last event ID, last sequence, streaming text, tool activity, pending approvals, run state, and `needs_sync`.
- Produces: `reduceWebEvent(state, event) -> state`.
- Produces: `SessionEventClient(sessionId, callbacks)` with `connect()` and `close()`.

- [ ] **Step 1: Write failing reducer tests**

Cover connected, message delta concatenation, tool start/completion, approval addition/removal, cancellation, terminal refresh flag, duplicate event IDs, sequence gaps, and `sync_required`.

- [ ] **Step 2: Run and observe missing reducer failure**

Run from `web`: `npm test -- --run src/events/reducer.test.ts`

Expected: FAIL.

- [ ] **Step 3: Implement the pure reducer**

Store a bounded set of recent event IDs. Ignore duplicates. If `sequence > last_sequence + 1`, set `needs_sync=true` before applying only safe connection metadata; do not append potentially incomplete message deltas.

- [ ] **Step 4: Write failing EventSource lifecycle tests**

Use a fake `EventSource`. Assert correct URL, event parsing, connection/error callbacks, reconnect with exponential delays capped at 10 seconds, and no reconnect after `close()`.

- [ ] **Step 5: Implement the SSE client**

Because native browser `EventSource` cannot set `Last-Event-ID` manually, persist the last ID as a `last_event_id` query parameter for explicit reconnects while also accepting native automatic reconnect headers. The backend events route must accept header first and query parameter second.

- [ ] **Step 6: Run frontend event tests**

Run from `web`: `npm test -- --run src/events`

Expected: PASS with fake timers restored after each test.

- [ ] **Step 7: Commit**

```bash
git add web/src/events src/codepilot/interfaces/web/routes/events.py test/test_web_api.py
git commit -m "feat: add resilient web event client"
```

---

### Task 8: Build session navigation and the chat workspace

**Files:**
- Create: `web/src/pages/WorkspacePage.tsx`
- Create: `web/src/pages/WorkspacePage.test.tsx`
- Create: `web/src/components/SessionSidebar.tsx`
- Create: `web/src/components/ChatTimeline.tsx`
- Create: `web/src/components/Composer.tsx`
- Create: `web/src/components/ApprovalCard.tsx`
- Create: `web/src/components/SessionInspector.tsx`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes all REST client and event-state interfaces from Tasks 6-7.
- Produces route-driven session selection and accessible controls for all first-release actions.

- [ ] **Step 1: Write failing session-navigation tests**

Render with a memory router. Assert root selects the most recently updated session, an empty list shows `Create your first session`, creation navigates to `/sessions/<id>`, search filters visible sessions, and confirmed deletion navigates away from the removed session.

- [ ] **Step 2: Implement the session sidebar and route orchestration**

Use TanStack Query keys `['sessions']`, `['session', id]`, and `['messages', id]`. Do not copy REST state into component-local state.

- [ ] **Step 3: Write failing chat and streaming tests**

Assert persisted messages render, Markdown raw HTML is sanitized, deltas append to one temporary assistant bubble, completion invalidates messages, active run disables submission, and cancel remains enabled.

- [ ] **Step 4: Implement timeline and composer**

Message text is rendered through `react-markdown` plus `rehype-sanitize`. Treat input beginning with `/` as a command and all other nonblank input as a message. Clear the composer only after a `202` response.

- [ ] **Step 5: Write failing approval and inspector tests**

Assert tool name, sanitized arguments, risk explanation, approve/deny buttons, optional reason, current workspace/model/mode/permission, SSE status, run state, and session ID are visible.

- [ ] **Step 6: Implement approval and inspector components**

Disable an approval card after submitting a decision until authoritative status or a new event arrives. Never use `dangerouslySetInnerHTML` for arguments or errors.

- [ ] **Step 7: Run frontend component tests**

Run from `web`:

```bash
npm test -- --run
npm run typecheck
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add web/src
git commit -m "feat: build web chat workspace"
```

---

### Task 9: Add responsive styling and production asset hosting

**Files:**
- Create: `web/src/styles/tokens.css`
- Create: `web/src/styles/app.css`
- Modify: `web/src/main.tsx`
- Modify: `src/codepilot/interfaces/web/app.py`
- Modify: `pyproject.toml`
- Modify: `test/test_web_api.py`

**Interfaces:**
- Produces: three-panel desktop layout and drawer-style narrow layout.
- Produces: FastAPI SPA fallback for non-API GET routes.

- [ ] **Step 1: Write failing static-hosting tests**

With a temporary frontend directory containing `index.html` and `assets/app.js`, assert `/assets/app.js` returns the asset, `/sessions/s1` returns `index.html`, `/api/missing` stays JSON 404, and a missing build directory returns a clear Web UI build error.

- [ ] **Step 2: Implement static hosting after API routes**

Mount `/assets` and add a final GET fallback that rejects paths beginning with `/api/`. Resolve packaged assets with `importlib.resources`; allow `frontend_dir` injection for tests.

- [ ] **Step 3: Configure package data**

Add `scripts/build_web.ps1` that runs `npm ci`, runs `npm run build`, resolves and verifies the exact `src/codepilot/interfaces/web/static/` target, clears only that directory, and copies `web/dist/` into it. Configure setuptools package data for `static/**/*` and commit the packaged static assets so `pip install ".[web]"` runs without Node. Add `web/node_modules/` and `web/dist/` to `.gitignore`; never add them to Git.

- [ ] **Step 4: Add responsive, accessible styling**

Define color, spacing, typography, focus, and status tokens. At widths below 900px, side panels become overlay drawers. Respect `prefers-reduced-motion`; maintain visible keyboard focus and semantic button labels.

- [ ] **Step 5: Run static and frontend verification**

Run:

```bash
pytest test/test_web_api.py -q
cd web
npm run typecheck
npm test -- --run
npm run build
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml scripts/build_web.ps1 src/codepilot/interfaces/web/app.py src/codepilot/interfaces/web/static web/src/styles web/src/main.tsx test/test_web_api.py
git commit -m "feat: serve production web workspace"
```

---

### Task 10: End-to-end runtime integration and final verification

**Files:**
- Create: `test/test_web_runtime_integration.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-12-web-interface-design.md` only if implementation revealed an approved factual correction

**Interfaces:**
- Verifies the complete public boundary from HTTP action through real `RuntimeGateway` to SSE and persisted recovery.

- [ ] **Step 1: Write the failing end-to-end integration test**

Create a real `RuntimeGateway` with fake model/tool ports. Through the FastAPI app:

1. Create a session.
2. Open SSE.
3. Submit a prompt.
4. Observe progress and `run_finished` in order.
5. Reload messages and assert the assistant response is persisted.
6. Reconnect after the first event and assert no duplicate execution.

- [ ] **Step 2: Add approval and cancellation integration cases**

Use a fake protected tool to reach `approval_required`, approve it through REST, and observe continuation. Use a blocking fake model for cancellation and assert one terminal cancellation outcome with no leaked active run.

- [ ] **Step 3: Fix only boundary defects exposed by the tests**

Keep fixes within Web adapter/runtime public-contract integration. If a runtime public API is genuinely missing, add the smallest public method and a runtime regression test; do not reach into private controller fields from Web code.

- [ ] **Step 4: Document installation and operation**

Add exact commands:

```bash
pip install -e ".[web]"
codepilot web
codepilot web --workspace E:/path/to/repo --port 8000
```

Document local-only security, frontend development commands, production build requirements, and the non-loopback warning.

- [ ] **Step 5: Run complete verification**

Run from the repository root:

```bash
pytest -q
cd web
npm ci
npm run typecheck
npm test -- --run
npm run build
```

Then launch `codepilot web --host 127.0.0.1 --port 8765` and verify:

```text
GET http://127.0.0.1:8765/api/health -> 200 {"status":"ok"}
GET http://127.0.0.1:8765/ -> 200 HTML
GET http://127.0.0.1:8765/sessions/example -> 200 HTML
```

- [ ] **Step 6: Review the diff for architectural and security compliance**

Confirm no Web module imports `codepilot.core`, model provider implementations, tool executors, or private session-controller symbols. Search responses and frontend state for secret-bearing configuration fields. Confirm all workspace inputs originate from server options.

- [ ] **Step 7: Commit**

```bash
git add README.md test/test_web_runtime_integration.py
git commit -m "test: verify web workspace end to end"
```

---

## Final Review Checklist

- [ ] Every new backend behavior was introduced by a failing pytest.
- [ ] Every frontend state transition was introduced by a failing Vitest test.
- [ ] One user action creates one runtime dispatch consumer regardless of SSE subscriber count.
- [ ] Refresh and replay do not duplicate messages or tool actions.
- [ ] Missing/expired events force authoritative resynchronization.
- [ ] Workspace paths cannot be supplied through HTTP requests.
- [ ] Non-loopback binding visibly warns about missing authentication.
- [ ] Markdown and technical error rendering do not execute arbitrary HTML.
- [ ] API/config responses contain no credentials or environment values.
- [ ] Python, TypeScript, frontend tests, production build, health endpoint, and SPA fallback all pass.
