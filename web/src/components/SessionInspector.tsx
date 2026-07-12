import type { SessionSummary } from "../api/types";

export function SessionInspector({ session, connected }: { session?: SessionSummary; connected: boolean }) {
  return <aside className="inspector panel"><div className="section-label">RUNTIME TELEMETRY</div><dl><dt>LINK</dt><dd><span className={connected ? "signal online" : "signal"} />{connected ? "STREAMING" : "OFFLINE"}</dd><dt>MODEL</dt><dd>{session?.model_id ?? "—"}</dd><dt>MODE</dt><dd>{session?.current_mode ?? "—"}</dd><dt>ACCESS</dt><dd>{session?.permission_mode ?? "—"}</dd><dt>SESSION</dt><dd className="mono">{session?.session_id ?? "—"}</dd></dl><div className="workspace"><span>WORKSPACE</span><code>{session?.workspace ?? "No active session"}</code></div></aside>;
}
