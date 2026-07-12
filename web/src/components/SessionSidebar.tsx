import type { SessionSummary } from "../api/types";

export function SessionSidebar({ sessions, activeId, onCreate, onSelect, onDelete }: { sessions: SessionSummary[]; activeId?: string; onCreate: () => void; onSelect: (id: string) => void; onDelete: (id: string) => void }) {
  return <aside className="sidebar panel">
    <div className="brand"><span className="brand-mark">CP</span><div><strong>CODEPILOT</strong><small>LOCAL WEB DECK</small></div></div>
    <button className="new-session" onClick={onCreate}>＋ NEW SESSION</button>
    <div className="section-label">SESSIONS / {sessions.length.toString().padStart(2, "0")}</div>
    <nav className="session-list">{sessions.map(session => <button key={session.session_id} className={session.session_id === activeId ? "session active" : "session"} onClick={() => onSelect(session.session_id)}>
      <span className={session.is_running ? "status live" : "status"} /><span className="session-copy"><b>{session.session_id.slice(0, 14)}</b><small>{session.current_mode} · {session.message_count} messages</small></span>
      <span className="delete" role="button" aria-label="Delete session" onClick={event => { event.stopPropagation(); onDelete(session.session_id); }}>×</span>
    </button>)}</nav>
  </aside>;
}
