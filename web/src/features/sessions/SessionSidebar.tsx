import { useMemo, useState } from "react";
import { MessageSquarePlus, Search, X } from "lucide-react";
import type { SessionSummary } from "../../api/types";
import { IconButton } from "../../components/ui/IconButton";
import { SessionItem } from "./SessionItem";
import { groupSessionsByAge } from "./session-title";
import styles from "./SessionSidebar.module.css";

export function SessionSidebar({ projectName, sessions, activeId, creating, onCreate, onSelect, onRename, onDelete }: { projectName: string; sessions: SessionSummary[]; activeId?: string; creating: boolean; onCreate: () => void; onSelect: (id: string) => void; onRename: (id: string, title: string) => Promise<void>; onDelete: (id: string) => Promise<void> }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => sessions.filter(session => session.title.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())), [sessions, query]);
  const groups = useMemo(() => groupSessionsByAge(filtered), [filtered]);
  return <div className={styles.sidebarContent}>
    <header className={styles.brand}><strong>Codepilot</strong><span title={projectName}>{projectName || "当前项目"}</span></header>
    <div className={styles.controls}>
      <button className={styles.newButton} disabled={creating} onClick={onCreate}><MessageSquarePlus size={17} />新建 Session</button>
      {searchOpen ? <div className={styles.searchBox}><Search size={15} /><input autoFocus aria-label="搜索 Session" value={query} onChange={event => setQuery(event.target.value)} placeholder="搜索 Session" /><IconButton label="关闭搜索" onClick={() => { setSearchOpen(false); setQuery(""); }}><X size={15} /></IconButton></div> : <button className={styles.searchButton} onClick={() => setSearchOpen(true)}><Search size={15} />搜索 Session</button>}
    </div>
    <nav className={styles.sessionList} aria-label="Session 历史">
      {groups.map(group => <section key={group.label} className={styles.group}><h2>{group.label}</h2>{group.sessions.map(session => <SessionItem key={session.session_id} session={session} active={session.session_id === activeId} onSelect={() => onSelect(session.session_id)} onRename={title => onRename(session.session_id, title)} onDelete={() => onDelete(session.session_id)} />)}</section>)}
      {groups.length === 0 && <p className={styles.empty}>{query ? "没有匹配的 Session" : "还没有 Session"}</p>}
    </nav>
  </div>;
}
