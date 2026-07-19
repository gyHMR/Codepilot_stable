import { useEffect, useState } from "react";
import { CircleDot, Link2, Link2Off } from "lucide-react";
import type { SessionSummary } from "../../api/types";
import type { SessionEventState } from "../../events/reducer";
import styles from "./ProjectPanel.module.css";

const labels: Record<SessionEventState["runState"], string> = { idle: "空闲", running: "正在执行", paused: "等待处理", cancelling: "正在停止", failed: "执行失败" };

function elapsed(startedAt: string | null, now: number): string {
  if (!startedAt) return "";
  const seconds = Math.max(0, Math.floor((now - Date.parse(startedAt)) / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

export function RuntimeStatus({ session, events }: { session?: SessionSummary; events: SessionEventState }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { if (!events.runStartedAt) return; const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, [events.runStartedAt]);
  return <>
    <section className={styles.section}><h2>本次运行</h2><div className={styles.runState}><CircleDot size={15} /><strong>{labels[events.runState]}</strong>{events.runStartedAt && <time>{elapsed(events.runStartedAt, now)}</time>}</div></section>
    <section className={styles.section}><h2>环境</h2><dl className={styles.environment}>
      <dt>模型</dt><dd>{session?.model_id ?? "—"}</dd>
      <dt>权限</dt><dd>{session?.permission_mode ?? "—"}</dd>
      <dt>连接</dt><dd className={events.connected ? styles.success : styles.muted}>{events.connected ? <Link2 size={14} /> : <Link2Off size={14} />}{events.connected ? "已连接" : "已断开"}</dd>
    </dl></section>
  </>;
}
