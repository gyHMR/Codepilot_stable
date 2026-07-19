import type { SessionSummary } from "../../api/types";

export type SessionGroup = { label: string; sessions: SessionSummary[] };

export function groupSessionsByAge(sessions: SessionSummary[], now = new Date()): SessionGroup[] {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const groups: SessionGroup[] = [
    { label: "今天", sessions: [] },
    { label: "最近 7 天", sessions: [] },
    { label: "更早", sessions: [] },
  ];
  [...sessions].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at)).forEach(session => {
    const timestamp = Date.parse(session.updated_at);
    const age = Number.isFinite(timestamp) ? today - timestamp : Number.POSITIVE_INFINITY;
    const target = age < 86_400_000 ? groups[0] : age < 7 * 86_400_000 ? groups[1] : groups[2];
    target.sessions.push(session);
  });
  return groups.filter(group => group.sessions.length > 0);
}

export function sessionTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
