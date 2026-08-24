import { CheckCircle2, CircleEllipsis, FileCode2, Terminal } from "lucide-react";
import type { Activity } from "../../api/types";
import { Collapsible } from "../../components/ui/Collapsible";
import styles from "./Conversation.module.css";

function activityLabel(activity: Activity): string {
  const name = String(activity.name ?? activity.tool_name ?? activity.type ?? "操作");
  const status = String(activity.status ?? "");
  const rawArgs = activity.arguments ?? activity.args;
  const args = rawArgs && typeof rawArgs === "object" ? rawArgs as Record<string, unknown> : {};
  const target = String(activity.target ?? activity.path ?? activity.command ?? args.command ?? args.cmd ?? args.path ?? args.file_path ?? args.query ?? "").trim();
  const suffix = target ? `：${target}` : "";
  return status === "completed" ? `${name}${suffix} 已完成` : `${name}${suffix}`;
}

export function ActivityGroup({ activities, defaultOpen = false }: { activities: Activity[]; defaultOpen?: boolean }) {
  if (!activities.length) return null;
  const files = activities.filter(item => String(item.type ?? item.name ?? "").toLowerCase().match(/file|read|write|edit/)).length;
  const commands = activities.filter(item => String(item.type ?? item.name ?? "").toLowerCase().match(/shell|command|bash|test/)).length;
  const summary = [files ? `${files} 个文件操作` : "", commands ? `${commands} 个命令` : "", !files && !commands ? `${activities.length} 项操作` : ""].filter(Boolean).join("，");
  return <div className={styles.activityGroup}><Collapsible defaultOpen={defaultOpen} label={<span className={styles.activitySummary}><CircleEllipsis size={16} />{summary}</span>}>
    <ul>{activities.map((activity, index) => <li key={String(activity.activity_id ?? index)}>{String(activity.status) === "completed" ? <CheckCircle2 size={15} /> : String(activity.type ?? "").match(/shell|command/) ? <Terminal size={15} /> : <FileCode2 size={15} />}<span><strong>{activityLabel(activity)}</strong>{Boolean(activity.summary) && <small>{String(activity.summary)}</small>}</span></li>)}</ul>
  </Collapsible></div>;
}
