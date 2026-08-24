import { useState } from "react";
import { ShieldAlert } from "lucide-react";
import type { Approval } from "../../api/types";
import { Collapsible } from "../../components/ui/Collapsible";
import styles from "./Conversation.module.css";

export function ApprovalRequest({ approval, onDecision }: { approval: Approval; onDecision: (decision: "approve" | "deny", reason: string) => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  const [resolved, setResolved] = useState<"approve" | "deny" | null>(null);
  const decide = async (decision: "approve" | "deny") => {
    setBusy(true);
    try { await onDecision(decision, ""); setResolved(decision); } finally { setBusy(false); }
  };
  const args = approval.arguments ?? {};
  const command = approval.command ?? (typeof args.command === "string" ? args.command : typeof args.cmd === "string" ? args.cmd : undefined);
  const path = approval.path ?? (typeof args.path === "string" ? args.path : typeof args.file_path === "string" ? args.file_path : undefined);
  if (resolved) return <div className={styles.approvalResolved}><ShieldAlert size={15} />{resolved === "approve" ? "已允许此操作" : "已拒绝此操作"}</div>;
  return <section className={styles.approval} aria-label="需要审批">
    <header><ShieldAlert size={18} /><div><strong>需要你的批准</strong><span>{approval.risk_level ? `风险等级：${approval.risk_level}` : "受保护的操作"}</span></div></header>
    <h3>{approval.tool_name ?? "工具操作"}</h3>
    <p>{approval.reason ?? "Codepilot 需要获得许可才能继续。"}</p>
    {(command || path || approval.effects?.length) && <Collapsible label="查看操作详情" defaultOpen><dl>{command && <><dt>命令</dt><dd><code>{command}</code></dd></>}{path && <><dt>路径</dt><dd><code>{path}</code></dd></>}{approval.effects?.length && <><dt>影响</dt><dd>{approval.effects.join("、")}</dd></>}</dl></Collapsible>}
    <footer><button disabled={busy} onClick={() => void decide("deny")}>拒绝</button><button disabled={busy} className={styles.approveButton} onClick={() => void decide("approve")}>允许一次</button></footer>
  </section>;
}
