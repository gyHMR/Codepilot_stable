import { useState } from "react";
import { ChevronDown, ChevronUp, Pin, ShieldAlert } from "lucide-react";
import type { Approval } from "../../api/types";
import styles from "./Conversation.module.css";

export function ApprovalDock({ approvals, onDecision }: { approvals: Approval[]; onDecision: (id: string, decision: "approve" | "deny") => Promise<void> }) {
  const [expanded, setExpanded] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  if (!approvals.length) return null;
  const visible = expanded ? approvals : approvals.slice(0, 1);
  const decide = async (id: string, decision: "approve" | "deny") => {
    setBusy(id);
    try { await onDecision(id, decision); } finally { setBusy(null); }
  };
  return <section className={`${styles.approvalDock} ${expanded ? styles.approvalDockExpanded : ""} ${pinned ? styles.approvalDockPinned : ""}`} aria-label="待审批操作">
    <header><ShieldAlert size={16} /><strong>{approvals.length > 1 ? `${approvals.length} 项操作需要审批` : "操作需要审批"}</strong><div><button title={pinned ? "取消锁定" : "锁定面板"} onClick={() => setPinned(value => !value)}><Pin size={15} fill={pinned ? "currentColor" : "none"} /></button><button title={expanded ? "收起" : "展开"} onClick={() => setExpanded(value => !value)}>{expanded ? <ChevronDown size={16} /> : <ChevronUp size={16} />}</button></div></header>
    {visible.map(approval => {
      const args = approval.arguments ?? approval.safe_preview ?? {};
      const command = approval.command ?? (typeof args.command === "string" ? args.command : typeof args.cmd === "string" ? args.cmd : "");
      const path = approval.path ?? (typeof args.path === "string" ? args.path : typeof args.file_path === "string" ? args.file_path : "");
      return <article key={approval.approval_id}><div><strong>{approval.tool_name ?? "工具操作"}</strong><small>{command || path || approval.reason || "受保护的操作"}</small></div><footer><button disabled={busy === approval.approval_id} onClick={() => void decide(approval.approval_id, "deny")}>拒绝</button><button disabled={busy === approval.approval_id} className={styles.approveButton} onClick={() => void decide(approval.approval_id, "approve")}>{busy === approval.approval_id ? "提交中" : "允许"}</button></footer></article>;
    })}
  </section>;
}
