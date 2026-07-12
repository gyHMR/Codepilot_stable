import { useState } from "react";
import type { Approval } from "../api/types";

export function ApprovalCard({ approval, onDecision }: { approval: Approval; onDecision: (decision: "approve" | "deny", reason: string) => Promise<void> }) {
  const [reason, setReason] = useState(""); const [busy, setBusy] = useState(false);
  const decide = async (decision: "approve" | "deny") => { setBusy(true); try { await onDecision(decision, reason); } finally { setBusy(false); } };
  return <section className="approval-card"><div className="approval-kicker">ACTION REQUIRES CLEARANCE</div><h3>{approval.tool_name ?? "Protected tool"}</h3><p>{approval.reason ?? "Codepilot needs permission to continue."}</p><label>Decision note<input value={reason} onChange={event => setReason(event.target.value)} placeholder="Optional reason" /></label><div><button disabled={busy} onClick={() => decide("deny")}>DENY</button><button disabled={busy} className="approve" onClick={() => decide("approve")}>APPROVE</button></div></section>;
}
