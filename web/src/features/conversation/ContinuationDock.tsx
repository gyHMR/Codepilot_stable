import { useState } from "react";
import { RotateCw } from "lucide-react";
import type { WaitState } from "../../api/types";
import styles from "./Conversation.module.css";

export function ContinuationDock({ wait, onContinue }: { wait: WaitState | null; onContinue: (requestId: string) => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  if (!wait) return null;
  const reason = wait.request_id.replace(/^continuation:/, "");
  const submit = async () => { setBusy(true); try { await onContinue(wait.request_id); } finally { setBusy(false); } };
  return <section className={styles.continuationDock}>
    <div><RotateCw size={16} /><span><strong>本轮执行已到达边界</strong><small>{reason}</small></span></div>
    <button disabled={busy} onClick={() => void submit()}>{busy ? "继续中" : "继续执行"}</button>
  </section>;
}
