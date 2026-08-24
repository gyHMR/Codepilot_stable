import { useState } from "react";
import { ListChecks } from "lucide-react";
import type { TaskPlan } from "../../api/types";
import { Collapsible } from "../../components/ui/Collapsible";
import styles from "./Conversation.module.css";

export function PlanApproval({ plan, onDecision }: { plan: TaskPlan; onDecision: (decision: "approve" | "reject") => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  const revision = plan.pending_revision;
  if (plan.status !== "proposed" && !revision) return null;
  const definition = revision?.definition ?? plan.definition;
  const steps = revision?.steps ?? plan.steps;
  const decide = async (decision: "approve" | "reject") => { setBusy(true); try { await onDecision(decision); } finally { setBusy(false); } };
  return <section className={styles.planApproval} aria-label="任务计划等待确认">
    <header><ListChecks size={18} /><div><strong>{revision ? "确认计划修订" : "确认任务计划"}</strong><span>修订 {revision?.proposed_at_revision ?? plan.revision}</span></div></header>
    <h3>{definition.summary}</h3>
    <Collapsible label={`${steps.length} 个任务步骤`} defaultOpen>
      <ol>{steps.map(step => <li key={step.step_id}><strong>{step.step}</strong>{step.details && <p>{step.details}</p>}</li>)}</ol>
    </Collapsible>
    <footer><button disabled={busy} onClick={() => void decide("reject")}>拒绝</button><button disabled={busy} className={styles.primaryAction} onClick={() => void decide("approve")}>批准计划</button></footer>
  </section>;
}
