import { useState } from "react";
import type { TaskPlan } from "../api/types";

type PlanPanelProps = {
  plan?: TaskPlan | null;
  onApprove: () => void | Promise<void>;
  onReject: () => void | Promise<void>;
};

export function PlanPanel({ plan, onApprove, onReject }: PlanPanelProps) {
  const [busy, setBusy] = useState(false);
  if (!plan) return null;

  const decide = async (action: () => void | Promise<void>) => {
    setBusy(true);
    try {
      await action();
    } finally {
      setBusy(false);
    }
  };

  return <section className="plan-panel" aria-label="Canonical task plan">
    <header className="plan-header">
      <div><span>CANONICAL TASK PLAN</span><strong>{plan.status.toUpperCase()}</strong></div>
      <b>REVISION {plan.revision}</b>
    </header>
    <h2>{plan.definition.summary}</h2>
    <div className="plan-criteria">
      <span>DONE WHEN</span>
      <ul>{plan.definition.completion_criteria.map(item => <li key={item}>{item}</li>)}</ul>
    </div>
    <ol className="plan-steps">
      {plan.steps.map((item, index) => <li key={item.step_id}>
        <span className={`plan-step-state ${item.status}`}>{String(index + 1).padStart(2, "0")}</span>
        <div><strong>{item.step}</strong><p>{item.details}</p><small>VERIFY · {item.verification}</small></div>
      </li>)}
    </ol>
    {plan.status === "proposed" && <footer>
      <button disabled={busy} aria-label="Reject plan" onClick={() => decide(onReject)}>REJECT</button>
      <button disabled={busy} className="plan-approve" aria-label="Approve plan" onClick={() => decide(onApprove)}>APPROVE PLAN</button>
    </footer>}
  </section>;
}
