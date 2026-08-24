import { Check, Circle, CircleDot } from "lucide-react";
import type { TaskPlan } from "../../api/types";
import { Collapsible } from "../../components/ui/Collapsible";
import styles from "./ProjectPanel.module.css";

export function TaskProgress({ plan }: { plan?: TaskPlan | null }) {
  if (!plan) return <section className={`${styles.section} ${styles.taskSection}`}><h2>当前任务</h2><p className={styles.muted}>当前未创建任务计划</p></section>;
  const completed = plan.steps.filter(step => step.status === "completed").length;
  return <section className={`${styles.section} ${styles.taskSection}`}>
    <h2>当前任务</h2>
    <strong className={styles.taskTitle}>{plan.definition.summary}</strong>
    <div className={styles.progressLabel}><span>{completed} / {plan.steps.length}</span><div><i style={{ width: `${plan.steps.length ? completed / plan.steps.length * 100 : 0}%` }} /></div></div>
    <ol className={styles.steps}>{plan.steps.map(step => <li key={step.step_id} className={styles[step.status]}>
      {step.status === "completed" ? <Check size={15} /> : step.status === "in_progress" ? <CircleDot size={15} /> : <Circle size={15} />}
      <Collapsible label={<span>{step.step}</span>}><p>{step.details}</p>{step.verification && <small>验证：{step.verification}</small>}</Collapsible>
    </li>)}</ol>
    {plan.status === "proposed" && <p className={styles.approvalHint}>任务计划等待你在对话中确认</p>}
  </section>;
}
