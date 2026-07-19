import type { Activity, Message, TaskPlan, TimelineItem } from "../../api/types";
import { ActivityGroup } from "./ActivityGroup";
import { MessageItem } from "./MessageItem";
import { PlanApproval } from "./PlanApproval";
import { RunResult } from "./RunResult";
import styles from "./Conversation.module.css";

function itemMessage(item: TimelineItem): Message {
  const message = item.data.message;
  return message && typeof message === "object" ? message as Message : { content: "" };
}

function itemActivities(item: TimelineItem): Activity[] {
  return Array.isArray(item.data.activities) ? item.data.activities as Activity[] : [item.data as Activity];
}

function groupedItems(items: TimelineItem[]): Array<TimelineItem | { item_id: string; type: "execution_group"; activities: Activity[] }> {
  const grouped: Array<TimelineItem | { item_id: string; type: "execution_group"; activities: Activity[] }> = [];
  for (const item of items) {
    if (item.type !== "activity_group") {
      grouped.push(item);
      continue;
    }
    const previous = grouped[grouped.length - 1];
    if (previous && previous.type === "execution_group") {
      previous.activities.push(...itemActivities(item));
    } else {
      grouped.push({ item_id: `execution-${item.run_id ?? item.item_id}`, type: "execution_group", activities: [...itemActivities(item)] });
    }
  }
  return grouped;
}

export function ConversationTimeline({ items, streamingText, activities, plan, lastResult, lastError, onPlanDecision }: { items: TimelineItem[]; streamingText: string; activities: Activity[]; plan?: TaskPlan | null; lastResult: Record<string, unknown> | null; lastError: Record<string, unknown> | null; onPlanDecision: (decision: "approve" | "reject") => Promise<void> }) {
  const empty = items.length === 0 && !streamingText && !activities.length && !lastResult && !lastError;
  return <div className={styles.timeline}>
    {empty && <div className={styles.emptyState}><h1>从这个项目开始</h1><p>描述你希望分析、修改或验证的内容。Codepilot 会在当前工作区中执行任务，并在需要时请求批准。</p></div>}
    <div className={styles.timelineInner}>
      {groupedItems(items).map(item => {
        if (item.type === "execution_group") return <ActivityGroup key={item.item_id} activities={item.activities} />;
        if (item.type === "user_message") return <MessageItem key={item.item_id} role="user" message={itemMessage(item)} />;
        if (item.type === "assistant_message") return <MessageItem key={item.item_id} role="assistant" message={itemMessage(item)} />;
        if (item.type === "run_result" || item.type === "error") return <RunResult key={item.item_id} data={item.data} />;
        return null;
      })}
      <ActivityGroup activities={activities} defaultOpen />
      {streamingText && <MessageItem role="assistant" streaming message={{ content: streamingText }} />}
      {plan && <PlanApproval plan={plan} onDecision={onPlanDecision} />}
      {lastResult && <RunResult data={lastResult} />}
      {lastError && <RunResult data={{ ...lastError, status: "failed" }} />}
    </div>
  </div>;
}
