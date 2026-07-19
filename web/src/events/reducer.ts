import type { Activity, Approval, Interaction, WaitState, WebEvent } from "../api/types";

export type SessionEventState = {
  connected: boolean;
  lastEventId: string | null;
  lastSequence: number;
  seenIds: string[];
  streamingText: string;
  activities: Activity[];
  lastResult: Record<string, unknown> | null;
  lastError: Record<string, unknown> | null;
  pendingApprovals: Approval[];
  resolvedApprovalIds: string[];
  pendingInteraction: Interaction | null;
  pendingContinuation: WaitState | null;
  runState: "idle" | "running" | "paused" | "cancelling" | "failed";
  runStartedAt: string | null;
  needsSync: boolean;
  syncRevision: number;
  snapshotReceived: boolean;
};

export const initialEventState: SessionEventState = {
  connected: false, lastEventId: null, lastSequence: 0, seenIds: [], streamingText: "",
  activities: [], lastResult: null, lastError: null, pendingApprovals: [], resolvedApprovalIds: [], pendingInteraction: null, pendingContinuation: null, runState: "idle", runStartedAt: null,
  needsSync: false, syncRevision: 0, snapshotReceived: false,
};

export function reduceWebEvent(state: SessionEventState, event: WebEvent): SessionEventState {
  if (event.type === "session_reset") return { ...initialEventState };
  if (event.type === "connected") return { ...state, connected: true };
  if (event.type === "disconnected") return { ...state, connected: false };
  if (state.seenIds.includes(event.event_id)) return state;
  const hasGap = state.lastSequence > 0 && event.sequence > state.lastSequence + 1;
  const base = {
    ...state,
    lastEventId: event.event_id,
    lastSequence: event.sequence,
    seenIds: [...state.seenIds.slice(-127), event.event_id],
    needsSync: state.needsSync || hasGap,
    syncRevision: state.syncRevision + (hasGap ? 1 : 0),
  };
  if (event.type === "session.snapshot") {
    const execution = event.data.execution && typeof event.data.execution === "object" ? event.data.execution as Record<string, unknown> : {};
    const status = String(execution.status ?? "idle");
    const pendingApprovals = Array.isArray(event.data.pending_approvals) ? event.data.pending_approvals as Approval[] : [];
    const pendingInteraction = event.data.pending_interaction && typeof event.data.pending_interaction === "object" ? event.data.pending_interaction as Interaction : null;
    const wait = event.data.wait && typeof event.data.wait === "object" ? event.data.wait as WaitState : null;
    const runState = status.startsWith("waiting_") || status === "paused" ? "paused" : status === "running" ? "running" : "idle";
    return {
      ...base,
      snapshotReceived: true,
      runState,
      pendingApprovals,
      pendingInteraction,
      pendingContinuation: wait?.kind === "continuation" ? wait : null,
      streamingText: runState === "idle" ? "" : base.streamingText,
      activities: runState === "idle" ? [] : base.activities,
      runStartedAt: runState === "idle" ? null : base.runStartedAt,
      needsSync: true,
      syncRevision: base.syncRevision + 1,
    };
  }
  if (event.type === "assistant.delta") return { ...base, runState: "running", pendingContinuation: null, runStartedAt: base.runStartedAt ?? event.timestamp, lastResult: null, lastError: null, streamingText: base.streamingText + String(event.data.delta ?? "") };
  if (event.type === "activity.updated") {
    const activity = event.data as Activity;
    const activityId = String(activity.activity_id ?? activity.tool_call_id ?? activity.id ?? event.event_id);
    const previous = base.activities.find(item => String(item.activity_id ?? item.tool_call_id ?? item.id) === activityId);
    const appendSummary = String(activity.append_summary ?? "");
    const nextActivity = appendSummary
      ? { ...previous, ...activity, summary: String(previous?.summary ?? "") + appendSummary, activity_id: activityId }
      : { ...previous, ...activity, activity_id: activityId };
    let activities = base.activities.filter(item => String(item.activity_id ?? item.tool_call_id ?? item.id) !== activityId);
    const startsToolCall = String(activity.type ?? "") === "tool_call";
    if (startsToolCall && base.streamingText.trim()) {
      activities = [...activities, {
        activity_id: `narration-${event.sequence}`,
        type: "narration",
        name: "执行说明",
        status: "completed",
        summary: base.streamingText,
      }];
    }
    return { ...base, runState: "running", pendingContinuation: null, runStartedAt: base.runStartedAt ?? event.timestamp, streamingText: startsToolCall ? "" : base.streamingText, activities: [...activities, nextActivity] };
  }
  if (event.type === "approval.requested") {
    const approval = event.data as Approval;
    const pendingApprovals = base.pendingApprovals.some(item => item.approval_id === approval.approval_id)
      ? base.pendingApprovals
      : [...base.pendingApprovals, approval];
    return { ...base, runState: "paused", resolvedApprovalIds: base.resolvedApprovalIds.filter(id => id !== approval.approval_id), pendingApprovals };
  }
  if (event.type === "approval.resolved") {
    const id = String(event.data.approval_id ?? "");
    return { ...base, pendingApprovals: base.pendingApprovals.filter(item => item.approval_id !== id), resolvedApprovalIds: [...base.resolvedApprovalIds, id] };
  }
  if (event.type === "interaction.requested") {
    const interaction: Interaction = {
      run_id: String(event.run_id ?? event.data.run_id ?? ""),
      kind: "user_input",
      request_id: String(event.data.request_id ?? ""),
      payload: event.data.payload && typeof event.data.payload === "object" ? event.data.payload as Record<string, unknown> : {},
    };
    return { ...base, runState: "paused", pendingInteraction: interaction, activities: base.activities.filter(item => String(item.name ?? item.tool_name ?? "") !== "request_user_input") };
  }
  if (event.type === "interaction.resolved") return { ...base, pendingInteraction: null, runState: "running" };
  if (event.type === "plan.confirmation_requested") return { ...base, runState: "paused", needsSync: true, syncRevision: base.syncRevision + 1 };
  if (event.type === "continuation.requested") return { ...base, runState: "paused", pendingContinuation: { run_id: String(event.run_id ?? ""), kind: "continuation", request_id: String(event.data.request_id ?? ""), payload: event.data.payload && typeof event.data.payload === "object" ? event.data.payload as Record<string, unknown> : {} } };
  if (event.type === "run.status_changed") {
    const status = String(event.data.status ?? "running");
    const runState = status === "paused" ? "paused" : status === "cancelling" ? "cancelling" : "running";
    return { ...base, runState, pendingContinuation: runState === "running" ? null : base.pendingContinuation, runStartedAt: base.runStartedAt ?? event.timestamp, lastResult: null, lastError: null };
  }
  if (event.type === "run.completed") return { ...base, snapshotReceived: true, runState: "idle", streamingText: "", activities: [], pendingApprovals: [], pendingInteraction: null, pendingContinuation: null, runStartedAt: null, lastResult: event.data, needsSync: true, syncRevision: base.syncRevision + 1 };
  if (event.type === "error") return { ...base, runState: "failed", streamingText: "", pendingApprovals: [], lastError: event.data, needsSync: true, syncRevision: base.syncRevision + 1 };
  if (event.type === "plan.updated" || event.type === "workspace.changed") return { ...base, needsSync: true, syncRevision: base.syncRevision + 1 };
  if (event.type === "sync_required") return { ...base, needsSync: true, syncRevision: base.syncRevision + 1 };
  return base;
}
