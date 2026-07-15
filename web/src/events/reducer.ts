import type { Approval, WebEvent } from "../api/types";

export type SessionEventState = {
  connected: boolean;
  lastEventId: string | null;
  lastSequence: number;
  seenIds: string[];
  streamingText: string;
  pendingApprovals: Approval[];
  runState: "idle" | "running" | "paused" | "cancelling" | "failed";
  needsSync: boolean;
  syncRevision: number;
};

export const initialEventState: SessionEventState = {
  connected: false, lastEventId: null, lastSequence: 0, seenIds: [], streamingText: "",
  pendingApprovals: [], runState: "idle", needsSync: false, syncRevision: 0,
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
  if (event.type === "message_delta") return { ...base, runState: "running", streamingText: base.streamingText + String(event.data.delta ?? "") };
  if (event.type === "tool_activity") return { ...base, runState: "running" };
  if (event.type === "approval_required") {
    const approval = event.data as Approval;
    const pendingApprovals = base.pendingApprovals.some(item => item.approval_id === approval.approval_id)
      ? base.pendingApprovals
      : [...base.pendingApprovals, approval];
    return { ...base, runState: "paused", pendingApprovals };
  }
  if (event.type === "run_paused") return { ...base, runState: "paused", needsSync: true, syncRevision: base.syncRevision + 1 };
  if (["run_finished", "cancelled", "command_finished"].includes(event.type)) return { ...base, runState: "idle", streamingText: "", pendingApprovals: [], needsSync: true, syncRevision: base.syncRevision + 1 };
  if (event.type === "failed") return { ...base, runState: "failed", streamingText: "", pendingApprovals: [], needsSync: true, syncRevision: base.syncRevision + 1 };
  if (event.type === "sync_required") return { ...base, needsSync: true, syncRevision: base.syncRevision + 1 };
  return base;
}
