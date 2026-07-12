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
};

export const initialEventState: SessionEventState = {
  connected: false, lastEventId: null, lastSequence: 0, seenIds: [], streamingText: "",
  pendingApprovals: [], runState: "idle", needsSync: false,
};

export function reduceWebEvent(state: SessionEventState, event: WebEvent): SessionEventState {
  if (state.seenIds.includes(event.event_id)) return state;
  if (state.lastSequence > 0 && event.sequence > state.lastSequence + 1) {
    return { ...state, needsSync: true };
  }
  const base = {
    ...state,
    lastEventId: event.event_id,
    lastSequence: event.sequence,
    seenIds: [...state.seenIds.slice(-127), event.event_id],
  };
  if (event.type === "connected") return { ...base, connected: true };
  if (event.type === "message_delta") return { ...base, runState: "running", streamingText: base.streamingText + String(event.data.delta ?? "") };
  if (event.type === "approval_required") return { ...base, runState: "paused", pendingApprovals: [...base.pendingApprovals, event.data as Approval] };
  if (["run_finished", "cancelled", "command_finished"].includes(event.type)) return { ...base, runState: "idle", streamingText: "", needsSync: true };
  if (event.type === "failed") return { ...base, runState: "failed" };
  if (event.type === "sync_required") return { ...base, needsSync: true };
  return base;
}
