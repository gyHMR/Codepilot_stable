import { describe, expect, it } from "vitest";
import { initialEventState, reduceWebEvent } from "./reducer";

describe("event reducer", () => {
  it("appends deltas and ignores duplicates", () => {
    const event = { event_id: "e1", session_id: "s1", run_id: "r1", type: "message_delta", sequence: 1, timestamp: "now", data: { delta: "hi" } };
    const once = reduceWebEvent(initialEventState, event);
    const twice = reduceWebEvent(once, event);
    expect(twice.streamingText).toBe("hi");
  });

  it("requests sync when a sequence gap appears", () => {
    const state = { ...initialEventState, lastSequence: 1 };
    const next = reduceWebEvent(state, { event_id: "e3", session_id: "s1", run_id: null, type: "progress", sequence: 3, timestamp: "now", data: {} });
    expect(next.needsSync).toBe(true);
    expect(next.syncRevision).toBe(1);
  });

  it("requests a fresh sync for every terminal event and clears approvals", () => {
    const waiting = { ...initialEventState, pendingApprovals: [{ approval_id: "a1" }], runState: "paused" as const };
    const first = reduceWebEvent(waiting, { event_id: "e1", session_id: "s1", run_id: "r1", type: "run_finished", sequence: 1, timestamp: "now", data: {} });
    const second = reduceWebEvent(first, { event_id: "e2", session_id: "s1", run_id: "r2", type: "run_finished", sequence: 2, timestamp: "now", data: {} });
    expect(first.pendingApprovals).toEqual([]);
    expect(second.syncRevision).toBe(2);
  });

  it("resets session-specific event state", () => {
    const active = { ...initialEventState, connected: true, streamingText: "old", lastSequence: 8 };
    const reset = reduceWebEvent(active, { event_id: "reset", session_id: "s2", run_id: null, type: "session_reset", sequence: 0, timestamp: "now", data: {} });
    expect(reset).toEqual(initialEventState);
  });
});
