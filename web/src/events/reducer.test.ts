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
  });
});
