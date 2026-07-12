import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./client";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("creates a session with the stable REST path", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ session_id: "s1" }), { status: 201 })
    );
    await api.createSession();
    expect(fetchMock).toHaveBeenCalledWith("/api/sessions", expect.objectContaining({ method: "POST" }));
  });

  it("maps the backend error envelope", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "runtime.run_active", message: "Busy" } }), { status: 409 })
    );
    await expect(api.submitMessage("s1", "hello")).rejects.toEqual(
      expect.objectContaining({ code: "runtime.run_active", status: 409 })
    );
  });
});
