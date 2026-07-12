import type { AcceptedAction, ApiErrorPayload, Message, SessionSummary } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details?: unknown) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init.body ? { "Content-Type": "application/json", ...init.headers } : init.headers,
  });
  if (!response.ok) {
    const payload = (await response.json()) as ApiErrorPayload;
    throw new ApiError(response.status, payload.error.code, payload.error.message, payload.error.details);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  listSessions: () => request<SessionSummary[]>("/api/sessions"),
  createSession: () => request<SessionSummary>("/api/sessions", { method: "POST", body: "{}" }),
  getSession: (id: string) => request<SessionSummary>(`/api/sessions/${id}`),
  getMessages: (id: string) => request<Message[]>(`/api/sessions/${id}/messages`),
  deleteSession: (id: string) => request<void>(`/api/sessions/${id}`, { method: "DELETE" }),
  submitMessage: (id: string, text: string) => request<AcceptedAction>(`/api/sessions/${id}/messages`, { method: "POST", body: JSON.stringify({ text }) }),
  submitCommand: (id: string, text: string) => request<AcceptedAction>(`/api/sessions/${id}/commands`, { method: "POST", body: JSON.stringify({ text }) }),
  decideApproval: (id: string, approvalId: string, decision: "approve" | "deny", reason = "") => request<AcceptedAction>(`/api/sessions/${id}/approvals/${approvalId}`, { method: "POST", body: JSON.stringify({ decision, reason }) }),
  cancelRun: (id: string) => request<AcceptedAction>(`/api/sessions/${id}/cancel`, { method: "POST" }),
};
