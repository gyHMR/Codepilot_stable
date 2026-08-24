import type { AcceptedAction, ApiErrorPayload, Message, SessionSummary, TimelineItem, WorkspaceSummary } from "./types";

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
    let payload: ApiErrorPayload | null = null;
    try { payload = (await response.json()) as ApiErrorPayload; } catch { payload = null; }
    throw new ApiError(response.status, payload?.error.code ?? "web.request_failed", payload?.error.message ?? `请求失败 (${response.status})`, payload?.error.details);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  listSessions: () => request<SessionSummary[]>("/api/sessions"),
  createSession: (title?: string) => request<SessionSummary>("/api/sessions", { method: "POST", body: JSON.stringify(title ? { title } : {}) }),
  getSession: (id: string) => request<SessionSummary>(`/api/sessions/${id}`),
  updateSession: (id: string, title: string) => request<SessionSummary>(`/api/sessions/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  getMessages: (id: string) => request<Message[]>(`/api/sessions/${id}/messages`),
  getTimeline: (id: string) => request<TimelineItem[]>(`/api/sessions/${id}/timeline`),
  getWorkspaceSummary: () => request<WorkspaceSummary>("/api/workspace/summary"),
  deleteSession: (id: string) => request<void>(`/api/sessions/${id}`, { method: "DELETE" }),
  submitMessage: (id: string, text: string) => request<AcceptedAction>(`/api/sessions/${id}/messages`, { method: "POST", body: JSON.stringify({ text }) }),
  submitCommand: (id: string, text: string) => request<AcceptedAction>(`/api/sessions/${id}/commands`, { method: "POST", body: JSON.stringify({ text }) }),
  decideApproval: (id: string, approvalId: string, decision: "approve" | "deny", reason = "") => request<AcceptedAction>(`/api/sessions/${id}/approvals/${approvalId}`, { method: "POST", body: JSON.stringify({ decision, reason }) }),
  respondInteraction: (id: string, requestId: string, answer: string) => request<AcceptedAction>(`/api/sessions/${id}/interactions/${requestId}`, { method: "POST", body: JSON.stringify({ answer }) }),
  continueRun: (id: string, requestId: string) => request<AcceptedAction>(`/api/sessions/${id}/continuations/${requestId}`, { method: "POST" }),
  cancelRun: (id: string) => request<AcceptedAction>(`/api/sessions/${id}/cancel`, { method: "POST" }),
};
