export type SessionSummary = {
  session_id: string;
  workspace: string;
  model_id: string;
  permission_mode: string;
  current_mode: string;
  is_running: boolean;
  message_count: number;
  pending_approvals: Approval[];
};

export type Approval = {
  approval_id: string;
  tool_name?: string;
  reason?: string;
  risk_level?: string;
};

export type Message = { role?: string; content?: unknown; [key: string]: unknown };
export type AcceptedAction = { session_id: string; accepted: boolean; run_id: string | null };
export type ApiErrorPayload = { error: { code: string; message: string; details?: unknown } };

export type WebEvent = {
  event_id: string;
  session_id: string;
  run_id: string | null;
  type: string;
  sequence: number;
  timestamp: string;
  data: Record<string, unknown>;
};
