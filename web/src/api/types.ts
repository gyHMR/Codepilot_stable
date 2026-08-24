export type SessionSummary = {
  session_id: string;
  title: string;
  workspace: string;
  model_id: string;
  permission_mode: string;
  current_mode: string;
  is_running: boolean;
  message_count: number;
  created_at: string;
  updated_at: string;
  status: "idle" | "running" | "approval" | "waiting" | "failed";
  pending_approvals: Approval[];
  wait?: WaitState | null;
  plan?: TaskPlan | null;
};

export type PlanStep = {
  step_id: string;
  step: string;
  details: string;
  verification: string;
  status: "pending" | "in_progress" | "completed";
};

export type TaskPlan = {
  plan_id: string;
  status: "proposed" | "active" | "completed" | "rejected" | "abandoned";
  revision: number;
  definition: {
    summary: string;
    completion_criteria: string[];
    [key: string]: unknown;
  };
  steps: PlanStep[];
  pending_revision?: {
    reason: string;
    definition: TaskPlan["definition"];
    steps: PlanStep[];
    proposed_at_revision: number;
  } | null;
};

export type Approval = {
  approval_id: string;
  tool_name?: string;
  reason?: string;
  risk_level?: string;
  effects?: string[];
  command?: string;
  path?: string;
  arguments?: Record<string, unknown>;
  safe_preview?: Record<string, unknown>;
};

export type WaitState = {
  run_id: string;
  kind: "tool_approval" | "user_input" | "plan_confirmation" | "continuation";
  request_id: string;
  payload: Record<string, unknown>;
};

export type Interaction = WaitState & {
  kind: "user_input";
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

export type TimelineItemType =
  | "user_message"
  | "assistant_message"
  | "activity_group"
  | "approval_request"
  | "run_result"
  | "error";

export type TimelineItem = {
  item_id: string;
  session_id: string;
  run_id: string | null;
  timestamp: string;
  type: TimelineItemType;
  data: Record<string, unknown>;
};

export type Activity = Record<string, unknown> & {
  activity_id?: string;
  type?: string;
  name?: string;
  status?: string;
};

export type WorkspaceChange = {
  path: string;
  status: "added" | "modified" | "deleted";
};

export type WorkspaceSummary = {
  name: string;
  path: string;
  git: {
    available: boolean;
    branch: string | null;
    clean: boolean;
    change_count: number;
    changes: WorkspaceChange[];
  };
};
