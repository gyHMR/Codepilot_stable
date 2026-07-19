import { useEffect, useMemo, useReducer } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import { CircleDot, LoaderCircle } from "lucide-react";
import type { Approval, TimelineItem } from "../api/types";
import { api } from "../api/client";
import { SessionEventClient } from "../events/client";
import { initialEventState, reduceWebEvent } from "../events/reducer";
import { WorkspaceShell } from "../layouts/WorkspaceShell";
import { SessionSidebar } from "../features/sessions/SessionSidebar";
import { ConversationTimeline } from "../features/conversation/ConversationTimeline";
import { Composer } from "../features/conversation/Composer";
import { ApprovalDock } from "../features/conversation/ApprovalDock";
import { InteractionDock } from "../features/conversation/InteractionDock";
import { ContinuationDock } from "../features/conversation/ContinuationDock";
import { ProjectPanel } from "../features/project/ProjectPanel";
import styles from "./WorkspacePage.module.css";

const runLabels = { idle: "空闲", running: "正在执行", paused: "等待处理", cancelling: "正在停止", failed: "执行失败" } as const;

export function WorkspacePage() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [events, dispatch] = useReducer(reduceWebEvent, initialEventState);
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.listSessions });
  const workspace = useQuery({ queryKey: ["workspace"], queryFn: api.getWorkspaceSummary });
  const session = useQuery({ queryKey: ["session", sessionId], queryFn: () => api.getSession(sessionId!), enabled: Boolean(sessionId) });
  const timeline = useQuery({ queryKey: ["timeline", sessionId], queryFn: () => api.getTimeline(sessionId!), enabled: Boolean(sessionId) });

  useEffect(() => {
    if (!sessionId) return;
    dispatch({ event_id: `reset-${sessionId}`, session_id: sessionId, run_id: null, type: "session_reset", sequence: 0, timestamp: new Date().toISOString(), data: {} });
    const client = new SessionEventClient(
      sessionId,
      event => dispatch(event),
      connected => dispatch({ event_id: `connection-${sessionId}-${connected}`, session_id: sessionId, run_id: null, type: connected ? "connected" : "disconnected", sequence: 0, timestamp: new Date().toISOString(), data: {} }),
    );
    client.connect();
    return () => client.close();
  }, [sessionId]);

  useEffect(() => {
    if (!events.syncRevision || !sessionId) return;
    void queryClient.invalidateQueries({ queryKey: ["timeline", sessionId] });
    void queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
    void queryClient.invalidateQueries({ queryKey: ["sessions"] });
    void queryClient.invalidateQueries({ queryKey: ["workspace"] });
  }, [events.syncRevision, queryClient, sessionId]);

  const create = useMutation({ mutationFn: () => api.createSession(), onSuccess: value => { void queryClient.invalidateQueries({ queryKey: ["sessions"] }); navigate(`/sessions/${value.session_id}`); } });
  const rename = async (id: string, title: string) => { await api.updateSession(id, title); await Promise.all([queryClient.invalidateQueries({ queryKey: ["sessions"] }), queryClient.invalidateQueries({ queryKey: ["session", id] })]); };
  const remove = async (id: string) => { await api.deleteSession(id); await queryClient.invalidateQueries({ queryKey: ["sessions"] }); if (id === sessionId) navigate("/"); };

  const pendingApprovals = useMemo(() => Array.from(new Map([...(session.data?.pending_approvals ?? []), ...events.pendingApprovals].map(item => [item.approval_id, item])).values()).filter(item => !events.resolvedApprovalIds.includes(item.approval_id)), [events.pendingApprovals, events.resolvedApprovalIds, session.data?.pending_approvals]);
  const submit = async (text: string) => {
    if (!sessionId) return;
    const optimistic: TimelineItem = { item_id: `optimistic-${Date.now()}`, session_id: sessionId, run_id: null, timestamp: new Date().toISOString(), type: "user_message", data: { message: { role: "user", content: text } } };
    queryClient.setQueryData<TimelineItem[]>(["timeline", sessionId], previous => [...(previous ?? []), optimistic]);
    try { await api.submitMessage(sessionId, text); } catch (error) { await queryClient.invalidateQueries({ queryKey: ["timeline", sessionId] }); throw error; }
  };
  const decideApproval = async (approvalId: string, decision: "approve" | "deny", reason = "") => { if (sessionId) { await api.decideApproval(sessionId, approvalId, decision, reason); await Promise.all([queryClient.invalidateQueries({ queryKey: ["session", sessionId] }), queryClient.invalidateQueries({ queryKey: ["sessions"] })]); } };

  const sidebar = <SessionSidebar projectName={workspace.data?.name ?? "当前项目"} sessions={sessions.data ?? []} activeId={sessionId} creating={create.isPending} onCreate={() => create.mutate()} onSelect={id => navigate(`/sessions/${id}`)} onRename={rename} onDelete={remove} />;
  const project = <ProjectPanel workspace={workspace.data} session={session.data} events={events} />;
  const conversation = <div className={styles.workbench}>
    <header className={styles.topbar}>
      <div className={styles.title}><strong>{session.data?.title ?? (sessionId ? "正在加载 Session" : "项目工作台")}</strong>{session.data && <span>{session.data.current_mode}</span>}</div>
      {sessionId && <div className={`${styles.runBadge} ${styles[events.runState]}`}><CircleDot size={14} />{runLabels[events.runState]}</div>}
    </header>
    {timeline.isLoading ? <div className={styles.loading}><LoaderCircle size={18} />正在加载会话</div> : timeline.isError ? <div className={styles.error}><strong>无法加载会话记录</strong><button onClick={() => void timeline.refetch()}>重试</button></div> : <ConversationTimeline items={timeline.data ?? []} streamingText={events.streamingText} activities={events.activities} plan={session.data?.plan} lastResult={events.lastResult} lastError={events.lastError} onPlanDecision={async decision => { if (sessionId) await api.submitCommand(sessionId, `/plan ${decision}`); }} />}
    <ApprovalDock approvals={pendingApprovals as Approval[]} onDecision={(id, decision) => decideApproval(id, decision)} />
    <InteractionDock key={events.pendingInteraction?.request_id ?? "no-interaction"} interaction={events.pendingInteraction} onSubmit={async (requestId, answer) => { if (sessionId) await api.respondInteraction(sessionId, requestId, answer); }} />
    <ContinuationDock wait={events.pendingContinuation} onContinue={async requestId => { if (sessionId) await api.continueRun(sessionId, requestId); }} />
    <Composer busy={events.snapshotReceived ? ["running", "paused", "cancelling"].includes(events.runState) : Boolean(session.data?.is_running)} disabled={!sessionId} onSubmit={submit} onCommand={async command => { if (sessionId) await api.submitCommand(sessionId, command); }} onCancel={async () => { if (sessionId) await api.cancelRun(sessionId); }} />
  </div>;

  return <WorkspaceShell sidebar={sidebar} conversation={conversation} project={project} />;
}
