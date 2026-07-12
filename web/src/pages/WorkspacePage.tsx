import { useEffect, useReducer } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import { SessionEventClient } from "../events/client";
import { initialEventState, reduceWebEvent } from "../events/reducer";
import { ApprovalCard } from "../components/ApprovalCard";
import { ChatTimeline } from "../components/ChatTimeline";
import { Composer } from "../components/Composer";
import { SessionInspector } from "../components/SessionInspector";
import { SessionSidebar } from "../components/SessionSidebar";

export function WorkspacePage() {
  const { sessionId } = useParams(); const navigate = useNavigate(); const queryClient = useQueryClient();
  const [events, dispatch] = useReducer(reduceWebEvent, initialEventState);
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.listSessions });
  const session = useQuery({ queryKey: ["session", sessionId], queryFn: () => api.getSession(sessionId!), enabled: Boolean(sessionId) });
  const messages = useQuery({ queryKey: ["messages", sessionId], queryFn: () => api.getMessages(sessionId!), enabled: Boolean(sessionId) });
  useEffect(() => { if (!sessionId) return; const client = new SessionEventClient(sessionId, event => dispatch(event), connected => { if (connected) dispatch({ event_id: `connection-${Date.now()}`, session_id: sessionId, run_id: null, type: "connected", sequence: Math.max(1, events.lastSequence + 1), timestamp: new Date().toISOString(), data: {} }); }); client.connect(); return () => client.close(); }, [sessionId]);
  useEffect(() => { if (events.needsSync && sessionId) { void queryClient.invalidateQueries({ queryKey: ["messages", sessionId] }); void queryClient.invalidateQueries({ queryKey: ["session", sessionId] }); } }, [events.needsSync, sessionId]);
  const create = useMutation({ mutationFn: api.createSession, onSuccess: value => { void queryClient.invalidateQueries({ queryKey: ["sessions"] }); navigate(`/sessions/${value.session_id}`); } });
  const current = session.data;
  return <main className="app-shell"><SessionSidebar sessions={sessions.data ?? []} activeId={sessionId} onCreate={() => create.mutate()} onSelect={id => navigate(`/sessions/${id}`)} onDelete={async id => { if (confirm("Delete this session record?")) { await api.deleteSession(id); void queryClient.invalidateQueries({ queryKey: ["sessions"] }); if (id === sessionId) navigate("/"); } }} />
    <section className="workbench"><header className="topbar"><div><span className="eyebrow">ACTIVE THREAD</span><b>{sessionId?.slice(0, 18) ?? "NO SESSION SELECTED"}</b></div><span className="top-status">{events.runState.toUpperCase()}</span></header><ChatTimeline messages={messages.data ?? []} streamingText={events.streamingText} />{events.pendingApprovals.map(approval => <ApprovalCard key={approval.approval_id} approval={approval} onDecision={(decision, reason) => api.decideApproval(sessionId!, approval.approval_id, decision, reason).then(() => undefined)} />)}{sessionId ? <Composer busy={current?.is_running || events.runState === "running"} onSubmit={text => text.startsWith("/") ? api.submitCommand(sessionId, text).then(() => undefined) : api.submitMessage(sessionId, text).then(() => undefined)} onCancel={() => api.cancelRun(sessionId).then(() => undefined)} /> : <button className="launch" onClick={() => create.mutate()}>INITIALIZE FIRST SESSION</button>}</section>
    <SessionInspector session={current} connected={events.connected} /></main>;
}
