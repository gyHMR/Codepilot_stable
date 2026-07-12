import type { WebEvent } from "../api/types";

export class SessionEventClient {
  private source: EventSource | null = null;
  private closed = false;
  private attempts = 0;
  private lastEventId: string | null = null;

  constructor(private sessionId: string, private onEvent: (event: WebEvent) => void, private onConnection: (connected: boolean) => void) {}

  connect() {
    this.closed = false;
    const query = this.lastEventId ? `?last_event_id=${encodeURIComponent(this.lastEventId)}` : "";
    this.source = new EventSource(`/api/sessions/${this.sessionId}/events${query}`);
    this.source.onopen = () => { this.attempts = 0; this.onConnection(true); };
    this.source.onmessage = (message) => this.consume(message);
    for (const type of ["connected", "progress", "message_delta", "tool_activity", "approval_required", "run_paused", "run_finished", "command_finished", "cancelled", "failed", "sync_required"]) {
      this.source.addEventListener(type, (message) => this.consume(message as MessageEvent));
    }
    this.source.onerror = () => {
      this.onConnection(false);
      this.source?.close();
      if (!this.closed) window.setTimeout(() => this.connect(), Math.min(10000, 500 * 2 ** this.attempts++));
    };
  }

  close() { this.closed = true; this.source?.close(); this.source = null; }

  private consume(message: MessageEvent) {
    const event = JSON.parse(message.data) as Record<string, unknown>;
    const normalized = { ...event, event_id: message.lastEventId || event.event_id } as WebEvent;
    this.lastEventId = normalized.event_id;
    this.onEvent(normalized);
  }
}
