import type { SessionSummary, WorkspaceSummary } from "../../api/types";
import type { SessionEventState } from "../../events/reducer";
import { GitStatus } from "./GitStatus";
import { RuntimeStatus } from "./RuntimeStatus";
import { TaskProgress } from "./TaskProgress";
import styles from "./ProjectPanel.module.css";

export function ProjectPanel({ workspace, session, events }: { workspace?: WorkspaceSummary; session?: SessionSummary; events: SessionEventState }) {
  return <div className={styles.panel}><GitStatus workspace={workspace} /><TaskProgress plan={session?.plan} /><RuntimeStatus session={session} events={events} /></div>;
}
