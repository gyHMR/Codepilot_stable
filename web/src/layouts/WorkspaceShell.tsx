import type { ReactNode } from "react";
import { Dialog } from "radix-ui";
import { PanelLeft, PanelRight, X } from "lucide-react";
import { IconButton } from "../components/ui/IconButton";
import styles from "./WorkspaceShell.module.css";

function Drawer({ side, label, icon, children }: { side: "left" | "right"; label: string; icon: ReactNode; children: ReactNode }) {
  return <Dialog.Root>
    <Dialog.Trigger asChild><span className={side === "left" ? styles.leftTrigger : styles.rightTrigger}><IconButton label={label}>{icon}</IconButton></span></Dialog.Trigger>
    <Dialog.Portal>
      <Dialog.Overlay className={styles.overlay} />
      <Dialog.Content aria-describedby={undefined} className={`${styles.drawer} ${styles[side]}`}>
        <Dialog.Title className={styles.srOnly}>{label}</Dialog.Title>
        <Dialog.Close asChild><span className={styles.drawerClose}><IconButton label="关闭"><X size={18} /></IconButton></span></Dialog.Close>
        {children}
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}

export function WorkspaceShell({ sidebar, conversation, project }: { sidebar: ReactNode; conversation: ReactNode; project: ReactNode }) {
  return <main className={styles.shell}>
    <aside className={styles.sidebar}>{sidebar}</aside>
    <section className={styles.conversation}>
      <div className={styles.mobileControls}>
        <Drawer side="left" label="打开会话列表" icon={<PanelLeft size={18} />}>{sidebar}</Drawer>
        <Drawer side="right" label="打开项目详情" icon={<PanelRight size={18} />}>{project}</Drawer>
      </div>
      {conversation}
    </section>
    <aside className={styles.project}>{project}</aside>
  </main>;
}
