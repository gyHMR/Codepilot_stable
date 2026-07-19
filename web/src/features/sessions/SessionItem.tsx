import { useState } from "react";
import { AlertDialog, Dialog } from "radix-ui";
import { Pencil, Trash2 } from "lucide-react";
import type { SessionSummary } from "../../api/types";
import { Menu } from "../../components/ui/Menu";
import { sessionTime } from "./session-title";
import styles from "./SessionSidebar.module.css";

export function SessionItem({ session, active, onSelect, onRename, onDelete }: { session: SessionSummary; active: boolean; onSelect: () => void; onRename: (title: string) => Promise<void>; onDelete: () => Promise<void> }) {
  const [renameOpen, setRenameOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [title, setTitle] = useState(session.title);
  const indicator = session.status === "approval" ? "等待审批" : session.status === "running" ? "正在运行" : session.status === "failed" ? "执行失败" : null;
  return <>
    <div className={`${styles.sessionItem} ${active ? styles.active : ""}`}>
      <button className={styles.sessionSelect} onClick={onSelect} aria-current={active ? "page" : undefined}>
        <span className={styles.sessionTitle}>{session.title}</span>
        <span className={styles.sessionMeta}>{indicator && <i className={styles[session.status]}>{indicator}</i>}<time>{sessionTime(session.updated_at)}</time></span>
      </button>
      <span className={styles.sessionMenu}><Menu items={[
        { label: "重命名", icon: <Pencil size={14} />, onSelect: () => { setTitle(session.title); setRenameOpen(true); } },
        { label: "删除", icon: <Trash2 size={14} />, danger: true, onSelect: () => setDeleteOpen(true) },
      ]} /></span>
    </div>
    <Dialog.Root open={renameOpen} onOpenChange={setRenameOpen}>
      <Dialog.Portal><Dialog.Overlay className={styles.dialogOverlay} /><Dialog.Content aria-describedby={undefined} className={styles.dialog}>
        <Dialog.Title>重命名 Session</Dialog.Title>
        <form onSubmit={event => { event.preventDefault(); const value = title.trim(); if (value) void onRename(value).then(() => setRenameOpen(false)); }}>
          <input aria-label="Session 标题" autoFocus value={title} onChange={event => setTitle(event.target.value)} maxLength={120} />
          <footer><Dialog.Close type="button">取消</Dialog.Close><button type="submit" className={styles.primaryButton}>保存</button></footer>
        </form>
      </Dialog.Content></Dialog.Portal>
    </Dialog.Root>
    <AlertDialog.Root open={deleteOpen} onOpenChange={setDeleteOpen}>
      <AlertDialog.Portal><AlertDialog.Overlay className={styles.dialogOverlay} /><AlertDialog.Content className={styles.dialog}>
        <AlertDialog.Title>删除这个 Session？</AlertDialog.Title>
        <AlertDialog.Description>会话记录将被永久删除，此操作无法撤销。</AlertDialog.Description>
        <footer><AlertDialog.Cancel>取消</AlertDialog.Cancel><AlertDialog.Action className={styles.dangerButton} onClick={() => void onDelete()}>删除</AlertDialog.Action></footer>
      </AlertDialog.Content></AlertDialog.Portal>
    </AlertDialog.Root>
  </>;
}
