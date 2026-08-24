import { useEffect, useRef, useState } from "react";
import { Send, Square } from "lucide-react";
import { IconButton } from "../../components/ui/IconButton";
import { CommandMenu } from "./CommandMenu";
import styles from "./Conversation.module.css";

export function Composer({ busy, disabled = false, onSubmit, onCommand, onCancel }: { busy: boolean; disabled?: boolean; onSubmit: (text: string) => Promise<void>; onCommand: (command: string) => Promise<void>; onCancel: () => Promise<void> }) {
  const [text, setText] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (!disabled) input.current?.focus(); }, [disabled]);
  const resize = () => { const node = input.current; if (node) { node.style.height = "auto"; node.style.height = `${Math.min(220, node.scrollHeight)}px`; } };
  const submit = async () => { const value = text.trim(); if (!value || busy || disabled) return; await onSubmit(value); setText(""); requestAnimationFrame(resize); };
  return <div className={styles.composer}>
    <textarea ref={input} aria-label="给 Codepilot 发送消息" value={text} disabled={disabled} onChange={event => { setText(event.target.value); resize(); }} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); } }} placeholder="描述你希望分析、修改或验证的内容" />
    <footer><CommandMenu onSelect={command => void onCommand(command)} /><div className={styles.composerActions}>{busy ? <IconButton label="停止当前运行" tone="danger" onClick={() => void onCancel()}><Square size={17} fill="currentColor" /></IconButton> : <IconButton label="发送消息" disabled={!text.trim() || disabled} onClick={() => void submit()}><Send size={18} /></IconButton>}</div></footer>
  </div>;
}
