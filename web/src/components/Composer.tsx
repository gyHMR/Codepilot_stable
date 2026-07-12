import { useState } from "react";

export function Composer({ busy, onSubmit, onCancel }: { busy: boolean; onSubmit: (text: string) => Promise<void>; onCancel: () => Promise<void> }) {
  const [text, setText] = useState("");
  const submit = async () => { const value = text.trim(); if (!value || busy) return; await onSubmit(value); setText(""); };
  return <div className="composer"><textarea aria-label="Message Codepilot" value={text} onChange={event => setText(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); } }} placeholder="Describe what you want to build, inspect, or fix…" /><div className="composer-bar"><span>ENTER TO SEND · SHIFT+ENTER FOR LINE</span>{busy ? <button className="cancel" onClick={() => void onCancel()}>■ CANCEL RUN</button> : <button className="send" onClick={() => void submit()}>SEND ↗</button>}</div></div>;
}
