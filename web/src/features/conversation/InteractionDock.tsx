import { useMemo, useState } from "react";
import { CircleHelp } from "lucide-react";
import type { Interaction } from "../../api/types";
import styles from "./Conversation.module.css";

export function InteractionDock({ interaction, onSubmit }: { interaction: Interaction | null; onSubmit: (requestId: string, answer: string) => Promise<void> }) {
  const [selected, setSelected] = useState("");
  const [freeText, setFreeText] = useState("");
  const [busy, setBusy] = useState(false);
  const payload = interaction?.payload ?? {};
  const prompt = String(payload.prompt ?? "Codepilot 需要你补充信息后才能继续。");
  const options = useMemo(() => Array.isArray(payload.options) ? payload.options.map(String) : [], [payload.options]);
  const allowFreeText = payload.allow_free_text !== false;
  if (!interaction) return null;
  const answer = freeText.trim() || selected;
  const submit = async () => {
    if (!answer) return;
    setBusy(true);
    try { await onSubmit(interaction.request_id, answer); } finally { setBusy(false); }
  };
  return <section className={styles.interactionDock} aria-label="等待用户输入">
    <header><CircleHelp size={17} /><strong>Codepilot 需要确认</strong></header>
    <p>{prompt}</p>
    {options.length > 0 && <div className={styles.interactionOptions}>{options.map(option => <label key={option}><input type="radio" name={`interaction-${interaction.request_id}`} checked={selected === option} onChange={() => { setSelected(option); setFreeText(""); }} />{option}</label>)}</div>}
    {allowFreeText && <textarea value={freeText} onChange={event => { setFreeText(event.target.value); setSelected(""); }} placeholder="输入补充信息" />}
    <footer><button disabled={busy || !answer} onClick={() => void submit()}>{busy ? "提交中" : "提交并继续"}</button></footer>
  </section>;
}
