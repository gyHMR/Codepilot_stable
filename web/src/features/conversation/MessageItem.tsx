import type { Message } from "../../api/types";
import { MarkdownRenderer } from "./MarkdownRenderer";
import styles from "./Conversation.module.css";

type ContentBlock = Record<string, unknown>;

export function messageText(message: Message): string {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return String(message.content ?? "");
  return message.content.map(block => {
    if (typeof block === "string") return block;
    if (!block || typeof block !== "object") return "";
    const value = block as ContentBlock;
    if (typeof value.text === "string") return value.text;
    const type = String(value.type ?? "").replace(/[_-]/g, "").toLowerCase();
    if (type === "toolcall") return `正在调用工具：${String(value.name ?? "tool")}`;
    return "";
  }).filter(Boolean).join("\n\n");
}

export function MessageItem({ role, message, streaming = false }: { role: "user" | "assistant"; message: Message; streaming?: boolean }) {
  return <article className={`${styles.message} ${styles[role]} ${streaming ? styles.streaming : ""}`}>
    <header><span>{role === "user" ? "你" : "Codepilot"}</span>{streaming && <i>正在回复</i>}</header>
    <MarkdownRenderer>{messageText(message)}</MarkdownRenderer>
  </article>;
}
