import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import type { Message } from "../api/types";

type ContentBlock = Record<string, unknown>;

function contentBlockText(block: unknown): string {
  if (typeof block === "string") return block;
  if (!block || typeof block !== "object") return String(block ?? "");
  const value = block as ContentBlock;
  const type = String(value.type ?? "").replace(/[_-]/g, "").toLowerCase();
  if (typeof value.text === "string") return value.text;
  if (type === "toolcall") {
    const name = String(value.name ?? "tool");
    const args = value.arguments ?? value.raw_arguments ?? {};
    const rendered = typeof args === "string" ? args : JSON.stringify(args, null, 2);
    return `**Tool call · ${name}**\n\n\`\`\`json\n${rendered}\n\`\`\``;
  }
  return `\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\``;
}

function messageText(message: Message) {
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content.map(contentBlockText).filter(Boolean).join("\n\n");
  }
  return contentBlockText(message.content);
}

function messageLabel(message: Message) {
  if (message.role === "user") return "YOU";
  if (message.role === "toolResult") return `TOOL · ${String(message.tool_name ?? "result")}`;
  return "CODEPILOT";
}

export function ChatTimeline({ messages, streamingText }: { messages: Message[]; streamingText: string }) {
  return <div className="timeline">{messages.length === 0 && !streamingText && <div className="empty-state"><span>01</span><h1>Build with<br />full context.</h1><p>Start a session. Codepilot can inspect, plan, edit, test, and explain inside this workspace.</p></div>}
    {messages.map((message, index) => <article className={`message ${message.role === "user" ? "user" : "assistant"}`} key={String(message.id ?? index)}><header>{messageLabel(message)}</header><ReactMarkdown rehypePlugins={[rehypeSanitize]}>{messageText(message)}</ReactMarkdown></article>)}
    {streamingText && <article className="message assistant streaming"><header>CODEPILOT <i>LIVE</i></header><ReactMarkdown rehypePlugins={[rehypeSanitize]}>{streamingText}</ReactMarkdown></article>}
  </div>;
}
