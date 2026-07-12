import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import type { Message } from "../api/types";

function messageText(message: Message) { return typeof message.content === "string" ? message.content : JSON.stringify(message.content ?? "", null, 2); }

export function ChatTimeline({ messages, streamingText }: { messages: Message[]; streamingText: string }) {
  return <div className="timeline">{messages.length === 0 && !streamingText && <div className="empty-state"><span>01</span><h1>Build with<br />full context.</h1><p>Start a session. Codepilot can inspect, plan, edit, test, and explain inside this workspace.</p></div>}
    {messages.map((message, index) => <article className={`message ${message.role === "user" ? "user" : "assistant"}`} key={String(message.id ?? index)}><header>{message.role === "user" ? "YOU" : "CODEPILOT"}</header><ReactMarkdown rehypePlugins={[rehypeSanitize]}>{messageText(message)}</ReactMarkdown></article>)}
    {streamingText && <article className="message assistant streaming"><header>CODEPILOT <i>LIVE</i></header><ReactMarkdown rehypePlugins={[rehypeSanitize]}>{streamingText}</ReactMarkdown></article>}
  </div>;
}
