import { Children, isValidElement, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import { Check, ChevronDown, ChevronUp, Copy } from "lucide-react";
import { IconButton } from "../../components/ui/IconButton";
import styles from "./Conversation.module.css";

function CodeBlock({ language, value }: { language: string; value: string }) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };
  const long = value.split("\n").length > 24;
  const rendered = language === "diff" ? value.split("\n").map((line, index) => <span key={`${index}-${line}`} className={line.startsWith("+") && !line.startsWith("+++") ? styles.diffAdded : line.startsWith("-") && !line.startsWith("---") ? styles.diffDeleted : styles.diffContext}>{line}{"\n"}</span>) : value;
  return <div className={`${styles.codeBlock} ${long && !expanded ? styles.longCode : ""}`}>
    <header><span>{language || "text"}</span><div>{long && <IconButton label={expanded ? "收起代码" : "展开代码"} onClick={() => setExpanded(value => !value)}>{expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}</IconButton>}<IconButton label={copied ? "已复制" : "复制代码"} onClick={() => void copy()}>{copied ? <Check size={15} /> : <Copy size={15} />}</IconButton></div></header>
    <pre><code className={language === "diff" ? styles.diffCode : undefined}>{rendered}</code></pre>
  </div>;
}

export function MarkdownRenderer({ children }: { children: string }) {
  return <div className="markdown"><ReactMarkdown
    remarkPlugins={[remarkGfm]}
    rehypePlugins={[rehypeSanitize]}
    skipHtml
    components={{
      pre({ children: preChildren }) {
        const child = Children.only(preChildren);
        if (!isValidElement<{ className?: string; children?: ReactNode }>(child)) return <pre>{preChildren}</pre>;
        const language = child.props.className?.replace("language-", "") ?? "";
        const value = String(child.props.children ?? "").replace(/\n$/, "");
        return <CodeBlock language={language} value={value} />;
      },
      code({ children: codeChildren, className }) {
        return <code className={className}>{codeChildren}</code>;
      },
      table({ children: tableChildren }) {
        return <div className="table-scroll"><table>{tableChildren}</table></div>;
      },
      a({ children: linkChildren, href }) {
        return <a href={href} target="_blank" rel="noreferrer">{linkChildren}</a>;
      },
    }}
  >{children}</ReactMarkdown></div>;
}
