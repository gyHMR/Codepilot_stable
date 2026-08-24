import { CheckCircle2, XCircle } from "lucide-react";
import { Collapsible } from "../../components/ui/Collapsible";
import styles from "./Conversation.module.css";

export function RunResult({ data }: { data: Record<string, unknown> }) {
  const failed = data.status === "failed" || data.success === false;
  const blocked = failed && data.source === "core";
  const files = Array.isArray(data.files) ? data.files.map(String) : [];
  const error = data.error;
  const errorMessage = typeof error === "string"
    ? error
    : error && typeof error === "object" && "message" in error
      ? String((error as { message?: unknown }).message ?? "")
      : typeof data.message === "string" ? data.message : "";
  return <section className={`${styles.runResult} ${failed ? styles.resultFailed : ""}`}>
    <header>{failed ? <XCircle size={17} /> : <CheckCircle2 size={17} />}<strong>{failed ? blocked ? "任务未完成" : "运行失败" : String(data.summary ?? "运行完成")}</strong></header>
    {failed && errorMessage && <p className={styles.runError}>{errorMessage}</p>}
    {files.length > 0 && <Collapsible label={`修改了 ${files.length} 个文件`}><ul>{files.map(file => <li key={file}><code>{file}</code></li>)}</ul></Collapsible>}
  </section>;
}
