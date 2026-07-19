import { Check, FilePlus2, FilePenLine, FileX2, GitBranch } from "lucide-react";
import type { WorkspaceSummary } from "../../api/types";
import { Collapsible } from "../../components/ui/Collapsible";
import styles from "./ProjectPanel.module.css";

export function GitStatus({ workspace }: { workspace?: WorkspaceSummary }) {
  const git = workspace?.git;
  return <section className={styles.section}>
    <h2>项目</h2>
    <strong className={styles.projectName}>{workspace?.name ?? "当前项目"}</strong>
    <code className={styles.path}>{workspace?.path ?? "正在读取工作区"}</code>
    {!git?.available ? <p className={styles.muted}>当前目录不是 Git 仓库</p> : <>
      <div className={styles.detailRow}><GitBranch size={15} /><code>{git.branch ?? "unknown"}</code></div>
      {git.clean ? <div className={`${styles.detailRow} ${styles.success}`}><Check size={15} />无未提交修改</div> : <Collapsible label={<span className={styles.detailRow}>{git.change_count} 个文件有修改</span>}>
        <ul className={styles.changeList}>{git.changes.map(change => <li key={change.path}>{change.status === "added" ? <FilePlus2 size={14} /> : change.status === "deleted" ? <FileX2 size={14} /> : <FilePenLine size={14} />}<code title={change.path}>{change.path}</code></li>)}</ul>
      </Collapsible>}
    </>}
  </section>;
}
