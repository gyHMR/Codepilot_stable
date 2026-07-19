import { useState } from "react";
import { Popover } from "radix-ui";
import { ListTree, Search } from "lucide-react";
import { IconButton } from "../../components/ui/IconButton";
import styles from "./Conversation.module.css";

const commands = [
  { command: "/plan", label: "查看或管理任务计划" },
  { command: "/context", label: "查看当前上下文" },
  { command: "/memory", label: "管理长期记忆" },
  { command: "/help", label: "查看可用命令" },
];

export function CommandMenu({ onSelect }: { onSelect: (command: string) => void }) {
  const [query, setQuery] = useState("");
  const filtered = commands.filter(item => `${item.command} ${item.label}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  return <Popover.Root><Popover.Trigger asChild><span><IconButton label="打开命令菜单"><ListTree size={18} /></IconButton></span></Popover.Trigger>
    <Popover.Portal><Popover.Content align="start" side="top" sideOffset={8} className={styles.commandMenu}>
      <header>命令</header><label><Search size={14} /><input aria-label="搜索命令" value={query} onChange={event => setQuery(event.target.value)} placeholder="搜索命令" /></label>{filtered.map(item => <Popover.Close key={item.command} asChild><button onClick={() => onSelect(item.command)}><code>{item.command}</code><span>{item.label}</span></button></Popover.Close>)}
    </Popover.Content></Popover.Portal>
  </Popover.Root>;
}
