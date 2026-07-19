import type { ReactNode } from "react";
import { DropdownMenu } from "radix-ui";
import { MoreHorizontal } from "lucide-react";
import { IconButton } from "./IconButton";
import styles from "./ui.module.css";

export type MenuItem = { label: string; onSelect: () => void; danger?: boolean; icon?: ReactNode };

export function Menu({ label = "更多操作", items }: { label?: string; items: MenuItem[] }) {
  return <DropdownMenu.Root>
    <DropdownMenu.Trigger asChild><span><IconButton label={label}><MoreHorizontal size={17} /></IconButton></span></DropdownMenu.Trigger>
    <DropdownMenu.Portal><DropdownMenu.Content align="end" sideOffset={5} className={styles.menu}>
      {items.map(item => <DropdownMenu.Item key={item.label} className={`${styles.menuItem} ${item.danger ? styles.dangerText : ""}`} onSelect={item.onSelect}>{item.icon}{item.label}</DropdownMenu.Item>)}
    </DropdownMenu.Content></DropdownMenu.Portal>
  </DropdownMenu.Root>;
}
