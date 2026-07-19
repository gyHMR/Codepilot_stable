import type { ReactNode } from "react";
import { Collapsible as RadixCollapsible } from "radix-ui";
import { ChevronDown } from "lucide-react";
import styles from "./ui.module.css";

export function Collapsible({ label, children, defaultOpen = false }: { label: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  return <RadixCollapsible.Root defaultOpen={defaultOpen} className={styles.collapsible}>
    <RadixCollapsible.Trigger className={styles.collapsibleTrigger}>{label}<ChevronDown size={15} /></RadixCollapsible.Trigger>
    <RadixCollapsible.Content className={styles.collapsibleContent}>{children}</RadixCollapsible.Content>
  </RadixCollapsible.Root>;
}
