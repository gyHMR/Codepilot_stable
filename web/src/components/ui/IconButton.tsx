import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Tooltip } from "radix-ui";
import styles from "./ui.module.css";

type IconButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string;
  children: ReactNode;
  tone?: "default" | "danger";
};

export function IconButton({ label, children, tone = "default", className = "", ...props }: IconButtonProps) {
  return <Tooltip.Root delayDuration={400}>
    <Tooltip.Trigger asChild>
      <button type="button" aria-label={label} className={`${styles.iconButton} ${styles[tone]} ${className}`} {...props}>{children}</button>
    </Tooltip.Trigger>
    <Tooltip.Portal><Tooltip.Content sideOffset={6} className={styles.tooltip}>{label}</Tooltip.Content></Tooltip.Portal>
  </Tooltip.Root>;
}
