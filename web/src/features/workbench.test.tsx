import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SessionSummary } from "../api/types";
import { MarkdownRenderer } from "./conversation/MarkdownRenderer";
import { groupSessionsByAge } from "./sessions/session-title";

function session(id: string, title: string, updatedAt: string): SessionSummary {
  return {
    session_id: id,
    title,
    workspace: "C:/repo",
    model_id: "unit/model",
    permission_mode: "workspace-write",
    current_mode: "build",
    is_running: false,
    message_count: 0,
    created_at: updatedAt,
    updated_at: updatedAt,
    status: "idle",
    pending_approvals: [],
    plan: null,
  };
}

describe("quiet workbench foundations", () => {
  it("groups sessions by recency using readable titles", () => {
    const now = new Date("2026-07-17T12:00:00Z");
    const groups = groupSessionsByAge([
      session("recent", "重构 Web 工作台", "2026-07-17T10:00:00Z"),
      session("older", "修复运行恢复", "2026-07-01T10:00:00Z"),
    ], now);
    expect(groups.map(group => group.label)).toEqual(["今天", "更早"]);
    expect(groups[0].sessions[0].title).toBe("重构 Web 工作台");
  });

  it("renders GFM safely without executing raw HTML", () => {
    render(<MarkdownRenderer>{"| 文件 | 状态 |\n| --- | --- |\n| app.tsx | 修改 |\n\n- [x] 完成\n\n<script>alert(1)</script>"}</MarkdownRenderer>);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("app.tsx")).toBeInTheDocument();
    expect(screen.getByRole("checkbox")).toBeChecked();
    expect(document.querySelector("script")).toBeNull();
  });
});
