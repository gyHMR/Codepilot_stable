import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatTimeline } from "./ChatTimeline";
import { Composer } from "./Composer";
import { PlanPanel } from "./PlanPanel";

describe("chat workspace components", () => {
  it("sanitizes raw HTML in assistant messages", () => {
    render(<ChatTimeline messages={[{ role: "assistant", content: "safe\n\n<script>alert(1)</script>" }]} streamingText="" />);
    expect(screen.getByText(/safe/)).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
  });

  it("renders structured protocol content without dumping raw JSON", () => {
    render(<ChatTimeline messages={[
      { role: "user", content: [{ type: "text", text: "hello" }] },
      { role: "assistant", content: [{ type: "toolCall", name: "read", arguments: { path: "README.md" } }] },
      { role: "toolResult", tool_name: "read", content: [{ type: "text", text: "result text" }] },
    ]} streamingText="" />);
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText(/Tool call · read/)).toBeInTheDocument();
    expect(screen.getByText("result text")).toBeInTheDocument();
    expect(screen.getByText("TOOL · read")).toBeInTheDocument();
  });

  it("submits on enter and disables normal sending while busy", async () => {
    const submit = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(<Composer busy={false} onSubmit={submit} onCancel={vi.fn()} />);
    const input = screen.getByLabelText("Message Codepilot");
    fireEvent.change(input, { target: { value: "inspect repo" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(submit).toHaveBeenCalledWith("inspect repo");
    rerender(<Composer busy={true} onSubmit={submit} onCancel={vi.fn()} />);
    expect(screen.getByText("■ CANCEL RUN")).toBeInTheDocument();
  });

  it("renders the canonical plan revision and submits decisions", async () => {
    const approve = vi.fn().mockResolvedValue(undefined);
    const reject = vi.fn().mockResolvedValue(undefined);
    render(<PlanPanel plan={{
      plan_id: "plan:web",
      status: "proposed",
      revision: 2,
      definition: {
        summary: "完善登录模块",
        completion_criteria: ["登录测试通过"],
      },
      steps: [{
        step_id: "plan:web:step:1",
        step: "实现登录服务",
        details: "替换演示实现",
        verification: "运行登录测试",
        status: "pending",
      }],
    }} onApprove={approve} onReject={reject} />);

    expect(screen.getByText("完善登录模块")).toBeInTheDocument();
    expect(screen.getByText("实现登录服务")).toBeInTheDocument();
    expect(screen.getByText("REVISION 2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Approve plan" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Reject plan" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Reject plan" }));
    expect(approve).toHaveBeenCalledOnce();
    expect(reject).toHaveBeenCalledOnce();
  });
});
