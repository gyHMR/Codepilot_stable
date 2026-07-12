import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatTimeline } from "./ChatTimeline";
import { Composer } from "./Composer";

describe("chat workspace components", () => {
  it("sanitizes raw HTML in assistant messages", () => {
    render(<ChatTimeline messages={[{ role: "assistant", content: "safe\n\n<script>alert(1)</script>" }]} streamingText="" />);
    expect(screen.getByText(/safe/)).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
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
});
