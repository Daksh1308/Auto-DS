import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatPanel } from "./ChatPanel";

// Mock the api module so we don't hit the network.
vi.mock("../api", () => ({
  sendChat: vi.fn(),
}));

// Stub the plotly renderer so jsdom doesn't need a canvas backend.
vi.mock("./ChartRenderer", () => ({
  ChartRenderer: () => <div data-testid="chart-stub" />,
}));

import { sendChat } from "../api";
const sendChatMock = sendChat as unknown as ReturnType<typeof vi.fn>;

function makeResponse(overrides: Partial<{
  answer: string;
  code: string;
  result: { type: "none" | "dataframe" | "scalar"; value: unknown; columns: string[]; records: Array<Record<string, unknown>> };
  plotly_spec: { data: Array<Record<string, unknown>>; layout: Record<string, unknown> } | null;
  error: string | null;
}> = {}) {
  return {
    job_id: "job-1",
    session_id: "sess-1",
    answer: overrides.answer ?? "Here you go.",
    code: overrides.code ?? "result = df['x'].sum()",
    result: overrides.result ?? {
      type: "scalar",
      value: 42,
      columns: [],
      records: [],
    },
    plotly_spec:
      overrides.plotly_spec === undefined
        ? null
        : overrides.plotly_spec,
    error: overrides.error === undefined ? null : overrides.error,
  };
}

describe("ChatPanel", () => {
  beforeEach(() => {
    sendChatMock.mockReset();
  });

  it("renders nothing when closed", () => {
    render(<ChatPanel jobId="job-1" open={false} onClose={() => {}} />);
    expect(screen.queryByTestId("chat-panel")).toBeNull();
  });

  it("renders an empty state when opened", () => {
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    expect(screen.getByTestId("chat-panel")).toBeInTheDocument();
    expect(
      screen.getByText(/Ask a question about your data/i)
    ).toBeInTheDocument();
  });

  it("sends a message and shows the assistant answer", async () => {
    sendChatMock.mockResolvedValueOnce(
      makeResponse({ answer: "Sum is 42.", code: "result = 42" })
    );
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    const input = screen.getByTestId("chat-input");
    await userEvent.type(input, "what is the sum?");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getByTestId("assistant-message")).toBeInTheDocument();
    });
    expect(screen.getByTestId("chat-answer")).toHaveTextContent("Sum is 42.");
    expect(sendChatMock).toHaveBeenCalledTimes(1);
    const [calledJobId, calledSessionId, calledMessage] = sendChatMock.mock.calls[0];
    expect(calledJobId).toBe("job-1");
    expect(calledSessionId).toMatch(/^sess-/);
    expect(calledMessage).toBe("what is the sum?");
  });

  it("shows the code in a collapsible details block", async () => {
    sendChatMock.mockResolvedValueOnce(makeResponse({ code: "result = 1 + 1" }));
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    await userEvent.type(screen.getByTestId("chat-input"), "x");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-code-details")).toBeInTheDocument();
    });
    expect(screen.getByTestId("chat-code").textContent).toContain("result = 1 + 1");
  });

  it("renders a dataframe result as a table", async () => {
    sendChatMock.mockResolvedValueOnce(
      makeResponse({
        result: {
          type: "dataframe",
          value: null,
          columns: ["a", "b"],
          records: [
            { a: 1, b: "x" },
            { a: 2, b: "y" },
          ],
        },
      })
    );
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    await userEvent.type(screen.getByTestId("chat-input"), "table");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getByRole("table")).toBeInTheDocument();
    });
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("y")).toBeInTheDocument();
  });

  it("renders a scalar result", async () => {
    sendChatMock.mockResolvedValueOnce(
      makeResponse({ result: { type: "scalar", value: 3.14, columns: [], records: [] } })
    );
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    await userEvent.type(screen.getByTestId("chat-input"), "scalar?");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getByTestId("assistant-message")).toBeInTheDocument();
    });
    expect(screen.getByTestId("chat-answer")).toBeInTheDocument();
    // The scalar pill should be present
    expect(screen.getByText("3.1400")).toBeInTheDocument();
  });

  it("renders a chart when plotly_spec is provided", async () => {
    sendChatMock.mockResolvedValueOnce(
      makeResponse({
        plotly_spec: {
          data: [{ type: "bar", x: ["a"], y: [1] }],
          layout: { title: { text: "demo" } },
        },
      })
    );
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    await userEvent.type(screen.getByTestId("chat-input"), "chart");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-chart")).toBeInTheDocument();
    });
  });

  it("surfaces an error returned by the backend", async () => {
    sendChatMock.mockResolvedValueOnce(
      makeResponse({ error: "Sandbox blocked the request." })
    );
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    await userEvent.type(screen.getByTestId("chat-input"), "block");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-error")).toBeInTheDocument();
    });
    expect(screen.getByTestId("chat-error")).toHaveTextContent(
      "Sandbox blocked the request."
    );
  });

  it("surfaces a thrown error as a chat error", async () => {
    sendChatMock.mockRejectedValueOnce(new Error("Network down"));
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    await userEvent.type(screen.getByTestId("chat-input"), "x");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-error")).toBeInTheDocument();
    });
    expect(screen.getByTestId("chat-error")).toHaveTextContent("Network down");
  });

  it("reuses the same session id across multiple sends", async () => {
    sendChatMock.mockResolvedValueOnce(
      makeResponse({ answer: "first" })
    );
    sendChatMock.mockResolvedValueOnce(
      makeResponse({ answer: "second" })
    );
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    await userEvent.type(screen.getByTestId("chat-input"), "1");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getAllByTestId("assistant-message")).toHaveLength(1);
    });
    await userEvent.type(screen.getByTestId("chat-input"), "2");
    await userEvent.click(screen.getByTestId("chat-send"));
    await waitFor(() => {
      expect(screen.getAllByTestId("assistant-message")).toHaveLength(2);
    });
    // First call: locally-generated session id (sess- prefix).
    const firstSession = sendChatMock.mock.calls[0][1];
    expect(firstSession).toMatch(/^sess-/);
    // Second call: server-echoed session id from the first response.
    const secondSession = sendChatMock.mock.calls[1][1];
    expect(secondSession).toBe("sess-1");
  });

  it("sends on Enter but not on Shift+Enter", async () => {
    sendChatMock.mockResolvedValue(makeResponse({ answer: "ok" }));
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    const input = screen.getByTestId("chat-input");
    input.focus();
    // Shift+Enter should not submit
    await userEvent.keyboard("{Shift>}{Enter}{/Shift}");
    expect(sendChatMock).not.toHaveBeenCalled();
    // Plain Enter on a non-empty input should submit
    await userEvent.type(input, "hello");
    await userEvent.keyboard("{Enter}");
    await waitFor(() => {
      expect(sendChatMock).toHaveBeenCalledTimes(1);
    });
  });

  it("does not send an empty message", async () => {
    render(<ChatPanel jobId="job-1" open={true} onClose={() => {}} />);
    const button = screen.getByTestId("chat-send");
    expect(button).toBeDisabled();
    const input = screen.getByTestId("chat-input");
    await userEvent.type(input, "   ");
    expect(button).toBeDisabled();
  });

  it("calls onClose when the X button is clicked", async () => {
    const onClose = vi.fn();
    render(<ChatPanel jobId="job-1" open={true} onClose={onClose} />);
    await userEvent.click(screen.getByTestId("chat-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
