import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MLPanel } from "./MLPanel";

vi.mock("../api", () => ({
  mlSuggest: vi.fn(),
  mlTrain: vi.fn(),
}));

vi.mock("./ChartRenderer", () => ({
  ChartRenderer: () => <div data-testid="ml-chart-stub" />,
}));

import { mlSuggest, mlTrain } from "../api";
const mlSuggestMock = mlSuggest as unknown as ReturnType<typeof vi.fn>;
const mlTrainMock = mlTrain as unknown as ReturnType<typeof vi.fn>;

function makeSuggest(target = "subscribed") {
  return {
    job_id: "job-1",
    target,
    target_type: "boolean",
    problem: {
      kind: "binary_classification",
      reason: "Target has exactly 2 unique values",
      n_classes: 2,
    },
    suggested_features: [
      { name: "age", detected_type: "integer", selected: true },
      { name: "city", detected_type: "categorical", selected: true },
      { name: "spend", detected_type: "float", selected: true },
    ],
    note: null,
  };
}

function makeReport(overrides: Partial<{
  models: Array<{
    name: string;
    metrics: Record<string, number>;
    train_seconds: number;
    is_winner: boolean;
    error: string | null;
  }>;
  problem: { kind: string; reason: string; n_classes: number | null };
  downsampled: boolean;
  rows_total: number;
  rows_used: number;
  cached: boolean;
  plotly_spec: object | null;
  error: string | null;
  note: string | null;
}> = {}) {
  return {
    job_id: "job-1",
    target: "subscribed",
    target_type: "boolean",
    problem: overrides.problem ?? {
      kind: "binary_classification",
      reason: "Target has exactly 2 unique values",
      n_classes: 2,
    },
    features: ["age", "city"],
    rows_used: overrides.rows_used ?? 80,
    rows_total: overrides.rows_total ?? 80,
    downsampled: overrides.downsampled ?? false,
    cached: overrides.cached ?? false,
    models: overrides.models ?? [
      {
        name: "LogisticRegression",
        metrics: { accuracy: 0.6, f1_macro: 0.5, roc_auc: 0.55 },
        train_seconds: 0.01,
        is_winner: false,
        error: null,
      },
      {
        name: "RandomForestClassifier",
        metrics: { accuracy: 0.7, f1_macro: 0.6, roc_auc: 0.7 },
        train_seconds: 0.1,
        is_winner: true,
        error: null,
      },
      {
        name: "HistGradientBoostingClassifier",
        metrics: { accuracy: 0.65, f1_macro: 0.55, roc_auc: 0.6 },
        train_seconds: 0.05,
        is_winner: false,
        error: null,
      },
    ],
    plotly_spec:
      overrides.plotly_spec === undefined
        ? { data: [], layout: {} }
        : overrides.plotly_spec,
    note: overrides.note ?? null,
    error: overrides.error ?? null,
  };
}

describe("MLPanel", () => {
  beforeEach(() => {
    mlSuggestMock.mockReset();
    mlTrainMock.mockReset();
  });

  it("renders nothing when closed", () => {
    render(<MLPanel jobId="job-1" open={false} onClose={() => {}} />);
    expect(screen.queryByTestId("ml-panel")).toBeNull();
  });

  it("calls mlSuggest on open and renders the configure form", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => {
      expect(mlSuggestMock).toHaveBeenCalledWith("job-1");
    });
    expect(screen.getByTestId("ml-target")).toBeInTheDocument();
    expect(screen.getByTestId("ml-run")).toBeInTheDocument();
    // Problem pill rendered
    expect(screen.getByTestId("ml-problem")).toHaveTextContent(
      /binary classification/i
    );
  });

  it("disables Run when no features are selected", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-target"));
    // Deselect all features
    const checkboxes = screen.getAllByRole("checkbox");
    for (const cb of checkboxes) await userEvent.click(cb);
    expect(screen.getByTestId("ml-run")).toBeDisabled();
  });

  it("disables Run when target is unsupported", async () => {
    mlSuggestMock.mockResolvedValueOnce({
      ...makeSuggest(),
      problem: {
        kind: "unsupported",
        reason: "Text target",
        n_classes: 5,
      },
    });
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-target"));
    expect(screen.getByTestId("ml-run")).toBeDisabled();
  });

  it("runs training and shows results with a winner and a chart", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    mlTrainMock.mockResolvedValueOnce(makeReport());
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-run"));
    await userEvent.click(screen.getByTestId("ml-run"));
    expect(mlTrainMock).toHaveBeenCalledWith("job-1", "subscribed", [
      "age",
      "city",
      "spend",
    ]);
    await waitFor(() => {
      expect(screen.getByTestId("ml-results-section")).toBeInTheDocument();
    });
    // Winner
    expect(screen.getByTestId("ml-winner-RandomForestClassifier")).toBeInTheDocument();
    // Table rows
    expect(screen.getByTestId("ml-row-LogisticRegression")).toBeInTheDocument();
    // Chart
    expect(screen.getByTestId("ml-chart-stub")).toBeInTheDocument();
  });

  it("surfaces training errors inline", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    mlTrainMock.mockResolvedValueOnce(
      makeReport({ error: "Dataset too small to train." })
    );
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-run"));
    await userEvent.click(screen.getByTestId("ml-run"));
    await waitFor(() => {
      expect(screen.getByTestId("ml-train-error")).toBeInTheDocument();
    });
    expect(screen.getByTestId("ml-train-error")).toHaveTextContent(
      "Dataset too small to train."
    );
  });

  it("surfaces suggest errors inline", async () => {
    mlSuggestMock.mockRejectedValueOnce(new Error("404 Not Found"));
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => {
      expect(screen.getByTestId("ml-error")).toBeInTheDocument();
    });
    expect(screen.getByTestId("ml-error")).toHaveTextContent("404 Not Found");
  });

  it("surfaces a thrown mlTrain error as an error message", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    mlTrainMock.mockRejectedValueOnce(new Error("Network down"));
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-run"));
    await userEvent.click(screen.getByTestId("ml-run"));
    await waitFor(() => {
      expect(screen.getByTestId("ml-error")).toBeInTheDocument();
    });
  });

  it("shows the downsampled note when report has it", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    mlTrainMock.mockResolvedValueOnce(
      makeReport({
        downsampled: true,
        rows_total: 80_000,
        rows_used: 50_000,
        note: "Downsampled from 80,000 to 50,000 rows for training.",
      })
    );
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-run"));
    await userEvent.click(screen.getByTestId("ml-run"));
    await waitFor(() => {
      expect(screen.getByTestId("ml-results-section")).toBeInTheDocument();
    });
    expect(screen.getByText(/Downsampled from 80,000/)).toBeInTheDocument();
  });

  it("shows cached flag when reloading an existing report", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    mlTrainMock.mockResolvedValueOnce(makeReport({ cached: true }));
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-run"));
    await userEvent.click(screen.getByTestId("ml-run"));
    await waitFor(() => {
      expect(screen.getByTestId("ml-results-section")).toBeInTheDocument();
    });
    expect(screen.getByText("(cached)")).toBeInTheDocument();
  });

  it("reconfigures back to the configure screen from results", async () => {
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    mlTrainMock.mockResolvedValueOnce(makeReport());
    render(<MLPanel jobId="job-1" open={true} onClose={() => {}} />);
    await waitFor(() => screen.getByTestId("ml-run"));
    await userEvent.click(screen.getByTestId("ml-run"));
    await waitFor(() => screen.getByTestId("ml-results-section"));
    await userEvent.click(screen.getByTestId("ml-reconfigure"));
    // Now in configure mode again
    await waitFor(() => {
      expect(screen.getByTestId("ml-target")).toBeInTheDocument();
    });
  });

  it("calls onClose when the X button is clicked", async () => {
    const onClose = vi.fn();
    mlSuggestMock.mockResolvedValueOnce(makeSuggest());
    render(<MLPanel jobId="job-1" open={true} onClose={onClose} />);
    await waitFor(() => screen.getByTestId("ml-close"));
    await userEvent.click(screen.getByTestId("ml-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
