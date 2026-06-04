import { useEffect, useState } from "react";
import { mlSuggest, mlTrain } from "../api";
import type {
  ColumnType,
  MLFeatureSuggestion,
  MLProblemInfo,
  MLReportResponse,
  MLSuggestResponse,
  PlotlyFigure,
} from "../types";
import { ChartRenderer } from "./ChartRenderer";

interface MLPanelProps {
  jobId: string;
  open: boolean;
  onClose: () => void;
}

type Phase =
  | { kind: "idle" }
  | { kind: "loading-suggest" }
  | {
      kind: "configure";
      suggestion: MLSuggestResponse;
      target: string;
      features: MLFeatureSuggestion[];
    }
  | { kind: "training"; suggestion: MLSuggestResponse; target: string; features: string[] }
  | { kind: "results"; report: MLReportResponse }
  | { kind: "error"; message: string };

const PROBLEM_LABEL: Record<MLProblemInfo["kind"], string> = {
  regression: "Regression",
  binary_classification: "Binary classification",
  multiclass_classification: "Multiclass classification",
  unsupported: "Unsupported",
};

const PROBLEM_COLOR: Record<MLProblemInfo["kind"], string> = {
  regression: "bg-emerald-100 text-emerald-800",
  binary_classification: "bg-sky-100 text-sky-800",
  multiclass_classification: "bg-indigo-100 text-indigo-800",
  unsupported: "bg-rose-100 text-rose-800",
};

function ProblemPill({ problem }: { problem: MLProblemInfo }) {
  return (
    <span
      className={`inline-block text-xs font-medium px-2 py-0.5 rounded ${PROBLEM_COLOR[problem.kind]}`}
      title={problem.reason}
      data-testid="ml-problem"
    >
      {PROBLEM_LABEL[problem.kind]}
      {problem.n_classes !== null && problem.n_classes !== undefined
        ? ` (${problem.n_classes} classes)`
        : ""}
    </span>
  );
}

function formatMetric(value: number): string {
  if (Number.isNaN(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toExponential(2);
  return value.toFixed(4);
}

function ResultsTable({ report }: { report: MLReportResponse }) {
  // Collect the union of metric names across successful models.
  const metricNames = Array.from(
    new Set(
      report.models
        .filter((m) => m.error === null)
        .flatMap((m) => Object.keys(m.metrics))
    )
  );
  // Best (max) per metric — for "best" highlighting.
  const bestPerMetric: Record<string, number> = {};
  for (const name of metricNames) {
    let best = -Infinity;
    for (const m of report.models) {
      if (m.error !== null) continue;
      const v = m.metrics[name];
      if (typeof v === "number" && v > best) best = v;
    }
    bestPerMetric[name] = best;
  }
  return (
    <div className="overflow-x-auto border border-slate-200 rounded" data-testid="ml-results">
      <table className="text-xs w-full">
        <thead className="bg-slate-50">
          <tr>
            <th className="text-left px-2 py-1 font-medium text-slate-600">
              Model
            </th>
            {metricNames.map((m) => (
              <th
                key={m}
                className="text-right px-2 py-1 font-medium text-slate-600"
              >
                {m}
              </th>
            ))}
            <th className="text-right px-2 py-1 font-medium text-slate-600">
              train s
            </th>
          </tr>
        </thead>
        <tbody>
          {report.models.map((m, i) => (
            <tr
              key={m.name}
              className={
                i % 2 === 0
                  ? "bg-white"
                  : "bg-slate-50/50" +
                    (m.is_winner ? " ring-1 ring-emerald-300" : "")
              }
              data-testid={`ml-row-${m.name}`}
            >
              <td className="px-2 py-1 text-slate-800">
                {m.name}
                {m.is_winner && (
                  <span
                    className="ml-2 text-[10px] font-semibold text-emerald-700"
                    data-testid={`ml-winner-${m.name}`}
                  >
                    ★ best
                  </span>
                )}
                {m.error && (
                  <div className="text-rose-700 text-[10px]">{m.error}</div>
                )}
              </td>
              {metricNames.map((name) => {
                const v = m.metrics[name];
                const isBest =
                  typeof v === "number" && v === bestPerMetric[name];
                return (
                  <td
                    key={name}
                    className={
                      "px-2 py-1 text-right font-mono " +
                      (isBest
                        ? "text-emerald-700 font-semibold"
                        : "text-slate-700")
                    }
                    data-testid={`ml-metric-${m.name}-${name}`}
                  >
                    {typeof v === "number" ? formatMetric(v) : "—"}
                  </td>
                );
              })}
              <td className="px-2 py-1 text-right font-mono text-slate-500">
                {m.train_seconds.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MLPanel({ jobId, open, onClose }: MLPanelProps) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  // Reset on open and immediately request suggestion.
  useEffect(() => {
    if (!open) {
      setPhase({ kind: "idle" });
      return;
    }
    if (phase.kind === "idle") {
      setPhase({ kind: "loading-suggest" });
      mlSuggest(jobId)
        .then((suggestion) => {
          setPhase({
            kind: "configure",
            suggestion,
            target: suggestion.target,
            features: suggestion.suggested_features,
          });
        })
        .catch((e: unknown) => {
          setPhase({
            kind: "error",
            message: e instanceof Error ? e.message : String(e),
          });
        });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, jobId]);

  if (!open) return null;

  const runTraining = () => {
    if (phase.kind !== "configure") return;
    const selected = phase.features.filter((f) => f.selected).map((f) => f.name);
    if (!phase.target || selected.length === 0) return;
    setPhase({
      kind: "training",
      suggestion: phase.suggestion,
      target: phase.target,
      features: selected,
    });
    mlTrain(jobId, phase.target, selected)
      .then((report) => {
        setPhase({ kind: "results", report });
      })
      .catch((e: unknown) => {
        setPhase({
          kind: "error",
          message: e instanceof Error ? e.message : String(e),
        });
      });
  };

  const reconfigure = () => {
    if (phase.kind === "results") {
      setPhase({
        kind: "configure",
        suggestion: {
          job_id: phase.report.job_id,
          target: phase.report.target,
          target_type: phase.report.target_type,
          problem: phase.report.problem,
          suggested_features: phase.report.features.map((name) => ({
            name,
            detected_type: "text" as ColumnType, // re-detected on next suggest
            selected: true,
          })),
          note: null,
        },
        target: phase.report.target,
        features: phase.report.features.map((name) => ({
          name,
          detected_type: "text" as ColumnType,
          selected: true,
        })),
      });
    }
  };

  return (
    <div
      className="fixed inset-y-0 right-0 w-full max-w-md bg-white border-l border-slate-200 shadow-xl z-40 flex flex-col"
      data-testid="ml-panel"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div>
          <h2 className="text-base font-semibold text-slate-800">
            Train models
          </h2>
          <p className="text-xs text-slate-500">job {jobId}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-slate-500 hover:text-slate-800 text-sm"
          data-testid="ml-close"
          aria-label="Close ML panel"
        >
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
        {phase.kind === "loading-suggest" && (
          <div className="text-sm text-slate-500" data-testid="ml-loading-suggest">
            Loading suggestions…
          </div>
        )}

        {phase.kind === "error" && (
          <div
            className="text-sm text-rose-800 bg-rose-50 border border-rose-200 rounded p-3"
            data-testid="ml-error"
          >
            {phase.message}
          </div>
        )}

        {phase.kind === "configure" && (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">
                Target column
              </label>
              <select
                data-testid="ml-target"
                className="w-full border border-slate-300 rounded px-2 py-1 text-sm"
                value={phase.target}
                onChange={(e) =>
                  setPhase((p) =>
                    p.kind === "configure" ? { ...p, target: e.target.value } : p
                  )
                }
              >
                {[
                  phase.suggestion.target,
                  ...phase.suggestion.suggested_features.map((f) => f.name),
                ]
                  .filter((v, i, a) => a.indexOf(v) === i)
                  .map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
              </select>
              <div className="mt-2">
                <ProblemPill problem={phase.suggestion.problem} />
                <p
                  className="text-[11px] text-slate-500 mt-1"
                  title={phase.suggestion.problem.reason}
                  data-testid="ml-problem-reason"
                >
                  {phase.suggestion.problem.reason}
                </p>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">
                Features
              </label>
              <div className="space-y-1 max-h-48 overflow-y-auto border border-slate-200 rounded p-2 bg-white">
                {phase.features.map((f) => (
                  <label
                    key={f.name}
                    className="flex items-center gap-2 text-sm"
                    data-testid={`ml-feature-${f.name}`}
                  >
                    <input
                      type="checkbox"
                      className="accent-sky-600"
                      checked={f.selected}
                      onChange={(e) =>
                        setPhase((p) => {
                          if (p.kind !== "configure") return p;
                          return {
                            ...p,
                            features: p.features.map((ff) =>
                              ff.name === f.name
                                ? { ...ff, selected: e.target.checked }
                                : ff
                            ),
                          };
                        })
                      }
                    />
                    <span className="text-slate-800">{f.name}</span>
                    <span className="text-[10px] text-slate-400 font-mono">
                      {f.detected_type}
                    </span>
                  </label>
                ))}
                {phase.features.length === 0 && (
                  <div className="text-xs text-slate-500 italic">
                    No eligible features. Choose a different target.
                  </div>
                )}
              </div>
            </div>

            <button
              type="button"
              onClick={runTraining}
              disabled={
                !phase.target ||
                phase.features.filter((f) => f.selected).length === 0 ||
                phase.suggestion.problem.kind === "unsupported"
              }
              data-testid="ml-run"
              className="w-full px-3 py-2 rounded bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-700 disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
              Train 3 models
            </button>
          </div>
        )}

        {phase.kind === "training" && (
          <div className="text-sm text-slate-600" data-testid="ml-training">
            <div className="animate-pulse">Training 3 models…</div>
            <p className="text-xs text-slate-500 mt-2">
              Target: {phase.target} · Features: {phase.features.join(", ")}
            </p>
          </div>
        )}

        {phase.kind === "results" && (
          <div className="space-y-3" data-testid="ml-results-section">
            {phase.report.error && (
              <div
                className="text-sm text-rose-800 bg-rose-50 border border-rose-200 rounded p-3"
                data-testid="ml-train-error"
              >
                {phase.report.error}
              </div>
            )}

            <div>
              <div className="flex items-center gap-2 mb-2">
                <h3 className="text-sm font-semibold text-slate-800">
                  Results
                </h3>
                <ProblemPill problem={phase.report.problem} />
                {phase.report.cached && (
                  <span className="text-[10px] text-slate-400 font-mono">
                    (cached)
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-500 mb-2">
                {phase.report.rows_used.toLocaleString()} rows used{" "}
                {phase.report.downsampled && (
                  <span className="text-amber-700">
                    (downsampled from {phase.report.rows_total.toLocaleString()})
                  </span>
                )}
              </p>
              <ResultsTable report={phase.report} />
            </div>

            {phase.report.plotly_spec && (
              <div
                className="border border-slate-200 rounded p-2"
                data-testid="ml-chart"
              >
                <ChartRenderer
                  spec={phase.report.plotly_spec as PlotlyFigure}
                />
              </div>
            )}

            {phase.report.note && (
              <p className="text-[11px] text-slate-500 italic">
                {phase.report.note}
              </p>
            )}

            <button
              type="button"
              onClick={reconfigure}
              data-testid="ml-reconfigure"
              className="w-full px-3 py-2 rounded border border-slate-300 text-slate-700 text-sm font-medium hover:bg-slate-100"
            >
              Change target or features
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
