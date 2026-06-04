import { useState } from "react";
import type {
  ChartSpec,
  ChartType,
  ColumnType,
  CleaningReport,
} from "../types";
import { buildPlotlySpec } from "../lib/plotly_specs";
import { ChartCard } from "./ChartCard";
import { ChatPanel } from "./ChatPanel";
import { MLPanel } from "./MLPanel";

interface DashboardStepProps {
  jobId: string;
  cleaningReport: CleaningReport | null;
  records: Array<Record<string, unknown>>;
  columns: string[];
  types: Record<string, ColumnType>;
  initialSpecs: ChartSpec[];
}

function genId(): string {
  return Math.random().toString(36).slice(2, 10);
}

const CHART_TYPES: ChartType[] = ["line", "bar", "pie"];

export function DashboardStep({
  jobId,
  cleaningReport,
  records,
  columns,
  types,
  initialSpecs,
}: DashboardStepProps) {
  const [specs, setSpecs] = useState<ChartSpec[]>(initialSpecs);
  const [adding, setAdding] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [mlOpen, setMlOpen] = useState(false);
  const [newType, setNewType] = useState<ChartType>("bar");
  const [newX, setNewX] = useState<string>(columns[0] ?? "");
  const [newY, setNewY] = useState<string>(
    columns.find((c) => types[c] === "integer" || types[c] === "float") ?? ""
  );

  const updateSpec = (next: ChartSpec) => {
    setSpecs((curr) =>
      curr.map((s) => {
        if (s.id !== next.id) return s;
        // Rebuild plotly_spec from the canonical fields so it can't drift.
        const rebuilt: ChartSpec = {
          ...next,
          plotly_spec: buildPlotlySpec(
            next.type,
            records,
            next.x_column,
            next.y_column,
            next.aggregation,
            next.title
          ),
        };
        return rebuilt;
      })
    );
  };

  const deleteSpec = (id: string) => {
    setSpecs((curr) => curr.filter((s) => s.id !== id));
  };

  const addSpec = () => {
    if (!newX) return;
    const y = newType === "pie" ? null : newY || null;
    const aggregation =
      newType === "pie" ? "count" : y ? "mean" : "count";
    const title =
      newType === "pie"
        ? `${newX} distribution`
        : y
        ? `${aggregation} ${y} by ${newX}`
        : `Count by ${newX}`;
    const spec: ChartSpec = {
      id: genId(),
      type: newType,
      title,
      x_column: newX,
      y_column: y,
      aggregation,
      plotly_spec: buildPlotlySpec(newType, records, newX, y, aggregation, title),
    };
    setSpecs((curr) => [...curr, spec]);
    setAdding(false);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <p className="text-sm text-slate-500">
            job <code className="text-xs">{jobId}</code> · {records.length} rows ·{" "}
            {columns.length} columns
            {cleaningReport && (
              <>
                {" "}
                · {cleaningReport.duplicates_removed} dupes removed
                {cleaningReport.dropped_columns.length > 0 &&
                  ` · ${cleaningReport.dropped_columns.length} columns dropped`}
              </>
            )}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setAdding((v) => !v)}
          className="px-3 py-2 rounded bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-700"
          data-testid="toggle-add"
        >
          {adding ? "Cancel" : "+ Add chart"}
        </button>
        <button
          type="button"
          onClick={() => setChatOpen(true)}
          className="ml-2 px-3 py-2 rounded bg-sky-600 text-white text-sm font-medium hover:bg-sky-700"
          data-testid="toggle-chat"
        >
          Chat with your data
        </button>
        <button
          type="button"
          onClick={() => setMlOpen(true)}
          className="ml-2 px-3 py-2 rounded bg-purple-600 text-white text-sm font-medium hover:bg-purple-700"
          data-testid="toggle-ml"
        >
          Train models
        </button>
      </div>

      {adding && (
        <div className="bg-white border border-slate-200 rounded-lg p-4 mb-4 flex flex-wrap gap-3 items-end">
          <label className="flex flex-col text-sm">
            <span className="text-slate-500">Type</span>
            <select
              data-testid="add-type"
              className="border border-slate-300 rounded px-2 py-1"
              value={newType}
              onChange={(e) => setNewType(e.target.value as ChartType)}
            >
              {CHART_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col text-sm">
            <span className="text-slate-500">X</span>
            <select
              data-testid="add-x"
              className="border border-slate-300 rounded px-2 py-1"
              value={newX}
              onChange={(e) => setNewX(e.target.value)}
            >
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          {newType !== "pie" && (
            <label className="flex flex-col text-sm">
              <span className="text-slate-500">Y (numeric)</span>
              <select
                data-testid="add-y"
                className="border border-slate-300 rounded px-2 py-1"
                value={newY}
                onChange={(e) => setNewY(e.target.value)}
              >
                <option value="">(count only)</option>
                {columns
                  .filter(
                    (c) => types[c] === "integer" || types[c] === "float"
                  )
                  .map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
              </select>
            </label>
          )}
          <button
            type="button"
            onClick={addSpec}
            className="px-3 py-2 rounded bg-sky-600 text-white text-sm hover:bg-sky-700"
            data-testid="add-chart"
          >
            Add
          </button>
        </div>
      )}

      <div
        className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4"
        data-testid="dashboard-grid"
      >
        {specs.map((spec) => (
          <ChartCard
            key={spec.id}
            spec={spec}
            columns={columns}
            types={types}
            onChange={updateSpec}
            onDelete={() => deleteSpec(spec.id)}
          />
        ))}
      </div>

      <ChatPanel
        jobId={jobId}
        open={chatOpen}
        onClose={() => setChatOpen(false)}
      />

      <MLPanel
        jobId={jobId}
        open={mlOpen}
        onClose={() => setMlOpen(false)}
      />
    </div>
  );
}
