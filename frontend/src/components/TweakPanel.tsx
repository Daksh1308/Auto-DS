import type { Aggregation, ChartSpec, ChartType, ColumnType } from "../types";

interface TweakPanelProps {
  spec: ChartSpec;
  columns: string[];
  types: Record<string, ColumnType>;
  onChange: (next: ChartSpec) => void;
}

const CHART_TYPES: ChartType[] = ["line", "bar", "pie"];
const AGGREGATIONS: Aggregation[] = [
  "sum",
  "mean",
  "count",
  "median",
  "min",
  "max",
];

function isNumeric(t: ColumnType): boolean {
  return t === "integer" || t === "float";
}

export function TweakPanel({ spec, columns, types, onChange }: TweakPanelProps) {
  const handleTypeChange = (type: ChartType) => {
    let next: ChartSpec = { ...spec, type };
    if (type === "pie") {
      next = { ...next, y_column: null, aggregation: "count" };
    } else if (type === "line") {
      // Line needs a y, default to first numeric if current y isn't numeric.
      if (!spec.y_column || !isNumeric(types[spec.y_column] ?? "text")) {
        const firstNumeric = columns.find((c) => isNumeric(types[c] ?? "text"));
        next = { ...next, y_column: firstNumeric ?? spec.y_column, aggregation: "sum" };
      } else {
        next = { ...next, aggregation: "sum" };
      }
    } else {
      // bar
      if (spec.y_column && !isNumeric(types[spec.y_column] ?? "text")) {
        const firstNumeric = columns.find((c) => isNumeric(types[c] ?? "text"));
        next = { ...next, y_column: firstNumeric ?? null, aggregation: "mean" };
      } else {
        next = { ...next, aggregation: next.y_column ? "mean" : "count" };
      }
    }
    onChange(next);
  };

  const handleXChange = (x: string) => {
    const xType = types[x] ?? "text";
    let next: ChartSpec = { ...spec, x_column: x };
    if (spec.type === "line" && !isNumeric(xType) && xType !== "datetime") {
      // Lines need datetime or numeric x; force y to be the first numeric.
      const firstNumeric = columns.find((c) => isNumeric(types[c] ?? "text"));
      next = { ...next, y_column: firstNumeric ?? null };
    }
    onChange(next);
  };

  const handleYChange = (y: string) => {
    onChange({ ...spec, y_column: y || null });
  };

  const handleAggregationChange = (aggregation: Aggregation) => {
    onChange({ ...spec, aggregation });
  };

  return (
    <div className="grid grid-cols-2 gap-2 text-sm">
      <label className="flex flex-col">
        <span className="text-slate-500">Type</span>
        <select
          data-testid="tweak-type"
          className="border border-slate-300 rounded px-2 py-1"
          value={spec.type}
          onChange={(e) => handleTypeChange(e.target.value as ChartType)}
        >
          {CHART_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col">
        <span className="text-slate-500">X</span>
        <select
          data-testid="tweak-x"
          className="border border-slate-300 rounded px-2 py-1"
          value={spec.x_column}
          onChange={(e) => handleXChange(e.target.value)}
        >
          {columns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
      {spec.type !== "pie" && (
        <>
          <label className="flex flex-col">
            <span className="text-slate-500">Y</span>
            <select
              data-testid="tweak-y"
              className="border border-slate-300 rounded px-2 py-1"
              value={spec.y_column ?? ""}
              onChange={(e) => handleYChange(e.target.value)}
            >
              <option value="">(count only)</option>
              {columns
                .filter((c) => isNumeric(types[c] ?? "text"))
                .map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
            </select>
          </label>
          <label className="flex flex-col">
            <span className="text-slate-500">Aggregation</span>
            <select
              data-testid="tweak-aggregation"
              className="border border-slate-300 rounded px-2 py-1"
              value={spec.aggregation}
              onChange={(e) => handleAggregationChange(e.target.value as Aggregation)}
            >
              {AGGREGATIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>
        </>
      )}
    </div>
  );
}
