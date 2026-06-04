import { useState } from "react";
import type { ChartSpec, ColumnType } from "../types";
import { ChartRenderer } from "./ChartRenderer";
import { TweakPanel } from "./TweakPanel";

interface ChartCardProps {
  spec: ChartSpec;
  columns: string[];
  types: Record<string, ColumnType>;
  onChange: (next: ChartSpec) => void;
  onDelete: () => void;
}

export function ChartCard({ spec, columns, types, onChange, onDelete }: ChartCardProps) {
  const [tweaking, setTweaking] = useState(false);
  return (
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-slate-800 text-sm">{spec.title}</h3>
        <div className="flex gap-1">
          <button
            type="button"
            onClick={() => setTweaking((v) => !v)}
            className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-100"
            data-testid="toggle-tweak"
          >
            {tweaking ? "Done" : "Tweak"}
          </button>
          <button
            type="button"
            onClick={onDelete}
            className="text-xs px-2 py-1 rounded border border-rose-300 text-rose-700 hover:bg-rose-50"
            data-testid="delete-chart"
          >
            Delete
          </button>
        </div>
      </div>
      <ChartRenderer spec={spec.plotly_spec} />
      {tweaking && (
        <TweakPanel spec={spec} columns={columns} types={types} onChange={onChange} />
      )}
      <div className="text-xs text-slate-500">
        {spec.type} · {spec.aggregation} · {spec.x_column}
        {spec.y_column ? ` × ${spec.y_column}` : ""}
      </div>
    </div>
  );
}
