import { useState } from "react";
import { fetchData, suggestCharts, uploadAndClean } from "./api";
import type {
  ChartSpec,
  CleaningReport,
  ColumnType,
} from "./types";
import { UploadStep } from "./components/UploadStep";
import { DashboardStep } from "./components/DashboardStep";

type AppState =
  | { kind: "upload" }
  | {
      kind: "dashboard";
      jobId: string;
      cleaningReport: CleaningReport | null;
      records: Array<Record<string, unknown>>;
      columns: string[];
      types: Record<string, ColumnType>;
      specs: ChartSpec[];
    };

export default function App() {
  const [state, setState] = useState<AppState>({ kind: "upload" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const clean = await uploadAndClean(file);
      const [data, suggestions] = await Promise.all([
        fetchData(clean.job_id, 5000),
        suggestCharts(clean.job_id, 8),
      ]);
      setState({
        kind: "dashboard",
        jobId: clean.job_id,
        cleaningReport: clean.report,
        records: data.records,
        columns: data.columns,
        types: suggestions.types,
        specs: suggestions.specs,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setState({ kind: "upload" });
    setError(null);
  };

  if (state.kind === "upload") {
    return (
      <UploadStep onUpload={handleUpload} loading={loading} error={error} />
    );
  }

  return (
    <div>
      <div className="border-b border-slate-200 bg-white">
        <div className="max-w-7xl mx-auto p-3 flex justify-between items-center">
          <h1 className="font-semibold text-slate-800">Automate DS — Dashboard</h1>
          <button
            type="button"
            onClick={handleReset}
            className="text-sm text-slate-600 hover:text-slate-900"
          >
            ← New upload
          </button>
        </div>
      </div>
      <DashboardStep
        jobId={state.jobId}
        cleaningReport={state.cleaningReport}
        records={state.records}
        columns={state.columns}
        types={state.types}
        initialSpecs={state.specs}
      />
    </div>
  );
}
