import { useRef, useState } from "react";

interface UploadStepProps {
  onUpload: (file: File) => Promise<void> | void;
  loading: boolean;
  error: string | null;
}

export function UploadStep({ onUpload, loading, error }: UploadStepProps) {
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFile = (f: File | null) => {
    setFile(f);
  };

  const handleSubmit = async () => {
    if (!file) return;
    await onUpload(file);
  };

  return (
    <div className="p-6 max-w-xl mx-auto mt-12">
      <h1 className="text-2xl font-semibold mb-1">Automate DS</h1>
      <p className="text-slate-600 text-sm mb-6">
        Upload a CSV or XLSX. We'll clean it, surface insights, and build a
        dashboard you can tweak.
      </p>

      <div
        className="bg-white border-2 border-dashed border-slate-300 rounded-lg p-8 text-center cursor-pointer hover:border-sky-500"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
        }}
        onDrop={(e) => {
          e.preventDefault();
          const dropped = e.dataTransfer.files?.[0];
          if (dropped) handleFile(dropped);
        }}
        data-testid="dropzone"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <p className="text-slate-700">
            <strong>{file.name}</strong> ({Math.round(file.size / 1024)} KB)
          </p>
        ) : (
          <p className="text-slate-500">
            Drag a file here, or click to choose a CSV / XLSX
          </p>
        )}
      </div>

      <button
        type="button"
        onClick={handleSubmit}
        disabled={!file || loading}
        data-testid="upload-button"
        className="mt-4 w-full px-4 py-3 rounded bg-sky-600 text-white font-medium hover:bg-sky-700 disabled:bg-slate-300"
      >
        {loading ? "Uploading…" : "Upload and analyze"}
      </button>

      {error && (
        <div
          className="mt-4 p-3 rounded bg-rose-50 border border-rose-200 text-rose-800 text-sm"
          data-testid="upload-error"
        >
          {error}
        </div>
      )}
    </div>
  );
}
