import { useEffect, useRef, useState } from "react";
import { sendChat } from "../api";
import type { ChatResponse, PlotlyFigure } from "../types";
import { ChartRenderer } from "./ChartRenderer";

interface ChatPanelProps {
  jobId: string;
  open: boolean;
  onClose: () => void;
}

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
  loading?: boolean;
}

function genId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function genSessionId(): string {
  // Reasonable opaque id — server accepts any string.
  return (
    "sess-" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
  );
}

function ResultTable({
  columns,
  records,
}: {
  columns: string[];
  records: Array<Record<string, unknown>>;
}) {
  if (columns.length === 0 || records.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic">Empty result</div>
    );
  }
  return (
    <div className="overflow-x-auto max-h-64 border border-slate-200 rounded">
      <table className="text-xs w-full">
        <thead className="bg-slate-50 sticky top-0">
          <tr>
            {columns.map((c) => (
              <th
                key={c}
                className="text-left px-2 py-1 font-medium text-slate-600"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {records.map((r, i) => (
            <tr
              key={i}
              className={i % 2 === 0 ? "bg-white" : "bg-slate-50/50"}
            >
              {columns.map((c) => (
                <td key={c} className="px-2 py-1 text-slate-700">
                  {r[c] === null || r[c] === undefined
                    ? "—"
                    : String(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ScalarResult({ value }: { value: unknown }) {
  let display: string;
  if (value === null || value === undefined) {
    display = "(no result)";
  } else if (typeof value === "number") {
    display = Number.isInteger(value) ? value.toString() : value.toFixed(4);
  } else if (typeof value === "boolean") {
    display = value ? "true" : "false";
  } else {
    display = String(value);
  }
  return (
    <div className="inline-block bg-slate-100 rounded px-3 py-1 text-sm font-mono text-slate-800">
      {display}
    </div>
  );
}

function AssistantMessage({ resp }: { resp: ChatResponse }) {
  return (
    <div className="space-y-2" data-testid="assistant-message">
      {resp.answer && (
        <div className="text-sm text-slate-800" data-testid="chat-answer">
          {resp.answer}
        </div>
      )}
      {resp.error && (
        <div
          className="text-xs text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1"
          data-testid="chat-error"
        >
          {resp.error}
        </div>
      )}
      {resp.code && (
        <details className="text-xs" data-testid="chat-code-details">
          <summary className="cursor-pointer text-slate-500 hover:text-slate-700">
            View code
          </summary>
          <pre
            className="mt-1 bg-slate-900 text-slate-100 rounded p-2 overflow-x-auto text-[11px]"
            data-testid="chat-code"
          >
            {resp.code}
          </pre>
        </details>
      )}
      {resp.result.type === "dataframe" && (
        <ResultTable
          columns={resp.result.columns}
          records={resp.result.records}
        />
      )}
      {resp.result.type === "scalar" && (
        <ScalarResult value={resp.result.value} />
      )}
      {resp.plotly_spec && (
        <div className="border border-slate-200 rounded" data-testid="chat-chart">
          <ChartRenderer spec={resp.plotly_spec as PlotlyFigure} />
        </div>
      )}
    </div>
  );
}

export function ChatPanel({ jobId, open, onClose }: ChatPanelProps) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Reset session when the panel is opened fresh; keep history within a session.
  useEffect(() => {
    if (open && sessionId === null) {
      setSessionId(genSessionId());
    }
  }, [open, sessionId]);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [turns]);

  const send = async () => {
    const message = draft.trim();
    if (!message || sending) return;
    const userTurn: ChatTurn = { id: genId(), role: "user", text: message };
    const placeholderId = genId();
    setTurns((curr) => [
      ...curr,
      userTurn,
      { id: placeholderId, role: "assistant", text: "", loading: true },
    ]);
    setDraft("");
    setSending(true);
    try {
      const resp = await sendChat(jobId, sessionId, message);
      setSessionId(resp.session_id);
      setTurns((curr) =>
        curr.map((t) =>
          t.id === placeholderId
            ? { ...t, loading: false, response: resp, text: resp.answer }
            : t
        )
      );
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      setTurns((curr) =>
        curr.map((t) =>
          t.id === placeholderId
            ? {
                ...t,
                loading: false,
                text: "",
                response: {
                  job_id: jobId,
                  session_id: sessionId ?? "",
                  answer: "",
                  code: "",
                  result: { type: "none", value: null, columns: [], records: [] },
                  plotly_spec: null,
                  error: err,
                },
              }
            : t
        )
      );
    } finally {
      setSending(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-y-0 right-0 w-full max-w-md bg-white border-l border-slate-200 shadow-xl z-40 flex flex-col"
      data-testid="chat-panel"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div>
          <h2 className="text-base font-semibold text-slate-800">
            Chat with your data
          </h2>
          <p className="text-xs text-slate-500">job {jobId}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-slate-500 hover:text-slate-800 text-sm"
          data-testid="chat-close"
          aria-label="Close chat"
        >
          ✕
        </button>
      </div>
      <div
        ref={listRef}
        className="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-50"
        data-testid="chat-messages"
      >
        {turns.length === 0 && (
          <div className="text-sm text-slate-500 italic">
            Ask a question about your data, e.g.{" "}
            <span className="font-mono text-xs">
              "mean of total_amount by region"
            </span>
            .
          </div>
        )}
        {turns.map((t) => (
          <div
            key={t.id}
            className={
              t.role === "user"
                ? "flex justify-end"
                : "flex justify-start"
            }
          >
            <div
              className={
                t.role === "user"
                  ? "max-w-[85%] bg-sky-600 text-white rounded-lg px-3 py-2 text-sm"
                  : "max-w-[90%] bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-sm w-full"
              }
            >
              {t.role === "user" ? (
                <div className="whitespace-pre-wrap">{t.text}</div>
              ) : t.loading ? (
                <div className="text-xs text-slate-500 italic" data-testid="chat-loading">
                  Thinking…
                </div>
              ) : t.response ? (
                <AssistantMessage resp={t.response} />
              ) : null}
            </div>
          </div>
        ))}
      </div>
      <div className="border-t border-slate-200 p-3 bg-white">
        <textarea
          rows={2}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask a question…  (Enter to send, Shift+Enter for newline)"
          className="w-full resize-none border border-slate-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-sky-400"
          data-testid="chat-input"
          disabled={sending}
        />
        <div className="flex items-center justify-between mt-2">
          <span className="text-[11px] text-slate-400">
            {sessionId ? `session ${sessionId.slice(0, 12)}…` : ""}
          </span>
          <button
            type="button"
            onClick={send}
            disabled={sending || draft.trim().length === 0}
            className="px-3 py-1.5 rounded bg-sky-600 text-white text-sm font-medium hover:bg-sky-700 disabled:bg-slate-300 disabled:cursor-not-allowed"
            data-testid="chat-send"
          >
            {sending ? "Sending…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
