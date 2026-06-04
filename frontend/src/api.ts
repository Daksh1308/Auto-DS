/** Thin fetch wrappers for the backend. */

import type {
  ChatRequest,
  ChatResponse,
  CleanResponse,
  DataResponse,
  MLReportResponse,
  MLSuggestResponse,
  SuggestChartsResponse,
} from "./types";

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: string;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return (await res.json()) as T;
}

export async function uploadAndClean(
  file: File,
  formats: string[] = ["csv", "xlsx"]
): Promise<CleanResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const params = new URLSearchParams({ formats: formats.join(",") });
  const res = await fetch(`${API_BASE}/clean?${params.toString()}`, {
    method: "POST",
    body: fd,
  });
  return asJson<CleanResponse>(res);
}

export async function fetchData(
  jobId: string,
  limit: number = 5000
): Promise<DataResponse> {
  const res = await fetch(`${API_BASE}/data/${jobId}?limit=${limit}`);
  return asJson<DataResponse>(res);
}

export async function suggestCharts(
  jobId: string,
  max: number = 8
): Promise<SuggestChartsResponse> {
  const res = await fetch(`${API_BASE}/suggest-charts/${jobId}?max=${max}`, {
    method: "POST",
  });
  return asJson<SuggestChartsResponse>(res);
}

export async function sendChat(
  jobId: string,
  sessionId: string | null,
  message: string
): Promise<ChatResponse> {
  const body: ChatRequest = {
    session_id: sessionId,
    message,
  };
  const res = await fetch(`${API_BASE}/chat/${jobId}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return asJson<ChatResponse>(res);
}

export async function mlSuggest(
  jobId: string,
  target?: string | null
): Promise<MLSuggestResponse> {
  const res = await fetch(`${API_BASE}/ml/suggest/${jobId}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ target: target ?? null }),
  });
  return asJson<MLSuggestResponse>(res);
}

export async function mlTrain(
  jobId: string,
  target: string,
  features: string[]
): Promise<MLReportResponse> {
  const res = await fetch(`${API_BASE}/ml/train/${jobId}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ target, features }),
  });
  return asJson<MLReportResponse>(res);
}

export async function mlGetResult(
  jobId: string,
  target: string
): Promise<MLReportResponse> {
  const res = await fetch(
    `${API_BASE}/ml/result/${jobId}?target=${encodeURIComponent(target)}`
  );
  return asJson<MLReportResponse>(res);
}
