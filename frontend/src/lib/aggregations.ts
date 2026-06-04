/** Aggregation helpers for client-side chart rebuilding. */

import type { Aggregation } from "../types";

export function isMissing(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === "number" && Number.isNaN(v));
}

function toNumber(v: unknown): number | null {
  if (isMissing(v)) return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

export function aggregate(
  records: Array<Record<string, unknown>>,
  xColumn: string,
  yColumn: string | null,
  aggregation: Aggregation
): { x: Array<string | number>; y: Array<number | null> } {
  const groups = new Map<string, { sum: number; count: number; min: number; max: number; values: number[] }>();

  for (const r of records) {
    const xKey = String(r[xColumn] ?? "");
    if (xKey === "") continue;

    if (aggregation === "count" || yColumn === null) {
      const g = groups.get(xKey) ?? { sum: 0, count: 0, min: 0, max: 0, values: [] };
      g.count += 1;
      g.sum += 1;
      groups.set(xKey, g);
      continue;
    }

    const y = toNumber(r[yColumn]);
    if (y === null) continue;
    const g = groups.get(xKey) ?? { sum: 0, count: 0, min: y, max: y, values: [] };
    g.sum += y;
    g.count += 1;
    g.values.push(y);
    if (y < g.min) g.min = y;
    if (y > g.max) g.max = y;
    groups.set(xKey, g);
  }

  const x: Array<string | number> = [];
  const y: Array<number | null> = [];

  for (const [key, g] of groups) {
    let value: number;
    switch (aggregation) {
      case "sum":
        value = g.sum;
        break;
      case "mean":
        value = g.count ? g.sum / g.count : 0;
        break;
      case "median": {
        if (g.values.length === 0) {
          value = 0;
          break;
        }
        const sorted = [...g.values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        value =
          sorted.length % 2 === 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid];
        break;
      }
      case "min":
        value = g.min;
        break;
      case "max":
        value = g.max;
        break;
      case "count":
        value = g.count;
        break;
    }
    // Try to coerce x to a number if it looks like a date or number
    const numeric = Number(key);
    const looksNumeric =
      key !== "" && !Number.isNaN(numeric) && String(numeric) === key;
    x.push(looksNumeric ? numeric : key);
    y.push(value);
  }

  return { x, y };
}
