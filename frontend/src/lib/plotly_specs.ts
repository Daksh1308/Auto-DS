/** Build Plotly figure dicts from a spec + records, mirroring the backend. */

import type { Aggregation, ChartType, PlotlyFigure } from "../types";
import { aggregate } from "./aggregations";

function looksLikeDatetime(xValues: Array<string | number>): boolean {
  if (xValues.length === 0) return false;
  const sample = xValues[0];
  if (typeof sample !== "string") return false;
  return /^\d{4}-\d{2}-\d{2}/.test(sample);
}

function sortedX(
  x: Array<string | number>,
  y: Array<number | null>,
  chartType: ChartType
): { x: Array<string | number>; y: Array<number | null> } {
  const pairs = x.map((xv, i) => [xv, y[i]] as [string | number, number | null]);
  if (chartType === "line" || looksLikeDatetime(x)) {
    pairs.sort((a, b) => {
      const av = a[0];
      const bv = b[0];
      if (typeof av === "number" && typeof bv === "number") return av - bv;
      return String(av).localeCompare(String(bv));
    });
  } else {
    // Bars and pies: sort by y desc so the dominant category leads.
    pairs.sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0));
  }
  return { x: pairs.map((p) => p[0]), y: pairs.map((p) => p[1]) };
}

function buildLine(
  x: Array<string | number>,
  y: Array<number | null>,
  xColumn: string,
  yColumn: string,
  title: string
): PlotlyFigure {
  return {
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x,
        y,
        name: yColumn,
      },
    ],
    layout: {
      title: { text: title },
      xaxis: { title: { text: xColumn } },
      yaxis: { title: { text: yColumn } },
      margin: { l: 50, r: 20, t: 50, b: 50 },
    },
  };
}

function buildBar(
  x: Array<string | number>,
  y: Array<number | null>,
  xColumn: string,
  yColumn: string | null,
  title: string
): PlotlyFigure {
  const xStr = x.map((v) => String(v));
  return {
    data: [
      { type: "bar", x: xStr, y, name: yColumn ?? "count" },
    ],
    layout: {
      title: { text: title },
      xaxis: { title: { text: xColumn } },
      yaxis: { title: { text: yColumn ?? "count" } },
      margin: { l: 50, r: 20, t: 50, b: 80 },
    },
  };
}

function buildPie(
  x: Array<string | number>,
  y: Array<number | null>,
  xColumn: string,
  title: string
): PlotlyFigure {
  return {
    data: [
      {
        type: "pie",
        labels: x.map((v) => String(v)),
        values: y,
        name: xColumn,
      },
    ],
    layout: {
      title: { text: title },
      margin: { l: 20, r: 20, t: 50, b: 20 },
    },
  };
}

export function buildPlotlySpec(
  chartType: ChartType,
  records: Array<Record<string, unknown>>,
  xColumn: string,
  yColumn: string | null,
  aggregation: Aggregation,
  title: string
): PlotlyFigure {
  if (chartType === "line" && yColumn === null) {
    throw new Error("line chart requires a y_column");
  }

  const { x: rawX, y: rawY } = aggregate(records, xColumn, yColumn, aggregation);
  const { x, y } = sortedX(rawX, rawY, chartType);

  if (x.length === 0) {
    return {
      data: [],
      layout: {
        title: { text: `${title} (no data)` },
        annotations: [
          {
            text: "No data for this combination",
            showarrow: false,
            xref: "paper",
            yref: "paper",
            x: 0.5,
            y: 0.5,
          },
        ],
      },
    };
  }

  if (chartType === "line") {
    return buildLine(x, y, xColumn, yColumn!, title);
  }
  if (chartType === "bar") {
    return buildBar(x, y, xColumn, yColumn, title);
  }
  return buildPie(x, y, xColumn, title);
}
