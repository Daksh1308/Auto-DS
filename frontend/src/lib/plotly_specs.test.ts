import { describe, expect, it } from "vitest";
import { buildPlotlySpec } from "./plotly_specs";

describe("buildPlotlySpec", () => {
  const records = [
    { d: "2024-01-01", sales: 10 },
    { d: "2024-01-02", sales: 20 },
    { d: "2024-01-03", sales: 30 },
  ];

  it("builds a line chart sorted by x (datetime)", () => {
    const fig = buildPlotlySpec("line", records, "d", "sales", "sum", "Sales");
    expect(fig.data[0].type).toBe("scatter");
    expect(fig.data[0].x).toEqual([
      "2024-01-01",
      "2024-01-02",
      "2024-01-03",
    ]);
    expect(fig.data[0].y).toEqual([10, 20, 30]);
  });

  it("builds a bar chart sorted by y desc", () => {
    const records = [
      { region: "N", sales: 10 },
      { region: "S", sales: 50 },
      { region: "E", sales: 30 },
    ];
    const fig = buildPlotlySpec("bar", records, "region", "sales", "sum", "T");
    expect(fig.data[0].type).toBe("bar");
    expect(fig.data[0].y).toEqual([50, 30, 10]);
  });

  it("builds a pie chart", () => {
    const records = [
      { c: "a" },
      { c: "a" },
      { c: "b" },
      { c: "c" },
    ];
    const fig = buildPlotlySpec("pie", records, "c", null, "count", "D");
    expect(fig.data[0].type).toBe("pie");
    // Sorted by y desc with stable order: a (2), then b and c (both 1) in input order.
    expect(fig.data[0].labels).toEqual(["a", "b", "c"]);
    expect(fig.data[0].values).toEqual([2, 1, 1]);
  });

  it("returns a placeholder for empty data", () => {
    const fig = buildPlotlySpec("line", [], "d", "sales", "sum", "Empty");
    expect(fig.data).toEqual([]);
  });

  it("throws for a line chart with no y column", () => {
    expect(() =>
      buildPlotlySpec("line", records, "d", null, "sum", "X")
    ).toThrow();
  });
});
