import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TweakPanel } from "./TweakPanel";
import type { ChartSpec, ColumnType } from "../types";
// ChartCard is not directly tested here; placeholder for parity.

const baseSpec: ChartSpec = {
  id: "test",
  type: "bar",
  title: "Sales by region",
  x_column: "region",
  y_column: "sales",
  aggregation: "sum",
  plotly_spec: { data: [], layout: {} },
};

const columns = ["region", "sales", "subscribed"];
const types: Record<string, ColumnType> = {
  region: "categorical",
  sales: "integer",
  subscribed: "boolean",
};

describe("TweakPanel", () => {
  it("renders the spec's current values", () => {
    render(
      <TweakPanel
        spec={baseSpec}
        columns={columns}
        types={types}
        onChange={() => {}}
      />
    );
    expect(screen.getByTestId("tweak-type")).toHaveValue("bar");
    expect(screen.getByTestId("tweak-x")).toHaveValue("region");
    expect(screen.getByTestId("tweak-y")).toHaveValue("sales");
    expect(screen.getByTestId("tweak-aggregation")).toHaveValue("sum");
  });

  it("calls onChange when type switches to pie", async () => {
    const onChange = vi.fn();
    render(
      <TweakPanel
        spec={baseSpec}
        columns={columns}
        types={types}
        onChange={onChange}
      />
    );
    await userEvent.selectOptions(screen.getByTestId("tweak-type"), "pie");
    const next = onChange.mock.calls[0][0] as ChartSpec;
    expect(next.type).toBe("pie");
    expect(next.y_column).toBeNull();
    expect(next.aggregation).toBe("count");
  });

  it("calls onChange when type switches to line and forces y to a numeric", async () => {
    const onChange = vi.fn();
    const spec: ChartSpec = { ...baseSpec, type: "pie", y_column: null, aggregation: "count" };
    render(
      <TweakPanel
        spec={spec}
        columns={columns}
        types={types}
        onChange={onChange}
      />
    );
    await userEvent.selectOptions(screen.getByTestId("tweak-type"), "line");
    const next = onChange.mock.calls[0][0] as ChartSpec;
    expect(next.type).toBe("line");
    expect(next.y_column).toBe("sales");
    expect(next.aggregation).toBe("sum");
  });

  it("hides y + aggregation for pie charts", () => {
    const spec: ChartSpec = { ...baseSpec, type: "pie", y_column: null, aggregation: "count" };
    render(
      <TweakPanel
        spec={spec}
        columns={columns}
        types={types}
        onChange={() => {}}
      />
    );
    expect(screen.queryByTestId("tweak-y")).toBeNull();
    expect(screen.queryByTestId("tweak-aggregation")).toBeNull();
  });

  it("updates the x column on change", async () => {
    const onChange = vi.fn();
    render(
      <TweakPanel
        spec={baseSpec}
        columns={columns}
        types={types}
        onChange={onChange}
      />
    );
    await userEvent.selectOptions(screen.getByTestId("tweak-x"), "subscribed");
    const next = onChange.mock.calls[0][0] as ChartSpec;
    expect(next.x_column).toBe("subscribed");
  });

  it("updates aggregation on change", async () => {
    const onChange = vi.fn();
    render(
      <TweakPanel
        spec={baseSpec}
        columns={columns}
        types={types}
        onChange={onChange}
      />
    );
    await userEvent.selectOptions(screen.getByTestId("tweak-aggregation"), "mean");
    const next = onChange.mock.calls[0][0] as ChartSpec;
    expect(next.aggregation).toBe("mean");
  });
});
