import { describe, expect, it } from "vitest";
import { aggregate } from "./aggregations";

describe("aggregate", () => {
  const records = [
    { c: "a", v: 10 },
    { c: "a", v: 20 },
    { c: "b", v: 5 },
    { c: "b", v: 15 },
    { c: "a", v: 30 },
  ];

  it("sums by category", () => {
    const out = aggregate(records, "c", "v", "sum");
    const map = Object.fromEntries(out.x.map((k, i) => [k, out.y[i]]));
    expect(map).toEqual({ a: 60, b: 20 });
  });

  it("computes mean by category", () => {
    const out = aggregate(records, "c", "v", "mean");
    const map = Object.fromEntries(out.x.map((k, i) => [k, out.y[i]]));
    expect(map).toEqual({ a: 20, b: 10 });
  });

  it("finds min and max by category", () => {
    const minOut = aggregate(records, "c", "v", "min");
    const maxOut = aggregate(records, "c", "v", "max");
    expect(Object.fromEntries(minOut.x.map((k, i) => [k, minOut.y[i]]))).toEqual({
      a: 10,
      b: 5,
    });
    expect(Object.fromEntries(maxOut.x.map((k, i) => [k, maxOut.y[i]]))).toEqual({
      a: 30,
      b: 15,
    });
  });

  it("computes median by category", () => {
    const records = [
      { c: "a", v: 1 },
      { c: "a", v: 2 },
      { c: "a", v: 3 },
      { c: "b", v: 10 },
      { c: "b", v: 20 },
    ];
    const out = aggregate(records, "c", "v", "median");
    const map = Object.fromEntries(out.x.map((k, i) => [k, out.y[i]]));
    expect(map).toEqual({ a: 2, b: 15 });
  });

  it("counts when y is null or aggregation is count", () => {
    const records = [
      { c: "a" },
      { c: "a" },
      { c: "b" },
      { c: "c" },
      { c: "c" },
      { c: "c" },
    ];
    const outNull = aggregate(records, "c", null, "count");
    const map = Object.fromEntries(outNull.x.map((k, i) => [k, outNull.y[i]]));
    expect(map).toEqual({ a: 2, b: 1, c: 3 });
  });

  it("skips rows with empty x", () => {
    const records = [
      { c: "a", v: 1 },
      { c: "", v: 99 },
      { c: "b", v: 2 },
    ];
    const out = aggregate(records, "c", "v", "sum");
    expect(out.x).toEqual(["a", "b"]);
    expect(out.y).toEqual([1, 2]);
  });

  it("skips rows where y is not numeric for non-count aggregations", () => {
    const records = [
      { c: "a", v: 1 },
      { c: "a", v: "x" as unknown as number },
      { c: "b", v: 2 },
    ];
    const out = aggregate(records, "c", "v", "sum");
    const map = Object.fromEntries(out.x.map((k, i) => [k, out.y[i]]));
    expect(map).toEqual({ a: 1, b: 2 });
  });
});
