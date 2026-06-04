import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { PlotlyFigure } from "../types";

interface ChartRendererProps {
  spec: PlotlyFigure;
  className?: string;
}

export function ChartRenderer({ spec, className }: ChartRendererProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    Plotly.react(el, spec.data, spec.layout, {
      responsive: true,
      displaylogo: false,
    });
  }, [spec]);

  return <div ref={ref} className={className ?? "h-72 w-full"} />;
}
