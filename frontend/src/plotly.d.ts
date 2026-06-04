declare module "plotly.js-dist-min" {
  // Minimal ambient shape we use. The real Plotly API is much larger but we
  // only need the bits exercised by ChartRenderer.
  interface PlotlyData {
    type?: string;
    [key: string]: unknown;
  }
  interface PlotlyLayout {
    [key: string]: unknown;
  }
  interface PlotlyConfig {
    [key: string]: unknown;
  }
  const Plotly: {
    react: (
      el: HTMLElement,
      data: PlotlyData[],
      layout: Partial<PlotlyLayout>,
      config?: Partial<PlotlyConfig>
    ) => Promise<unknown>;
  };
  export default Plotly;
}
