# Screenshots

Drop the actual screenshots here, then update the image references in the
main [`README.md`](../README.md).

| File name | Phase | What to capture |
| --- | --- | --- |
| `p1-cleaning-report.png` | 1 | Cleaning report panel right after uploading `Messy_Employee_dataset.csv` (rows in/out, duplicates removed, type inference, null handling) |
| `p2-insights.png` | 2 | Insights view — overview + correlations + a couple of NL insight sentences |
| `p3-dashboard.png` | 3 | The auto-suggested Plotly dashboard with all 8 charts in the 3-up grid |
| `p3-tweak.png` | 3 | The Tweak panel open on one chart showing type / x / y / aggregation dropdowns |
| `p4-chat.png` | 4 | The Chat panel mid-conversation showing answer + code + result table + chart |
| `p5-ml-results.png` | 5 | The ML results view — winner card, metrics table with the best-per-metric highlight, comparison bar chart |

## How to capture clean screenshots

1. Start the backend and frontend (see the main README).
2. Use the browser's **full-page screenshot** mode so scrollable content fits:
   - Chrome / Edge DevTools → `Ctrl+Shift+P` → "Capture full size screenshot"
   - Firefox → DevTools → `...` → "Take a fullpage screenshot"
3. Crop to the relevant panel; keep each file under ~500 KB for fast README
   loading. PNG is fine; convert to JPG with `convert p1.png -quality 85 p1.jpg`
   if you want smaller files.
4. Recommended width: **1280–1440 px** so the screenshots look sharp on both
   GitHub desktop and mobile.
5. Reference them in the README like so:
   ```markdown
   ![Cleaning report](docs/screenshots/p1-cleaning-report.png)
   ```
