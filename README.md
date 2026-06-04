# Automate DS

A multi-phase data automation tool: upload a CSV/XLSX, get a cleaned
file + insights + an interactive dashboard + an AI chat that writes
pandas against your data + one-click model comparison.

```
upload → clean → insights → dashboard → chat / train models
 (P1)    (P1)    (P2)       (P3)        (P4)     (P5)
```

## Quick start

You need **two terminals** — the backend (Python) and the frontend
(Vite/React).

### 1. Backend

```bash
cd "Automate DS/backend"
pip install -r requirements.txt

# Run the API on port 8001
PYTHONPATH=. python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
# Swagger UI: http://127.0.0.1:8001/docs
```

Required Python: **3.13+**. Key packages: FastAPI, pandas, scikit-learn
(1.5.2), httpx.

### 2. Frontend

```bash
cd "Automate DS/frontend"
npm install
npm run dev
# Open http://127.0.0.1:5173
```

The Vite dev server proxies `/api/*` → `http://127.0.0.1:8001`, so the
frontend talks to the backend transparently.

### Optional — LLM key for the chat (Phase 4)

The chat endpoint talks to any OpenAI-compatible API. Default is Groq;
override via env vars on the backend:

| Variable | Default |
| --- | --- |
| `OPENAI_API_KEY` | _(none — required for live chat)_ |
| `OPENAI_BASE_URL` | `https://api.groq.com/openai/v1` |
| `OPENAI_MODEL` | `llama-3.3-70b-versatile` |
| `OPENAI_TIMEOUT_S` | `30` |
| `ADS_CHAT_CODE_TIMEOUT_S` | `10` (sandbox timeout) |

When the key is missing, the panel surfaces a clear message instead of
crashing.

---

## The 5 phases

| # | Phase | What it adds |
| --- | --- | --- |
| 1 | **Data Cleaner** | CSV/XLSX upload → cleaned file (CSV + XLSX) + JSON report |
| 2 | **Smart Insights** | Numeric stats, correlations, trends, category breakdowns, NL sentences |
| 3 | **Dashboard Builder** | Auto-suggested Plotly charts with a tweak panel (type / columns / aggregation) |
| 4 | **AI Chat** | Natural-language questions → LLM writes pandas code → sandboxed exec → text + table + chart |
| 5 | **Smart Model Suggestions** | One-click AutoML-lite: 3 sklearn models, 80/20 split, metric comparison + winner |

The frontend is a single page with three optional side panels (chat, ML
training) that you can open from the dashboard header.

---

## Architecture

```
Automate DS/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app
│   │   ├── api/
│   │   │   ├── routes.py       # All HTTP endpoints
│   │   │   └── schemas.py      # Pydantic request/response models
│   │   ├── core/
│   │   │   ├── config.py       # Env-driven settings
│   │   │   └── jobs.py         # In-memory cleaned-Dataset job store
│   │   ├── cleaner/            # Phase 1: cleaning pipeline
│   │   ├── insights/           # Phase 2: stats, trends, correlations
│   │   ├── dashboards/         # Phase 3: chart suggestions + Plotly specs
│   │   ├── chat/               # Phase 4: schema, sandbox, executor, LLM client, session
│   │   └── models/             # Phase 5: problem detection, preprocess, train, registry, cache
│   ├── tests/                  # 263 pytest tests
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── api.ts              # Thin fetch wrappers
    │   ├── types.ts            # Mirrors backend pydantic schemas
    │   ├── lib/
    │   │   ├── aggregations.ts # group-by + aggregate (mirrors backend)
    │   │   └── plotly_specs.ts # Build Plotly figures from a ChartSpec
    │   ├── components/
    │   │   ├── UploadStep.tsx
    │   │   ├── DashboardStep.tsx
    │   │   ├── ChartCard.tsx
    │   │   ├── ChartRenderer.tsx
    │   │   ├── TweakPanel.tsx
    │   │   ├── ChatPanel.tsx
    │   │   └── MLPanel.tsx
    │   ├── App.tsx
    │   └── main.tsx
    ├── package.json
    └── vite.config.ts          # Proxies /api → 127.0.0.1:8001
```

---

## Phase 1 — Data Cleaner + Formatter

Upload a CSV or XLSX, get a cleaned file back (CSV and/or XLSX) plus a
JSON cleaning report.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET`  | `/healthz` | Liveness check |
| `POST` | `/clean` | Upload a file, receive a `job_id` + cleaning report |
| `GET`  | `/download/{job_id}/{csv\|xlsx}` | Stream the cleaned file |

### Upload

```bash
curl -X POST http://127.0.0.1:8001/clean \
  -F "file=@Automate DS/backend/tests/fixtures/dirty.csv" \
  -F "formats=csv,xlsx"
```

```json
{
  "job_id": "5d88...",
  "report": { "rows_in": 12, "rows_out": 11, "duplicates_removed": 1, ... },
  "downloads": { "csv": "/download/5d88.../csv", "xlsx": "/download/5d88.../xlsx" }
}
```

### What the cleaning pipeline does

- Type inference per column (`integer`, `float`, `datetime`, `boolean`,
  `categorical`, `text`)
- Whitespace + case normalization for object columns
- Null handling: median (numeric) / mode (categorical) / NaT (datetime)
- Datetime coercion with `format="mixed"`
- Drop columns that are > 90 % null
- Strip diacritics, drop duplicates, snake_case rename
- Returns the cleaned `DataFrame` + a structured `CleaningReport`

---

## Phase 2 — Smart Insights

After cleaning, surface deterministic, ML-free insights about the
data — no LLM, no auto-generated "AI" prose that could hallucinate.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/insights` | Upload + clean + insights in one call |
| `POST` | `/insights/{job_id}` | Run insights against a previously cleaned job |

### What you get back

- `overview` — row / col counts + by-type histogram
- `columns` — per-column stats (mean, std, min/max, unique counts, …)
- `correlations` — Pearson r pairs where `|r| ≥ 0.5` (configurable)
- `trends` — period-over-period change for any numeric column, with
  a sensible binning strategy (quarters / months / weeks)
- `category_breakdowns` — top/bottom categories with pct diff
- `insight_sentences` — up to 10 ranked NL sentences summarizing the
  most interesting findings

---

## Phase 3 — Dashboard Builder

Auto-suggested Plotly charts with a tweak panel that lets you change
the chart type, columns, and aggregation. The data flow is:

1. Backend suggests 8 charts (ranked by CV + variety).
2. Frontend renders them in a 3-up grid.
3. User clicks **Tweak** → pick a different type / x / y / aggregation.
4. Frontend rebuilds the `plotly_spec` client-side and re-renders.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET`  | `/data/{job_id}?limit=N` | Records as JSON (sampled at 5,000 by default) |
| `POST` | `/suggest-charts/{job_id}?max=N` | Initial 8 specs + column types |

### Frontend pieces

- `lib/plotly_specs.ts` — build Plotly figure dicts from a `ChartSpec`
  (mirrors the backend logic so charts stay in sync).
- `ChartCard` / `TweakPanel` — minimal tweak surface. Adding a chart
  uses the same dropdowns.

### What the suggester picks

- `mean` aggregation for bars (so bar charts feel right for typical
  numeric data)
- Lines rank first when there's a datetime column with enough
  variability (CV + variety score)
- Pies of high-cardinality categoricals next
- Bars last

---

## Phase 4 — AI Chat with Data

Ask a question in natural language. The LLM writes pandas code against
the cleaned `DataFrame`, the backend runs it inside a sandbox, and the
answer comes back as **text + table + auto-chart**. Multi-turn
sessions preserve the last 3 user + 3 assistant messages.

### Endpoints

| Method | Path | Body | Purpose |
| --- | --- | --- | --- |
| `POST` | `/chat/{job_id}` | `{"session_id?", "message"}` | Send a question; receive code + answer + result + chart |

### Configuration (env vars)

| Variable | Default |
| --- | --- |
| `OPENAI_API_KEY` | _(none — required for live chat)_ |
| `OPENAI_BASE_URL` | `https://api.groq.com/openai/v1` |
| `OPENAI_MODEL` | `llama-3.3-70b-versatile` |
| `OPENAI_TIMEOUT_S` | `30` |
| `ADS_CHAT_CODE_TIMEOUT_S` | `10` (sandbox timeout) |

### Sandbox

LLM-generated code is validated with an AST pass and then executed in
a restricted `exec` environment.

Forbidden:
- `import` / `from … import …` of `os`, `sys`, `subprocess`, `socket`,
  `requests`, `urllib`, `http`, `asyncio`, `shutil`, `pathlib`,
  `tempfile`, `ctypes`, `multiprocessing`, `threading`.
- Calls to `open`, `exec`, `eval`, `compile`, `globals`, `locals`,
  `__import__`, `input`, `breakpoint`, `help`, `dir`, `vars`,
  `memoryview`.
- Dunder attribute access (e.g. `df.__class__`, `getattr(df, '__dict__')`).
- Bare module references (e.g. `os`, `subprocess`, `sys`).

The execution namespace is `{__builtins__: <safe whitelist>, df, pd, np}`
and the final value must be assigned to a variable named `result`. A
`SIGALRM` kills runaway loops after `ADS_CHAT_CODE_TIMEOUT_S` seconds
(Unix only — on Windows the timeout is skipped).

### Frontend (`ChatPanel.tsx`)

- Side panel toggled from the dashboard header.
- Generous send UX: Enter to send, Shift+Enter for newline.
- Renders the assistant's `answer` text, a collapsible code block, the
  result table or scalar pill, and the auto-built chart.
- Errors surface inline (sandbox blocks, runtime errors, missing LLM
  key, LLM HTTP failures).

### Response shape

```json
{
  "job_id": "...",
  "session_id": "...",
  "answer": "Mean age is 32.1.",
  "code": "result = df['age'].mean()",
  "result": {
    "type": "dataframe",
    "value": null,
    "columns": ["age"],
    "records": [{ "age": 32.1 }]
  },
  "plotly_spec": { "data": [...], "layout": {...} },
  "error": null
}
```

---

## Phase 5 — Smart Model Suggestions (AutoML-lite)

One-click ML on the cleaned data. Auto-detects regression vs binary /
multiclass classification from a chosen target column, trains a small
fixed menu of sklearn models with a single deterministic 80/20 split,
and surfaces a side-by-side metric comparison with the winner marked.

### Endpoints

| Method | Path | Body | Purpose |
| --- | --- | --- | --- |
| `POST` | `/ml/suggest/{job_id}` | `{"target?": "col"}` | Returns the auto-detected target, problem type, and a candidate feature list. No training yet. |
| `POST` | `/ml/train/{job_id}` | `{"target": "col", "features": ["a", "b"]}` | Trains all 3 models. Returns the full report with metrics, a winner flag, and a Plotly comparison chart. Caches by `(job_id, target)`. |
| `GET`  | `/ml/result/{job_id}?target=col` | — | Returns the cached report. `404` if nothing cached. |

### Problem-type detection

- 2 unique non-null values → `binary_classification`
- 3–20 unique values, dtype in {boolean, categorical, integer, or
  float that looks like class labels} → `multiclass_classification`
- continuous numeric with high cardinality → `regression`
- text / datetime / single value / high-cardinality categorical →
  `unsupported`

### Models (fixed, no HPO)

| Problem | Models |
| --- | --- |
| regression | `LinearRegression`, `RandomForestRegressor(n_estimators=200, n_jobs=1)`, `HistGradientBoostingRegressor` |
| classification | `LogisticRegression(max_iter=1000)`, `RandomForestClassifier(n_estimators=200, n_jobs=1)`, `HistGradientBoostingClassifier` |

All seeded with `random_state=42` for reproducibility.

### Preprocessing

`ColumnTransformer` with:
- numeric features → `SimpleImputer(strategy="median")` + `StandardScaler`
- categorical / boolean features → `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
- datetime / text features are dropped

### Metrics

- Regression: **R²**, MAE, RMSE
- Classification: **accuracy**, F1 (macro), ROC AUC (binary only)

The "winner" is the model with the highest score on the headline
metric (R² for regression, accuracy for classification).

### Safeguards

- Rows with null target are dropped before training.
- Datasets over **50,000 rows** are silently downsampled to 50,000
  (`random_state=42`); the response carries `downsampled: true` and a
  `note`.
- Datasets with **< 20 non-null target rows** are rejected with a
  clear `error` in the response.
- Pipelines may not contain the target as a feature.
- Estimators that fail to fit are reported per-model with an `error`
  field; the others still run.

### Frontend (`MLPanel.tsx`)

- Side panel toggled from the dashboard header.
- Three states: **Configure** (target dropdown, feature checkboxes,
  problem-type pill, Run button) → **Training** (spinner) →
  **Results** (winner card, metrics table with best-per-metric
  highlighting, Plotly bar chart, "Change target or features" button
  to reconfigure).

---

## Testing

### Backend (pytest)

```bash
cd "Automate DS/backend"
PYTHONPATH=. python -m pytest -q
```

**263 tests, all green**, organized as:

| Test file | Count | Phase |
| --- | --- | --- |
| `test_pipeline.py`, `test_missing.py`, `test_normalize.py`, `test_type_detect.py`, `test_api.py` | 32 | 1 |
| `test_stats.py`, `test_correlations.py`, `test_trends.py`, `test_category_breakdown.py`, `test_sentences.py`, `test_insights_pipeline.py`, `test_insights_api.py` | 50 | 2 |
| `test_dashboards_serialization.py`, `test_dashboards_plotly_specs.py`, `test_dashboards_suggestions.py`, `test_dashboards_api.py` | 39 | 3 |
| `test_chat_schema.py`, `test_chat_sandbox.py`, `test_chat_executor.py`, `test_chat_session.py`, `test_chat_llm.py`, `test_chat_api.py` | 82 | 4 |
| `test_models_problem.py`, `test_models_preprocess.py`, `test_models_train.py`, `test_models_api.py` | 60 | 5 |

### Frontend (vitest)

```bash
cd "Automate DS/frontend"
npm test
```

**43 tests, all green**, organized as:

| Test file | Count |
| --- | --- |
| `src/lib/aggregations.test.ts` | 7 |
| `src/lib/plotly_specs.test.ts` | 5 |
| `src/components/TweakPanel.test.tsx` | 6 |
| `src/components/ChatPanel.test.tsx` | 13 |
| `src/components/MLPanel.test.tsx` | 12 |

### Production build

```bash
cd "Automate DS/frontend"
npm run build
```

Produces `dist/` with a minified bundle (~1.5 MB gzipped, mostly
Plotly).

---

## End-to-end verification (Phase 1 → 5)

```bash
# Terminal 1
cd "Automate DS/backend"
PYTHONPATH=. python -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Terminal 2
cd "Automate DS/frontend"
npm run dev
# Open http://127.0.0.1:5173

# 1. Upload a CSV
# 2. Inspect the cleaning report
# 3. Open the dashboard — review the auto-suggested charts
# 4. (Optional, with OPENAI_API_KEY set) Click "Chat with your data"
# 5. Click "Train models" — pick a target, see the 3-model comparison
```

For a backend-only smoke (no browser), a Phase 4 end-to-end script is
included at `backend/smoke_chat.py`. It patches the LLM call to return
canned responses and exercises the full chat flow against a real
uvicorn process.

---

## Out of scope (deliberately)

- **Persistence.** All state is in process memory (job store, ML
  result cache, chat session history). Restart wipes everything.
- **Auth / users / sharing.** Single-user, no accounts.
- **Filters / drill-downs on the dashboard.** Tweak panel only.
- **Streaming responses.** Both the chat and the ML endpoint return
  the full payload in one HTTP response.
- **Hyperparameter tuning / cross-validation.** Models use fixed
  params, single 80/20 split.
- **Feature engineering beyond impute + scale + one-hot.**
- **Neural nets, XGBoost, LightGBM, AutoGluon.** Sklearn core only.
- **ML inside the chat.** Chat stays analysis-only.
- **A UI to enter the OpenAI key.** It's a server-side env var.

Everything above is deferred to a future phase.
