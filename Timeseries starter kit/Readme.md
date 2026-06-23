# Timeseries Forecasting Starter Kit

A single-file Jupyter notebook (`timeseries-forecasting-starter-kit.ipynb`) that shows
how to use two timeseries foundation models (TFMs) — **Chronos-2** and **TimesFM 2.5** —
on the small datasets bundled in `./data/`. Each cell is a few lines of code with a few 
configurable variables at the top.

---

## Prerequisites

- **Python 3.12** is recommended. Python 3.13 currently runs into wheel
  availability issues with some of the model dependencies. (Python 3.11 also
  works.)
- An internet connection on first model load — the weights are downloaded from
  HuggingFace (~0.5 to ~4 GB depending on the model) into `~/.cache/huggingface/`.
- A CUDA-capable NVIDIA GPU is recommended. CPU works for short forecasts but
  expect tens of seconds per call.
- ~6 GB of free disk for the model caches if you try both models.

## 1. Create + activate a virtual environment

**Windows (PowerShell or CMD):**

```
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

**Linux / macOS:**

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install the always-needed libraries

```
pip install jupyter pandas numpy matplotlib holidays
```

## 3. Install PyTorch (with CUDA if you have a GPU)

Both model libraries bring in `torch` automatically as a transitive dep, but
installing it explicitly first gives you control over the CUDA variant. Pick the
one matching your NVIDIA driver:

```
# CUDA 12.1 (works with most modern drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.8 (newest drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only (no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify with:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Install the model libraries you want to try

You only need to install the model you plan to use. Install both if you want to
flip between them in the notebook.

### Chronos-2 (default in the notebook — multivariate, supports covariates)

```
pip install chronos-forecasting
```

### TimesFM 2.5 (univariate; no covariates)

```
pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
```

(The PyPI release of `timesfm` does not expose the `[torch]` extra; installing
from git is the supported path.)

## 5. Open the notebook

```
jupyter notebook timeseries-forecasting-starter-kit.ipynb
```

…or, in VS Code: open the file and pick your `.venv` as the kernel from the
top-right kernel selector.

## 6. Run the cells in order

The notebook has 5 sections, each with a single configuration code cell and an
optional output cell. Run them top-to-bottom the first time. After that, to
change anything (dataset, model, target, history length, horizon, covariates),
edit the variables at the top of the relevant cell and re-run **that cell and
every cell below it**.

---

## What lives where

```
.
├── timeseries-forecasting-starter-kit.ipynb    ← the notebook
├── Readme.md                          			← this file
└── data/                                       ← bundled CSVs
    ├── Total-load-on-Belgian-grid.csv          (15-min, default)
    ├── Belgian-offshore-wind-production.csv    (15-min)
    ├── DAM-prices-BELPEX.csv                   (hourly)
    ├── Oil-Temperature-ETTh1.csv               (hourly)    
    └── Seoul-Bike-Demand.csv                   (hourly)
```

## Bundled datasets

| Timeseries Dataset | Data range | Granularity | Units | Source |
| --- | --- | --- | --- | --- |
| Load on the Belgian grid, Elia forecasts | 2024-01-01 – 2025-12-31 | 15-minute | MW | Elia Open Data |
| Belgian offshore wind generation, Elia forecasts | 2024-01-01 – 2025-12-31 | 15-minute | MW | Elia Open Data |
| DAM prices | 2024-10-01 – 2025-10-01 | 1-hour | €/kWh | EPEX Spot |
| Oil temperature | 2016-07-01 – 2018-06-26 | 1-hour | C | Zhou, et. al, "Informer: beyond efficient transformer for long sequence time-series forecasting", Proc. AAAI Conf. Artif. Intell. 35 (12) (2021) |
| Seoul bike demand | 2017-12-01 – 2018-11-30 | 1-hour | N/A | https://archive.ics.uci.edu/ |


All bundled files use European date format (day-first), e.g. `31/12/2024 00:00`.
The notebook parses them with `pandas.to_datetime(..., dayfirst=True)`.

## Models

| Identifier in the notebook | HuggingFace repo | Univariate? | Covariates? |
|---|---|---|---|
| `"chronos-2"` (default) | `amazon/chronos-2` | No (multivariate) | Yes |
| `"timesfm-2.5"` | `google/timesfm-2.5-200m-pytorch` | Yes | No |

## Covariates (Chronos-2 only)

Three calendar features supported:

- `"Holidays (BE)"` — 1 on Belgian public holidays, 0 otherwise.
- `"Day of week"` — 0 (Monday) through 6 (Sunday).
- `"Hour of day"` — 0 through 23.

To enable / disable a covariate, edit the `COVARIATES = [...]` list in
section 3 of the notebook. Setting `COVARIATES = []` runs without any covariates.

The list is ignored when `MODEL == "timesfm-2.5"`.

---

## Common errors and how to fix them

| Error | Cause / fix |
|---|---|
| `ModuleNotFoundError: chronos` | Run `pip install chronos-forecasting`. |
| `ModuleNotFoundError: timesfm` | Install from git per section 4. |
| `AssertionError: TARGET '...' not in dataset columns` | Set `TARGET` to one of the column names printed when you load the dataset in section 1. |
| `AssertionError: FORECAST_START ... not in the dataset index` | Use a timestamp inside the data range, in `"YYYY-MM-DD HH:MM"` format. |
| Slow first run | First call to a model downloads weights (0.5–4 GB) into `~/.cache/huggingface/`. Subsequent runs reuse the cache. |
| `CUDA out of memory` | Lower `HISTORY`, switch to CPU, or pick a smaller model. |
| `TypeError: TimesFM_2p5_200M_torch.__init__() got an unexpected keyword argument 'proxies'` | A `huggingface_hub` / `timesfm` version mismatch. The notebook works around this by calling the internal `_from_pretrained` directly — if you see it, you're probably calling the public `from_pretrained` manually. |

---

## Going further

Once the notebook is running you have everything you need to integrate either
model into your own code:

- **For Chronos-2** — copy the `pipeline.predict_df(...)` snippet from section 4.
  The DataFrame-based API is the cleanest way to feed past + future covariates.
- **For TimesFM 2.5** — copy the `model.compile(...)` + `model.forecast(...)`
  snippet. Note the budget constraint: `max_context + max_horizon ≤ 16,384`.

Both models live behind clean, importable Python APIs. The notebook just shows
how to wire them up to a DataFrame and read the results.
