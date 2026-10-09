# Energy Anomaly Detection: Model × Split Benchmark

A reproducible benchmark of four classical machine-learning models for detecting
anomalous hourly electricity meter readings, each trained and evaluated on five
independent train/test splits of the Kaggle
[Large-scale Energy Anomaly Detection (LEAD)](https://www.kaggle.com/competitions/energy-anomaly-detection)
dataset.

| Models | Splits (train / test) |
|---|---|
| Random Forest, Isolation Forest (unsupervised), Decision Tree, XGBoost | 30/70, 40/60, 50/50, 60/40, 80/20 |

4 models × 5 splits = **20 runs**. Every run is trained on its own train portion
and scored only on its held-out test portion. The pipeline produces comparison
tables (CSV, XLSX, HTML, Markdown), multi-panel figures (PNG + SVG) and a single
self-contained `outputs/REPORT.html`.

---

## Quick start (Windows)

1. Install **Python 3.10+** and tick *"Add python.exe to PATH"*.
2. Set up Kaggle access (see below).
3. Double-click **`run.bat`**, or from a terminal in the project folder:
   - Command Prompt: `run.bat`
   - PowerShell: `.\run.bat` (PowerShell does not run scripts from the current folder without `.\`)

`run.bat` creates a virtual environment in `.venv`, installs the requirements,
downloads the data, runs all seven stages and opens the report in your browser.
A quick trial on a 5% stratified sample:

```powershell
.\run.bat --sample 0.05
```

On other platforms:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.main --stage all
```

## Live progress monitor

While the pipeline runs, open a second terminal and start the monitor:

```powershell
.\monitor.bat
```

It opens http://127.0.0.1:8765 with a live view of the run: overall percent
done, elapsed and remaining time, the local time it will finish, the stage
timeline, every model x split chunk with its duration and F1, and an animated
3D scene showing how the model that is training right now works. The monitor
only reads `outputs/logs`, so it can be opened or closed at any time without
affecting the run. Estimates recalibrate from the timings measured on your
machine after every finished chunk.

## Kaggle credentials

1. Sign in to Kaggle, open the
   [competition rules](https://www.kaggle.com/competitions/energy-anomaly-detection/rules)
   and click **Join Competition / I Understand and Accept** (needed once).
2. Go to **Settings → API → Create New Token**.
3. Save the token in one of these places (checked in this order):
   - environment variable `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`)
   - `%USERPROFILE%\.kaggle\access_token` (file containing only the token, **no `.txt` extension**)
   - `%USERPROFILE%\.kaggle\kaggle.json` (legacy key file)
   - `config\kaggle.json` (git-ignored)

No credentials? Download the zip from the
[data page](https://www.kaggle.com/competitions/energy-anomaly-detection/data),
extract it into `data\raw\` and run again. The download step is skipped when
`data\raw\train.csv` exists.

## Running parts of the pipeline

| Stage | What it does | Output |
|---|---|---|
| `download` | fetch the Kaggle files if missing | `data/raw/*.csv` |
| `preprocess` | merge, clean, engineer features | `data/processed/dataset.parquet` |
| `split` | five independent stratified splits | `data/processed/splits/*.npz` |
| `train_eval` | 20 model × split runs | `outputs/{models,predictions,metrics,importances}` |
| `aggregate` | comparison tables | `outputs/tables/` |
| `visualize` | all figures | `outputs/figures/` |
| `report` | single-page HTML report | `outputs/REPORT.html` |

```bat
run_stage.bat preprocess split
run_stage.bat train_eval --models random_forest xgboost --splits split_30_70 split_80_20
run_stage.bat visualize report
run_stage.bat train_eval --force
```

Each stage checks its inputs and skips work that is already done; `--force`
redoes it. Everything adjustable (seed, split mode and ratios, hyperparameters,
feature flags, palette, DPI) lives in [`config/config.yaml`](config/config.yaml).

## Method in brief

- **Data**: `train.csv` (labels) left-joined with `train_features.csv` (building
  metadata, weather, calendar and ASHRAE-winner features) on
  `(building_id, timestamp)`. `test.csv` has no labels and is not used.
- **Features**: calendar and cyclical time features, `log1p(meter_reading)`,
  integer-coded categoricals with an explicit `missing` level. `building_id` and
  composite keys derived from it are excluded by default.
- **No leakage**: median imputation is fitted on each train portion only;
  XGBoost's `scale_pos_weight` and Isolation Forest's contamination come from the
  train labels only; the best-F1 threshold is tuned on train data and reported
  separately from the default-threshold headline results.
- **Isolation Forest** is unsupervised: labels set its contamination and are
  used for evaluation, never for fitting.
- **Imbalance**: anomalies are about 2% of readings, so accuracy is inflated.
  Read F1, PR-AUC, MCC, recall and precision instead.

## Project layout

```
config/config.yaml        all settings
src/main.py               CLI entry point
src/data/                 download, load, preprocess, features
src/splitting/            train/test split generation
src/models/               model wrappers + registry
src/evaluation/           metrics, aggregation, table export
src/visualization/        theme and every chart
src/pipeline/             stages, experiment runner, report
tests/test_smoke.py       synthetic end-to-end test (pytest)
monitor/                  live progress page (server + 3D view)
run.bat / run_stage.bat   Windows launchers
```

## Outputs

```
outputs/REPORT.html              single-page report with embedded figures
outputs/tables/                  master results, per-metric pivots, rankings, ...
outputs/figures/                 dashboard, heatmaps, curves, confusion matrices, ...
outputs/figures/interactive_dashboard.html
outputs/metrics/                 one JSON per run
outputs/predictions/             y_true, y_pred, y_score per run
outputs/models/                  fitted model + imputer per run (joblib)
outputs/run_info.json            library versions and settings
outputs/logs/                    one log file per invocation
```

## Tests

```bash
.venv/Scripts/python -m pytest
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `403 Forbidden` during download | Accept the competition rules once on Kaggle, then rerun. |
| `No Kaggle credentials found` | See *Kaggle credentials*. Check the token file has no `.txt` extension (Explorer hides it). |
| `'run.bat' is not recognized` in PowerShell | Use `.\run.bat` (same for `.\run_stage.bat`). |
| `Python was not found` | Install Python 3.10+ with *Add to PATH*, then reopen the terminal. |
| Out of memory | Run with `--sample 0.2` (or smaller), or close other programs. The full data needs roughly 4–6 GB of RAM. |
| Stale results after changing settings | Rerun the affected stages with `--force`. |

## Citation

If you use this benchmark, please cite it using [`CITATION.cff`](CITATION.cff)
and the LEAD dataset:

> Gulati, M. and Arjunan, P. (2022). *LEAD1.0: A Large-scale Annotated Dataset
> for Energy Anomaly Detection in Commercial Buildings.* ACM e-Energy '22.

## License

[MIT](LICENSE). The dataset is subject to the Kaggle competition rules.
