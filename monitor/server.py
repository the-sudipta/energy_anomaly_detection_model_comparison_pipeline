"""Live progress monitor for the benchmark pipeline.

Serves ``monitor/index.html`` and a JSON status endpoint built by reading the
newest pipeline log. It never touches the running pipeline: it only parses
``outputs/logs/pipeline_*.log`` and estimates the remaining time from the run
durations observed so far.

Usage::

    python -m monitor.server            # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import os
import re
import threading
import statistics
import sys
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from monitor import digests

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("index.html")
PORT = int(os.environ.get("MONITOR_PORT", "8765"))
STAGES = ("download", "preprocess", "split", "tune", "train_eval", "aggregate", "visualize", "report")
FULL_ROWS = 1_749_494
# Prior full-data seconds per run at a 30% train share, refined from observed runs.
PRIOR_SECONDS = {"random_forest": 180.0, "isolation_forest": 45.0, "decision_tree": 25.0, "xgboost": 35.0}
STAGE_PRIOR = {"download": 2, "preprocess": 90, "split": 30, "tune": 4200, "aggregate": 15, "visualize": 420, "report": 20}
SCALE_EXPONENT = 1.1

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \| (\w+)\s*\| ([\w.]+) \| (.*)$")
RUN_DONE = re.compile(r"\[(\w+) \| (\w+)\] F1=([\d.]+) PR-AUC=([\d.na]+) MCC=([-\d.na]+) fit=([\d.]+)s")
RUN_SKIP = re.compile(r"\[(\w+) \| (\w+)\] already done")
RUN_FAIL = re.compile(r"\[(\w+) \| (\w+)\] failed")
STARTED = re.compile(r"^Started: (\w+)$")
FINISHED = re.compile(r"^Finished: (\w+) in (.+)$")
MATRIX = re.compile(r"Feature matrix: ([\d,]+) x (\d+)")
TUNE_TRIAL = re.compile(r"\[tune (\w+) \| (\w+)\] trial (\d+)/(\d+) PR-AUC=([\d.]+) (.*)$")
SPLIT_INFO = re.compile(r"^(split_\w+): train=([\d,]+) .*test=([\d,]+)")


def load_plan() -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Read model order, split ratios and display names from the config.

    Returns:
        ``(models, split_ratios, display_names)``.
    """
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    names = config["plots"].get("display_names", {})
    return list(config["models"]), {k: float(v) for k, v in config["split"]["ratios"].items()}, names


def newest_log() -> Path | None:
    """Return the most recently modified pipeline log, if any.

    Returns:
        Path of the log or None.
    """
    logs = sorted((ROOT / "outputs" / "logs").glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_log(path: Path) -> dict[str, Any]:
    """Extract stage events, run results and dataset size from a log file.

    Args:
        path: Log file.

    Returns:
        Parsed events.
    """
    state: dict[str, Any] = {"stages": {}, "runs": {}, "events": [], "rows": None, "tuning": {},
                             "splits": {}, "first": None, "last": None, "done": False, "error": None}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.match(raw)
        if not match:
            continue
        stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        level, message = match.group(2), match.group(4)
        state["first"] = state["first"] or stamp
        state["last"] = stamp
        _apply(state, stamp, level, message)
    return state


def _apply(state: dict[str, Any], stamp: datetime, level: str, message: str) -> None:
    """Update the parsed state with one log record.

    Args:
        state: State being built.
        stamp: Record time.
        level: Log level.
        message: Log message.
    """
    if m := TUNE_TRIAL.search(message):
        state["tuning"].setdefault(m.group(2), []).append({
            "split": m.group(1), "trial": int(m.group(3)), "of": int(m.group(4)), "pr_auc": float(m.group(5)),
            "params": m.group(6), "time": stamp.isoformat()})
    elif m := STARTED.match(message):
        state["stages"][m.group(1)] = {"start": stamp, "end": None}
        state["events"].append((stamp, f"Started stage {m.group(1)}"))
    elif m := FINISHED.match(message):
        state["stages"].setdefault(m.group(1), {"start": stamp})["end"] = stamp
    elif m := RUN_DONE.search(message):
        state["runs"][(m.group(1), m.group(2))] = {
            "end": stamp, "f1": float(m.group(3)), "pr_auc": _num(m.group(4)),
            "mcc": _num(m.group(5)), "fit": float(m.group(6)), "status": "done"}
        state["events"].append((stamp, f"Finished {m.group(2)} on {m.group(1)} (F1 {m.group(3)})"))
    elif m := RUN_SKIP.search(message):
        state["runs"][(m.group(1), m.group(2))] = {"end": stamp, "status": "skipped"}
    elif m := RUN_FAIL.search(message):
        state["runs"][(m.group(1), m.group(2))] = {"end": stamp, "status": "failed"}
    elif m := MATRIX.search(message):
        state["rows"] = int(m.group(1).replace(",", ""))
    elif m := SPLIT_INFO.match(message):
        state["splits"][m.group(1)] = int(m.group(2).replace(",", ""))
    elif message.startswith("Done:"):
        state["done"] = True
    if level == "ERROR" and not state["done"]:
        state["error"] = message.splitlines()[0][:240]


def _num(text: str) -> float | None:
    """Parse a float, returning None for ``nan``."""
    try:
        value = float(text)
    except ValueError:
        return None
    return None if value != value else value


def build_status() -> dict[str, Any]:
    """Assemble the full status payload for the page.

    Returns:
        JSON-serialisable status.
    """
    models, ratios, names = load_plan()
    log = newest_log()
    now = datetime.now()
    if log is None:
        return {"state": "idle", "now": now.isoformat(), "message": "No pipeline log found yet."}
    parsed = parse_log(log)
    runs = _run_table(parsed, models, ratios, now)
    stages = _stage_table(parsed, runs, now)
    return _summary(parsed, runs, stages, models, ratios, names, now, log)


def _run_table(parsed: dict[str, Any], models: list[str], ratios: dict[str, float],
               now: datetime) -> list[dict[str, Any]]:
    """Build the 20-run table with durations, status and estimates.

    Args:
        parsed: Parsed log.
        models: Model order.
        ratios: Split ratios in order.
        now: Current time.

    Returns:
        One record per run in execution order.
    """
    rows = parsed["rows"] or FULL_ROWS
    train_start = parsed["stages"].get("train_eval", {}).get("start")
    order = [(s, m) for s in ratios for m in models]
    cursor = train_start
    table = []
    for split, model in order:
        info = dict(parsed["runs"].get((split, model), {}))
        n_train = parsed["splits"].get(split) or int(rows * ratios[split])
        record = {"split": split, "model": model, "n_train": n_train, "status": info.get("status", "pending"),
                  "f1": info.get("f1"), "pr_auc": info.get("pr_auc"), "mcc": info.get("mcc"),
                  "fit": info.get("fit"), "start": None, "end": None, "seconds": None}
        if info.get("end") and cursor:
            record.update(start=cursor, end=info["end"], seconds=(info["end"] - cursor).total_seconds())
            cursor = info["end"]
        table.append(record)
    _mark_running(table, parsed, cursor, now)
    _estimate(table, now)
    return table


def _mark_running(table: list[dict[str, Any]], parsed: dict[str, Any], cursor: datetime | None,
                  now: datetime) -> None:
    """Flag the first pending run as running while train_eval is active.

    Args:
        table: Run table.
        parsed: Parsed log.
        cursor: End time of the last finished run (or train_eval start).
        now: Current time.
    """
    stage = parsed["stages"].get("train_eval")
    if not stage or stage.get("end") or parsed["error"] or cursor is None:
        return
    for record in table:
        if record["status"] == "pending":
            record.update(status="running", start=cursor, seconds=(now - cursor).total_seconds())
            return


def _estimate(table: list[dict[str, Any]], now: datetime) -> None:
    """Estimate the duration of every unfinished run, calibrated on finished runs.

    Each model gets a scale factor = observed / prior from its own finished runs
    (or the median over all models before it has finished one).

    Args:
        table: Run table, updated in place with ``estimate`` and ``remaining``.
        now: Current time.
    """
    def prior(record: dict[str, Any]) -> float:
        base_rows = 0.3 * FULL_ROWS
        return PRIOR_SECONDS[record["model"]] * (record["n_train"] / base_rows) ** SCALE_EXPONENT

    factors: dict[str, list[float]] = {}
    for record in table:
        if record["status"] == "done" and record["seconds"]:
            factors.setdefault(record["model"], []).append(record["seconds"] / prior(record))
    overall = statistics.median([f for fs in factors.values() for f in fs]) if factors else 1.0
    for record in table:
        if record["status"] == "done":
            record["estimate"], record["remaining"] = record["seconds"], 0.0
            continue
        scale = statistics.median(factors[record["model"]]) if record["model"] in factors else overall
        estimate = prior(record) * scale
        if record["status"] == "running":
            estimate = max(estimate, record["seconds"] * 1.05)
            record["remaining"] = max(estimate - record["seconds"], 0.0)
        else:
            record["remaining"] = estimate if record["status"] == "pending" else 0.0
        record["estimate"] = estimate


def _stage_table(parsed: dict[str, Any], runs: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Describe every stage with elapsed and remaining time.

    Args:
        parsed: Parsed log.
        runs: Run table with estimates.
        now: Current time.

    Returns:
        One record per stage in pipeline order.
    """
    rows_scale = ((parsed["rows"] or FULL_ROWS) / FULL_ROWS) ** 0.6
    table = []
    for name in STAGES:
        info = parsed["stages"].get(name)
        if name == "train_eval":
            estimate = sum(r["estimate"] for r in runs)
        elif name == "tune" and parsed["tuning"]:
            estimate = _tune_total(parsed, now)
        else:
            estimate = max(STAGE_PRIOR[name] * rows_scale, 3)
        record = {"name": name, "status": "pending", "elapsed": 0.0, "estimate": estimate, "remaining": estimate}
        if info and info.get("end"):
            spent = (info["end"] - info["start"]).total_seconds()
            record.update(status="done", elapsed=spent, estimate=spent, remaining=0.0)
        elif info:
            spent = (now - info["start"]).total_seconds()
            remaining = sum(r["remaining"] for r in runs) if name == "train_eval" else max(estimate - spent, 5)
            if name == "tune" and parsed["tuning"]:
                remaining = max(_tune_total(parsed, now) - spent, 5)
            record.update(status="error" if parsed["error"] else "running", elapsed=spent,
                          remaining=remaining, estimate=spent + remaining)
        table.append(record)
    return table


def _tune_total(parsed: dict[str, Any], now: datetime) -> float:
    """Estimate the whole tune stage from the trials already finished.

    Every (split, model) runs a fixed number of candidates (the ``of`` in
    "trial i/of"). Each model's observed seconds per trial is applied to its
    trials still to come, so heavy models (forests, boosting) weigh more.

    Args:
        parsed: Parsed log with ``tuning`` trials.
        now: Current time.

    Returns:
        Estimated total seconds for the tune stage.
    """
    start = parsed["stages"]["tune"]["start"]
    n_splits = len(load_plan()[1])
    remaining = 0.0
    for ts in parsed["tuning"].values():
        times = sorted(datetime.fromisoformat(t["time"]) for t in ts)
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if (b - a).total_seconds() < 900]
        rate = statistics.median(gaps) if gaps else 30.0
        per_split = ts[0]["of"]
        remaining += rate * max(per_split * n_splits - len(ts), 0)
    unseen = [m for m in PRIOR_SECONDS if m not in parsed["tuning"]]
    remaining += len(unseen) * 11 * n_splits * 40.0
    return (now - start).total_seconds() + remaining


def _summary(parsed: dict[str, Any], runs: list[dict[str, Any]], stages: list[dict[str, Any]],
             models: list[str], ratios: dict[str, float], names: dict[str, str], now: datetime,
             log: Path) -> dict[str, Any]:
    """Combine everything into the payload sent to the page.

    Args:
        parsed: Parsed log.
        runs: Run table.
        stages: Stage table.
        models: Model order.
        ratios: Split ratios.
        names: Display names.
        now: Current time.
        log: Log file being followed.

    Returns:
        The status payload.
    """
    elapsed = (now - parsed["first"]).total_seconds() if parsed["first"] else 0.0
    remaining = 0.0 if parsed["done"] else sum(s["remaining"] for s in stages)
    total = elapsed + remaining
    current = next((r for r in runs if r["status"] == "running"), None)
    active = next((s for s in stages if s["status"] in ("running", "error")), None)
    state = "done" if parsed["done"] else "error" if parsed["error"] else "running"
    if state == "running" and (now - parsed["last"]).total_seconds() > 1800 and not current:
        state = "stalled"
    return {
        "state": state, "now": now.isoformat(), "log": log.name, "error": parsed["error"],
        "started": parsed["first"].isoformat() if parsed["first"] else None,
        "elapsed": elapsed, "remaining": remaining, "total": total,
        "percent": 100.0 if parsed["done"] else (100 * elapsed / total if total else 0.0),
        "eta": (now + timedelta(seconds=remaining)).isoformat(),
        "rows": parsed["rows"], "models": models, "splits": list(ratios),
        "split_pct": {k: round(100 * v) for k, v in ratios.items()}, "names": names,
        "runs_done": sum(r["status"] in ("done", "skipped") for r in runs), "runs_total": len(runs),
        "runs_cached": sum(r["status"] == "skipped" for r in runs),
        "current_stage": active["name"] if active else None,
        "current_run": _jsonable(current) if current else None,
        "activity": _activity(active, current, names, state, sum(r["status"] == "skipped" for r in runs)),
        "stages": stages, "runs": [_jsonable(r) for r in runs], "tuning": parsed["tuning"],
        "events": [{"time": t.isoformat(), "text": text} for t, text in parsed["events"][-12:]][::-1],
    }


def _activity(stage: dict[str, Any] | None, run: dict[str, Any] | None, names: dict[str, str], state: str,
              cached: int = 0) -> str:
    """Describe in one sentence what the pipeline is doing right now.

    Args:
        stage: Active stage record.
        run: Active run record.
        names: Display names.
        state: Overall state.
        cached: Number of runs reused from an earlier invocation.

    Returns:
        A human sentence.
    """
    if state == "done" and cached:
        return (f"Finished in seconds: {cached} model runs were reused from an earlier run, so nothing needed "
                "training. Run .\\run.bat --force to retrain everything from scratch.")
    if state == "done":
        return "All stages finished. The report is ready in outputs/REPORT.html."
    if state == "error":
        return "The pipeline stopped with an error. Check the newest log in outputs/logs."
    if run:
        split = run["split"].replace("split_", "").replace("_", "/")
        return (f"Training {names.get(run['model'], run['model'])} on {run['n_train']:,} rows "
                f"({split} split), then scoring the test rows and tuning the threshold on train rows.")
    descriptions = {
        "download": "Checking for the Kaggle files in data/raw.",
        "preprocess": "Merging train.csv with train_features.csv and engineering features.",
        "split": "Drawing the five stratified train/test splits.",
        "tune": "Searching hyperparameters with 3-fold cross-validation inside each split's train portion.",
        "train_eval": "Preparing the next model run.",
        "aggregate": "Building the comparison tables (CSV, XLSX, HTML, Markdown).",
        "visualize": "Rendering heatmaps, curves, confusion matrices and the dashboard.",
        "report": "Writing the single-page HTML report.",
    }
    return descriptions.get(stage["name"], "Working.") if stage else "Waiting for the pipeline to start."


def _jsonable(record: dict[str, Any]) -> dict[str, Any]:
    """Convert datetimes in a record to ISO strings.

    Args:
        record: Any dict.

    Returns:
        A JSON-serialisable copy.
    """
    return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in record.items()}


_digest_lock = threading.Lock()
_digest_state: dict[str, str] = {}


def digest_for(model: str) -> dict[str, Any]:
    """Return a trace for the model, falling back to an earlier run while one is not ready.

    Args:
        model: Model name.

    Returns:
        A trace, or ``{"pending": reason}`` if none exists at all.
    """
    result = _current_digest(model)
    if "pending" in result:
        fallback = digests.latest_cached(model)
        if fallback:
            fallback["waiting_for"] = result["pending"]
            return fallback
    return result


def _current_digest(model: str) -> dict[str, Any]:
    """Return the real-model digest of the latest finished run of a model.

    Builds it in a background thread the first time; meanwhile the page gets a
    short ``pending`` explanation.

    Args:
        model: Model name.

    Returns:
        The digest, or ``{"pending": reason}``.
    """
    status = build_status()
    if status.get("state") == "idle":
        return {"pending": "No finished run yet."}
    done = [r for r in status["runs"] if r["model"] == model and r["status"] == "done"]
    if not done:
        return {"pending": "This model has not finished a run yet in the current pipeline."}
    run = done[-1]
    run_id = f"{run['split']}__{model}"
    signature = digests.run_signature(run_id)
    cached = digests.load_cached(run_id, signature)
    if cached:
        return cached
    training = status.get("current_stage") == "train_eval"
    allowed, reason = digests.can_build(ROOT / "outputs" / "models" / f"{run_id}.joblib", training)
    if not allowed:
        return {"pending": reason, "run": run_id}
    state = _digest_state.get(run_id, "")
    if state.startswith("failed"):
        _digest_state.pop(run_id, None)
        return {"pending": f"Could not read {run_id} ({state[8:][:160]}); retrying.", "run": run_id}
    with _digest_lock:
        if _digest_state.get(run_id) != "building":
            _digest_state[run_id] = "building"
            threading.Thread(target=_build, args=(run, model, run_id, signature), daemon=True).start()
    return {"pending": f"Reading the fitted model of {run['split']} ...", "run": run_id}


def _build(run: dict[str, Any], model: str, run_id: str, signature: str | None) -> None:
    """Build one digest in the background, recording failures for the page.

    Args:
        run: Run record.
        model: Model name.
        run_id: Run identifier.
        signature: Split signature.
    """
    try:
        digests.build_digest(run["split"], model, signature)
        _digest_state[run_id] = "done"
    except Exception as error:  # surfaced to the page, the server keeps running
        _digest_state[run_id] = f"failed: {error}"


class Handler(BaseHTTPRequestHandler):
    """Serves the page and the status endpoint."""

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        """Handle GET requests for ``/`` and ``/api/status``."""
        if self.path.startswith("/api/status"):
            try:
                body = json.dumps(build_status()).encode("utf-8")
            except Exception as error:  # report parse problems to the page instead of crashing
                body = json.dumps({"state": "error", "error": f"Monitor error: {error}"}).encode("utf-8")
            self._send(body, "application/json")
        elif self.path.startswith("/api/digest"):
            model = self.path.split("model=")[-1].split("&")[0]
            try:
                payload = digest_for(model) if model in PRIOR_SECONDS else {"pending": "Unknown model."}
            except Exception as error:  # keep the page alive on unexpected data
                payload = {"pending": f"Could not read the model: {error}"}
            self._send(json.dumps(payload).encode("utf-8"), "application/json")
        elif self.path.split("?")[0] in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def _send(self, body: bytes, content_type: str) -> None:
        """Write a 200 response.

        Args:
            body: Response bytes.
            content_type: MIME type.
        """
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        # Allow the page to be opened straight from disk (file://) and still poll the server.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: Any) -> None:
        """Silence per-request logging."""


def main() -> None:
    """Start the server on localhost and open the browser."""
    url = f"http://127.0.0.1:{PORT}"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"A monitor is already running at {url}; opening it.")
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
        return
    print(f"Live monitor running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
