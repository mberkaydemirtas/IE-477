#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from adapter import build_data_from_operations
from solver_core import solve_baseline

OUT_BASELINE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "outputs", "baseline")
)

WINDOW_HOURS = 250.0
OVERLAP_HOURS = 0.0
DEFAULT_REFERENCE_NOW_ISO = "2026-01-30T00:00:00+00:00"


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_atomic(path: str, obj: dict):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _print_late_jobs(baseline: dict):
    rows = baseline.get("job_delays", []) if isinstance(baseline, dict) else []
    if not isinstance(rows, list):
        rows = []

    late = []
    for r in rows:
        try:
            delay = float(r.get("delay_hours", 0.0))
            if delay > 1e-9:
                late.append((int(r.get("job_id")), delay))
        except Exception:
            continue

    late.sort(key=lambda x: x[1], reverse=True)

    print(f"late_jobs: {len(late)}")
    for job_id, delay in late:
        print(f"job_id={job_id}, delay_hours={delay:.3f}")


def _maybe_load_system_config(base_data_path: str, system_config_path: str = None):
    if system_config_path:
        if os.path.exists(system_config_path):
            return _load_json(system_config_path)
        print(f"WARN system config file not found: {system_config_path}")

    base_dir = os.path.dirname(os.path.abspath(base_data_path))
    cand = os.path.join(base_dir, "system_config.json")
    if os.path.exists(cand):
        return _load_json(cand)

    cand2 = os.path.join(os.path.dirname(__file__), "data", "system_config.json")
    if os.path.exists(cand2):
        return _load_json(cand2)

    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: py run_baseline.py data/base_data.json [data/system_config.json]")
        sys.exit(1)

    base_data_path = sys.argv[1]
    system_config_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(base_data_path):
        print(f"ERROR Base data file not found: {base_data_path}")
        sys.exit(1)

    os.makedirs(OUT_BASELINE_DIR, exist_ok=True)
    raw = _load_json(base_data_path)

    # if base_data payload is a list => treat it as operations
    if isinstance(raw, list):
        base_data = {"operations": raw}
    else:
        base_data = raw

    # Accept common API payload shapes where operations are not under `operations`.
    if isinstance(base_data, dict) and (
        "operations" not in base_data or not isinstance(base_data.get("operations"), list)
    ):
        for alt_key in ("assignments", "items", "data"):
            alt_val = base_data.get(alt_key)
            if isinstance(alt_val, list):
                base_data = {**base_data, "operations": alt_val}
                break

    system_config = _maybe_load_system_config(base_data_path, system_config_path)

    # Historical dataset: anchor "now" to Jan 30, 2026 unless caller overrides.
    plan_start_iso = (
        base_data.get("reference_now_iso")
        or DEFAULT_REFERENCE_NOW_ISO
    )

    plan_calendar = (
        base_data.get("plan_calendar")
        or base_data.get("calendar")
        or (system_config.get("calendar") if isinstance(system_config, dict) else None)
        or {"utc_offset": "+03:00"}
    )

    location_map = base_data.get("location_map", {}) or {}

    # ADAPTER
    if "operations" in base_data and isinstance(base_data["operations"], list):
        base_data = build_data_from_operations(
            base_data["operations"],
            base_data,
            plan_start_iso=plan_start_iso,
            plan_calendar=plan_calendar,
            location_map=location_map,
            system_config=system_config,
        )

    baseline = solve_baseline(
        base_data,
        plan_start_iso=plan_start_iso,
        plan_calendar=plan_calendar,
        k1=float(base_data.get("k1", 2.0)),
    )

    baseline["base_data_file"] = os.path.basename(base_data_path)
    baseline["baseline_run_iso_utc"] = datetime.now(timezone.utc).isoformat()
    baseline["result_type"] = "baseline"

    out_json = os.path.join(OUT_BASELINE_DIR, "base_data_baseline_solution.json")
    _save_json_atomic(out_json, baseline)

    print(f"Baseline saved: {out_json}")
    print("plan_start_iso:", baseline.get("plan_start_iso"))
    print("objective:", baseline.get("objective"))
    _print_late_jobs(baseline)

    gantt_dir = os.path.join(OUT_BASELINE_DIR, "base_data_gantts")
    os.makedirs(gantt_dir, exist_ok=True)

    plotter = os.path.join(os.path.dirname(__file__), "plot_gantt_baseline.py")
    if os.path.exists(plotter):
        print("Generating BASELINE Gantt charts...")
        subprocess.run(
            [
                sys.executable,
                plotter,
                out_json,
                gantt_dir,
                "",
                str(WINDOW_HOURS),
                str(OVERLAP_HOURS),
            ],
            cwd=os.path.dirname(__file__),
            check=False,
        )


if __name__ == "__main__":
    main()
