#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_baseline.py
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timezone

from solver_core import solve_baseline
from adapter import build_data_from_operations


OUT_BASELINE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "outputs", "baseline")
)

# ---- Gantt split settings (hours) ----
WINDOW_HOURS = 200.0
OVERLAP_HOURS = 0.0


def _load_json(path: str) -> dict:
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


def main():
    if len(sys.argv) < 2:
        print("Usage: py run_baseline.py data/base_data.json")
        sys.exit(1)

    base_data_path = sys.argv[1]
    if not os.path.exists(base_data_path):
        print(f"❌ Base data file not found: {base_data_path}")
        sys.exit(1)

    os.makedirs(OUT_BASELINE_DIR, exist_ok=True)

    # load raw json
    base_data = _load_json(base_data_path)

    # plan start & calendar
    plan_start_iso = datetime.now(timezone.utc).isoformat()
    plan_calendar = base_data.get("plan_calendar", {"utc_offset": "+03:00"})

    # location map (optional)
    location_map = base_data.get("location_map", {}) or {}

    # ADAPTER STEP
    # If incoming data is in new format (operations list), convert to solver-core format.
    if "operations" in base_data and isinstance(base_data["operations"], list):
        base_data = build_data_from_operations(
            base_data["operations"],
            base_data,
            plan_start_iso=plan_start_iso,
            plan_calendar=plan_calendar,
            location_map=location_map
        )

    baseline = solve_baseline(
        base_data,
        plan_start_iso=plan_start_iso,
        plan_calendar=plan_calendar,
        k1=float(base_data.get("k1", 2.0))
    )

    baseline["base_data_file"] = os.path.basename(base_data_path)
    baseline["baseline_run_iso_utc"] = datetime.now(timezone.utc).isoformat()
    baseline["result_type"] = "baseline"

    out_json = os.path.join(OUT_BASELINE_DIR, "base_data_baseline_solution.json")
    _save_json_atomic(out_json, baseline)

    print(f"✅ Baseline saved: {out_json}")
    print("plan_start_iso:", baseline["plan_start_iso"])
    print("objective:", baseline["objective"])

    # charts
    gantt_dir = os.path.join(OUT_BASELINE_DIR, "base_data_gantts")
    os.makedirs(gantt_dir, exist_ok=True)

    plotter = os.path.join(os.path.dirname(__file__), "plot_gantt_baseline.py")
    if os.path.exists(plotter):
        print("📊 Generating BASELINE Gantt charts...")

        # ✅ Don't pass empty sid_override ("") — omit it entirely.
        cmd = [
            sys.executable,
            plotter,
            out_json,
            gantt_dir,
            str(WINDOW_HOURS),
            str(OVERLAP_HOURS),
        ]

        print("CMD =", cmd)  # helpful debug
        subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            check=True
        )


if __name__ == "__main__":
    main()
