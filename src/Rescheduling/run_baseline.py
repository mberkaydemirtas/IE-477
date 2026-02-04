#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import shutil
import subprocess
from datetime import datetime, timezone

from solver_core import make_base_data, solve_baseline


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _scenario_tag(scn: dict, scenario_path: str) -> str:
    sid = scn.get("scenario_id", None)
    if sid is not None:
        s = str(sid).strip()
        return f"s{s.zfill(2)}" if s.isdigit() else f"s_{s}"

    base = os.path.basename(scenario_path).replace(".json", "")
    parts = base.split("_")
    for p in parts:
        if p.isdigit():
            return f"s{p.zfill(2)}"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"s_{ts}"


def _run_baseline_plotter_and_archive(tag: str, workdir: str):
    plot_script = os.path.join(workdir, "plot_gantt_baseline.py")
    if not os.path.exists(plot_script):
        print(f"⚠️ plot_gantt_baseline.py not found at: {plot_script}")
        return

    print("📊 Generating BASELINE Gantt charts...")
    subprocess.run([sys.executable, plot_script], cwd=workdir, check=True)

    outputs = [
        "gantt_machine_baseline.png",
        "gantt_station_baseline.png",
    ]

    for fn in outputs:
        src = os.path.join(workdir, fn)
        if os.path.exists(src):
            name, ext = os.path.splitext(fn)
            dst = os.path.join(workdir, f"{name}_{tag}{ext}")
            shutil.copyfile(src, dst)
            print(f"✅ Saved: {dst}")


def main():
    if len(sys.argv) < 2:
        print("Usage: py run_baseline.py scenarios/<scenario>.json")
        sys.exit(1)

    scenario_path = sys.argv[1]
    scn = _load_json(scenario_path)
    workdir = os.path.dirname(os.path.abspath(__file__))

    scenario_name = scn.get("scenario_name") or scn.get("name") or os.path.basename(scenario_path)
    plan_start_iso = scn.get("plan_start_iso", "2025-12-18T05:00:00+00:00")
    plan_calendar = scn.get("plan_calendar", {"utc_offset": "+03:00"})

    # ✅ 1) Eğer scenario "data" içeriyorsa direkt onu kullan
    if "data" in scn and isinstance(scn["data"], dict):
        data = scn["data"]
        print("ℹ️ Scenario provides top-level 'data' — using it directly.")
    else:
        # ✅ 2) Yoksa overrides ile base data üret
        overrides = scn.get("overrides", {}) or {}
        data = make_base_data(overrides=overrides)
        print("ℹ️ Scenario has no top-level 'data' — generated data via make_base_data(overrides).")

    baseline = solve_baseline(data, plan_start_iso=plan_start_iso)
    baseline["plan_calendar"] = plan_calendar
    baseline["scenario_name"] = scenario_name
    baseline["baseline_created_at_utc"] = datetime.now(timezone.utc).isoformat()

    baseline_path = os.path.join(workdir, "baseline_solution.json")
    _save_json(baseline_path, baseline)

    print(f"✅ Baseline saved: {baseline_path}")
    print("scenario:", scenario_name)
    print("plan_start_iso (UTC):", baseline.get("plan_start_iso"))
    print("objective:", baseline.get("objective"))

    tag = _scenario_tag(scn, scenario_path)
    _run_baseline_plotter_and_archive(tag=tag, workdir=workdir)


if __name__ == "__main__":
    main()
