#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import shutil
import subprocess
from datetime import datetime, timezone

from HeuristicBaseModel import run_heuristic  # ✅ baseline buradan


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _tuple_key_str(i: int, k: int) -> str:
    return f"{int(i)},{int(k)}"

def _baseline_to_solution_json(plan_start_iso: str, plan_calendar: dict, scenario_name: str, data: dict, res) -> dict:
    S_old = {int(i): float(res.S[int(i)]) for i in data["I"]}
    C_old = {int(i): float(res.C[int(i)]) for i in data["I"]}

    x_old = {}
    y_old = {}
    for i in data["I"]:
        i = int(i)
        m = int(res.assign_machine[i])
        l = int(res.assign_station[i])
        x_old[_tuple_key_str(i, m)] = 1
        y_old[_tuple_key_str(i, l)] = 1

    schedule = []
    job_of = {}
    for j, ops in data["O_j"].items():
        for op in ops:
            job_of[int(op)] = int(j)

    for i in data["I"]:
        i = int(i)
        schedule.append({
            "op_id": i,
            "op_label": str(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(res.S[i]),
            "finish": float(res.C[i]),
            "machine": int(res.assign_machine[i]),
            "station": int(res.assign_station[i]),
        })

    return {
        "plan_start_iso": plan_start_iso,
        "plan_calendar": plan_calendar,
        "scenario_name": scenario_name,
        "baseline_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {"T_max": float(res.T_max), "C_max": float(res.C_max)},
        "schedule": schedule,
        "S_old": S_old,
        "C_old": C_old,
        "x_old": x_old,
        "y_old": y_old,
    }

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

    # ✅ Bu baseline runner "data" ister (sende reschedule runner da bu şekildeydi)
    if "data" not in scn or not isinstance(scn["data"], dict):
        raise ValueError("Scenario JSON must include a top-level 'data' dict.")

    data = scn["data"]

    base_res = run_heuristic(data, k1=float(scn.get("k1", 2.0)))
    baseline = _baseline_to_solution_json(plan_start_iso, plan_calendar, scenario_name, data, base_res)

    baseline_path = os.path.join(workdir, "baseline_solution.json")
    _save_json(baseline_path, baseline)

    print(f"✅ Baseline saved: {baseline_path}")
    print("scenario:", scenario_name)
    print("plan_start_iso (UTC):", baseline["plan_start_iso"])
    print("objective:", baseline["objective"])

    tag = _scenario_tag(scn, scenario_path)
    _run_baseline_plotter_and_archive(tag=tag, workdir=workdir)

if __name__ == "__main__":
    main()
