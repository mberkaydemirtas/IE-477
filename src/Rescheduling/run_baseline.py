#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_baseline.py

Creates a baseline schedule and saves it to a scenario-specific filename.

Naming rules:
1) If scenario JSON provides "baseline_output", use it exactly.
2) Else, try to derive a scenario id:
   - Prefer scn["scenario_id"] if present
   - Else parse from filename like "scenario_04_..." or "scenario04_..."
   - Else parse first standalone 1–2 digit number from scenario_name/file
3) If an id is found -> "baseline_solution_{id:02d}.json"
4) Else fallback -> "baseline_solution.json"

This fixes the issue where Scenario 04 was overwriting/using baseline_solution.json
while Scenario 03 used baseline_solution_03.json.
"""

import json
import os
import re
import sys
import subprocess
from datetime import datetime, timezone

from solver_core import make_base_data, solve_baseline


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

def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _scenario_name(scn: dict, scenario_path: str) -> str:
    return scn.get("scenario_name") or scn.get("name") or os.path.basename(scenario_path)

def _infer_scenario_id(scn: dict, scenario_path: str) -> int:
    """
    Best-effort inference of scenario id as an integer.
    Returns -1 if not found.
    """
    sid = scn.get("scenario_id", None)
    if sid is not None:
        try:
            return int(sid)
        except Exception:
            pass

    base = os.path.basename(scenario_path)
    # scenario_04_xxx.json or scenario04_xxx.json
    m = re.search(r"scenario[_\- ]*(\d+)", base, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # try scenario_name / filename any 1-2 digit token
    s = _scenario_name(scn, scenario_path)
    m2 = re.search(r"\b(\d{1,2})\b", s)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            pass

    return -1

def _pick_baseline_output_name(scn: dict, scenario_path: str) -> str:
    if scn.get("baseline_output"):
        return str(scn["baseline_output"]).strip()

    sid = _infer_scenario_id(scn, scenario_path)
    if sid >= 0:
        return f"baseline_solution_{sid:02d}.json"

    return "baseline_solution.json"


def main():
    if len(sys.argv) < 2:
        print("Usage: py run_baseline.py scenarios/<scenario>.json")
        sys.exit(1)

    scenario_path = sys.argv[1]
    if not os.path.exists(scenario_path):
        print(f"❌ Scenario file not found: {scenario_path}")
        sys.exit(1)

    scn = _load_json(scenario_path)

    workdir = os.path.dirname(os.path.abspath(__file__))
    scenario_name = _scenario_name(scn, scenario_path)

    # dynamic start time (baseline creation time)
    plan_start_iso = datetime.now(timezone.utc).isoformat()
    plan_calendar = scn.get("plan_calendar", {"utc_offset": "+03:00"})

    overrides = scn.get("base_overrides", {}) or {}
    base = make_base_data(overrides=overrides)


    # allow optional top-level data patch
    if isinstance(scn.get("data"), dict):
        data = _deep_merge(base, scn["data"])
    else:
        data = base

    baseline = solve_baseline(
        data,
        plan_start_iso=plan_start_iso,
        plan_calendar=plan_calendar,
        k1=float(scn.get("k1", 2.0)),
    )
    baseline["scenario_name"] = scenario_name
    baseline["scenario_file"] = os.path.basename(scenario_path)
    baseline["baseline_run_iso_utc"] = datetime.now(timezone.utc).isoformat()

    # output name support (fixed: auto scenario id naming)
    out_name = _pick_baseline_output_name(scn, scenario_path)
    out_path = os.path.join(workdir, out_name)

    _save_json_atomic(out_path, baseline)

    print(f"✅ Baseline saved: {out_path}")
    print("scenario:", scenario_name)
    print("plan_start_iso:", baseline["plan_start_iso"])
    print("objective:", baseline["objective"])

    # baseline plots (optional)
    plot_script = os.path.join(workdir, "plot_gantt_baseline.py")
    if os.path.exists(plot_script):
        print("📊 Generating BASELINE Gantt charts...")
        subprocess.run([sys.executable, plot_script, out_path], cwd=workdir, check=False)


if __name__ == "__main__":
    main()
