#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import subprocess
from datetime import datetime, timezone
from typing import Optional, Tuple

from solver_core import make_base_data, solve_baseline, solve_reschedule


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

def _res_block(scn: dict) -> dict:
    return scn.get("reschedule", {}) if isinstance(scn.get("reschedule"), dict) else {}

def _pick_mode(scn: dict) -> str:
    rb = _res_block(scn)
    return (rb.get("mode") or scn.get("mode") or "continue").strip().lower()

def _pick_disruptions(scn: dict) -> Tuple[list, list]:
    rb = _res_block(scn)
    um = rb.get("unavailable_machines", scn.get("unavailable_machines", [])) or []
    us = rb.get("unavailable_stations", scn.get("unavailable_stations", [])) or []
    return [int(x) for x in um], [int(x) for x in us]

def _resolve_rel_path(base_dir: str, p: str) -> Optional[str]:
    if not isinstance(p, str) or not p.strip():
        return None
    if os.path.isabs(p):
        return p if os.path.exists(p) else None
    p2 = os.path.normpath(os.path.join(base_dir, p))
    if os.path.exists(p2):
        return p2
    if os.path.basename(base_dir).lower() == "scenarios":
        p3 = os.path.normpath(os.path.join(os.path.dirname(base_dir), p))
        if os.path.exists(p3):
            return p3
    return None

def _pick_urgent_payload(scn: dict, base_dir: str) -> Optional[dict]:
    rb = _res_block(scn)
    if isinstance(scn.get("urgent_job"), dict):
        return {"urgent_job": scn["urgent_job"]}
    if isinstance(scn.get("urgent_payload"), dict):
        return scn["urgent_payload"]
    p = rb.get("urgent_payload_path") or scn.get("urgent_payload_path")
    if isinstance(p, str) and p.strip():
        p2 = _resolve_rel_path(base_dir, p)
        if p2:
            return _load_json(p2)
    return None


# -------------------------------
# Baseline filename logic
# -------------------------------

def _infer_scenario_id(scn: dict, scenario_path: str) -> int:
    sid = scn.get("scenario_id", None)
    if sid is not None:
        try:
            return int(sid)
        except Exception:
            pass

    base = os.path.basename(scenario_path)
    m = re.search(r"scenario[_\- ]*(\d+)", base, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

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

def _ensure_baseline(scn: dict, scenario_path: str, workdir: str) -> str:
    out_name = _pick_baseline_output_name(scn, scenario_path)
    out_path = os.path.join(workdir, out_name)

    if os.path.exists(out_path):
        return out_path

    print(f"⚠️ Baseline missing ({out_name}). Creating baseline...")

    scenario_name = _scenario_name(scn, scenario_path)
    plan_calendar = scn.get("plan_calendar", {"utc_offset": "+03:00"})
    plan_start_iso = datetime.now(timezone.utc).isoformat()

    # ✅ Baseline data: normal system only (NO overrides)
    base = make_base_data(overrides={})
    data = _deep_merge(base, scn["data"]) if isinstance(scn.get("data"), dict) else base

    baseline = solve_baseline(
        data,
        plan_start_iso=plan_start_iso,
        plan_calendar=plan_calendar,
        k1=float(scn.get("k1", 2.0)),
    )
    baseline["scenario_name"] = scenario_name
    baseline["scenario_file"] = os.path.basename(scenario_path)
    baseline["baseline_run_iso_utc"] = datetime.now(timezone.utc).isoformat()

    _save_json_atomic(out_path, baseline)
    print(f"✅ Baseline saved: {out_path}")
    return out_path


def _run_plotter(workdir: str, reschedule_path: str):
    script = os.path.join(workdir, "plot_gantt_reschedule.py")
    if not os.path.exists(script):
        print("⚠️ Plotter not found: plot_gantt_reschedule.py (skipping charts)")
        return
    print("📊 Generating Gantt charts...")
    subprocess.run([sys.executable, script, reschedule_path], cwd=workdir, check=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: py run_reschedule.py scenarios/<scenario>.json")
        sys.exit(1)

    scenario_path = sys.argv[1]
    if not os.path.exists(scenario_path):
        print(f"❌ Scenario file not found: {scenario_path}")
        sys.exit(1)

    scn = _load_json(scenario_path)
    workdir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.abspath(scenario_path))

    scenario_name = _scenario_name(scn, scenario_path)
    print("scenario:", scenario_name)

    baseline_path = _ensure_baseline(scn, scenario_path, workdir)
    baseline = _load_json(baseline_path)

    print("baseline_file:", os.path.basename(baseline_path))
    print("plan_start_iso:", baseline.get("plan_start_iso"))

    # ✅ RESCHEDULE DATA:
    # normal base data + optional scn["data"] + scenario overrides (ONLY here!)
    base = make_base_data(overrides={})
    data_base = _deep_merge(base, scn["data"]) if isinstance(scn.get("data"), dict) else base

    overrides = scn.get("overrides", {}) or {}
    if isinstance(overrides, dict) and overrides:
        data_base = _deep_merge(data_base, overrides)

    # debug: confirm overrides actually applied
    if isinstance(overrides, dict) and ("d_j" in overrides or "r_j" in overrides):
        print("APPLIED overrides:",
              "r_j[8]=", data_base.get("r_j", {}).get(8),
              "d_j[8]=", data_base.get("d_j", {}).get(8),
              "d_j[3]=", data_base.get("d_j", {}).get(3))

    unavailable_machines, unavailable_stations = _pick_disruptions(scn)
    mode = _pick_mode(scn)
    urgent_payload = _pick_urgent_payload(scn, base_dir)

    k1 = float(scn.get("k1", 2.0))

    rb = _res_block(scn)
    t_now_iso = rb.get("t_now_iso") or scn.get("t_now_iso") or rb.get("reschedule_time_iso") or scn.get("reschedule_time_iso")
    if t_now_iso:
        print("t_now_iso:", t_now_iso)

    res = solve_reschedule(
        data_base=data_base,
        old_solution=baseline,
        urgent_payload=urgent_payload,
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        mode=mode,
        k1=k1,
        t_now_iso=t_now_iso,
    )

    res["scenario_name"] = scenario_name
    res["scenario_file"] = os.path.basename(scenario_path)
    res["baseline_file"] = os.path.basename(baseline_path)
    res["reschedule_run_iso_utc"] = datetime.now(timezone.utc).isoformat()
    res["disruptions"] = {"unavailable_machines": unavailable_machines, "unavailable_stations": unavailable_stations}

    out_name = scn.get("reschedule_output") or "reschedule_solution.json"
    out_path = os.path.join(workdir, out_name)
    _save_json_atomic(out_path, res)

    print(f"✅ Reschedule saved: {out_path}")
    print("t0:", res.get("t0"))
    print("objective:", res.get("objective"))

    _run_plotter(workdir=workdir, reschedule_path=out_path)


if __name__ == "__main__":
    main()
