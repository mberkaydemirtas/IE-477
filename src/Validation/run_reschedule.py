#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import subprocess
from typing import Optional, Tuple
from datetime import datetime, timezone

from solver_core import solve_reschedule
from adapter import build_data_from_operations

DEFAULT_BASE_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "data", "base_data.json")
)

OUT_BASELINE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "outputs", "baseline"))
OUT_RESCHEDULE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "outputs", "reschedule"))

REFERENCE_BASELINE_SOLUTION = os.path.join(OUT_BASELINE_DIR, "base_data_baseline_solution.json")

WINDOW_HOURS = 250.0
OVERLAP_HOURS = 0.0


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
    if isinstance(scn.get("disruptions"), dict):
        um = scn["disruptions"].get("unavailable_machines", um) or um
        us = scn["disruptions"].get("unavailable_stations", us) or us
    return [int(x) for x in um], [int(x) for x in us]

def _resolve_rel_path(base_dir: str, p: str) -> Optional[str]:
    if not isinstance(p, str) or not p.strip():
        return None
    if os.path.isabs(p):
        return p if os.path.exists(p) else None
    p2 = os.path.normpath(os.path.join(base_dir, p))
    if os.path.exists(p2):
        return p2
    if os.path.basename(base_dir).lower() in ["scenarios", "rescheduling_scenarios"]:
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

def _infer_scenario_id(scn: dict, scenario_path: str) -> int:
    sid = scn.get("scenario_id", None)
    if sid is not None:
        try:
            return int(str(sid))
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

def _resolve_base_data_path(scn: dict, scenario_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(scenario_path))

    p = scn.get("base_data_path", None)
    if isinstance(p, str) and p.strip():
        if os.path.isabs(p) and os.path.exists(p):
            return p
        p2 = os.path.normpath(os.path.join(base_dir, p))
        if os.path.exists(p2):
            return p2

    if os.path.exists(DEFAULT_BASE_DATA_PATH):
        return DEFAULT_BASE_DATA_PATH

    raise FileNotFoundError(
        "Base data file not found. Provide scenario['base_data_path'] or update DEFAULT_BASE_DATA_PATH.\n"
        f"DEFAULT_BASE_DATA_PATH={DEFAULT_BASE_DATA_PATH}"
    )

def _maybe_load_system_config(base_data_path: str):
    base_dir = os.path.dirname(os.path.abspath(base_data_path))
    cand = os.path.join(base_dir, "system_config.json")
    if os.path.exists(cand):
        return _load_json(cand)

    cand2 = os.path.join(os.path.dirname(__file__), "data", "system_config.json")
    if os.path.exists(cand2):
        return _load_json(cand2)

    return None

def _run_plotter(script_name: str, json_path: str, outdir: str, sid_override: str, window_hours: float, overlap_hours: float):
    script = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script):
        print(f"Plotter not found: {script_name}")
        return
    subprocess.run(
        [sys.executable, script, json_path, outdir, sid_override, str(window_hours), str(overlap_hours)],
        cwd=os.path.dirname(__file__),
        check=False
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: py run_reschedule.py data/<scenario>.json")
        sys.exit(1)

    scenario_path = sys.argv[1]
    if not os.path.exists(scenario_path):
        print(f"Scenario file not found: {scenario_path}")
        sys.exit(1)

    scn = _load_json(scenario_path)
    base_dir = os.path.dirname(os.path.abspath(scenario_path))

    scenario_name = _scenario_name(scn, scenario_path)
    sid = _infer_scenario_id(scn, scenario_path)
    sid_txt = f"{sid:02d}" if sid >= 0 else "xx"

    print("scenario:", scenario_name)

    base_data_path = _resolve_base_data_path(scn, scenario_path)
    print("base_data:", base_data_path)

    system_config = _maybe_load_system_config(base_data_path)
    if system_config:
        print("system_config:", os.path.join(os.path.dirname(base_data_path), "system_config.json"))

    if not os.path.exists(REFERENCE_BASELINE_SOLUTION):
        raise FileNotFoundError(
            "Reference baseline solution not found.\n"
            f"Expected: {REFERENCE_BASELINE_SOLUTION}\n"
            "Run: py run_baseline.py data/base_data.json"
        )

    baseline = _load_json(REFERENCE_BASELINE_SOLUTION)
    baseline_plan_start_iso = baseline.get("plan_start_iso")
    print("plan_start_iso:", baseline_plan_start_iso)

    raw_base = _load_json(base_data_path)
    if isinstance(raw_base, list):
        data_base = {"operations": raw_base}
    else:
        data_base = raw_base

    if isinstance(scn.get("data"), dict):
        data_base = _deep_merge(data_base, scn["data"])

    overrides = scn.get("overrides", {}) or {}
    if isinstance(overrides, dict) and overrides:
        data_base = _deep_merge(data_base, overrides)

    unavailable_machines, unavailable_stations = _pick_disruptions(scn)
    mode = _pick_mode(scn)
    urgent_payload = _pick_urgent_payload(scn, base_dir)
    k1 = float(scn.get("k1", 2.0))

    rb = _res_block(scn)
    t_now_iso = rb.get("t_now_iso") or scn.get("t_now_iso") or rb.get("reschedule_time_iso") or scn.get("reschedule_time_iso")
    if t_now_iso:
        print("t_now_iso:", t_now_iso)

    # ADAPTER STEP (RESCHEDULE)
    plan_calendar = (
        data_base.get("plan_calendar")
        or data_base.get("calendar")
        or (system_config.get("calendar") if isinstance(system_config, dict) else None)
        or {"utc_offset": "+03:00"}
    )
    plan_start_iso_for_adapter = baseline_plan_start_iso or datetime.now(timezone.utc).isoformat()
    location_map = data_base.get("location_map", {}) or {}

    if "operations" in data_base and isinstance(data_base["operations"], list):
        data_base = build_data_from_operations(
            data_base["operations"],
            data_base,
            plan_start_iso=plan_start_iso_for_adapter,
            plan_calendar=plan_calendar,
            location_map=location_map,
            system_config=system_config,   # ✅ pass
        )

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
    res["baseline_file"] = os.path.basename(REFERENCE_BASELINE_SOLUTION)
    res["base_data_file"] = os.path.basename(base_data_path)
    res["reschedule_run_iso_utc"] = datetime.now(timezone.utc).isoformat()
    res["disruptions"] = {"unavailable_machines": unavailable_machines, "unavailable_stations": unavailable_stations}
    res["result_type"] = "reschedule"

    os.makedirs(OUT_RESCHEDULE_DIR, exist_ok=True)

    out_name = scn.get("reschedule_output") or f"scenario{sid_txt}_reschedule_solution.json"
    out_path = os.path.join(OUT_RESCHEDULE_DIR, out_name)
    _save_json_atomic(out_path, res)

    print(f"Reschedule saved: {out_path}")
    print("t0:", res.get("t0"))
    print("objective:", res.get("objective"))

    compare_dir = os.path.join(OUT_RESCHEDULE_DIR, f"scenario{sid_txt}_compare_gantts")
    os.makedirs(compare_dir, exist_ok=True)

    print("Generating comparison Gantt charts...")
    _run_plotter("plot_gantt_baseline.py", REFERENCE_BASELINE_SOLUTION, compare_dir, sid_txt, WINDOW_HOURS, OVERLAP_HOURS)
    _run_plotter("plot_gantt_reschedule.py", out_path, compare_dir, sid_txt, WINDOW_HOURS, OVERLAP_HOURS)


if __name__ == "__main__":
    main()
