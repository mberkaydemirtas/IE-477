#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_reschedule.py

BASELINE:
  - HeuristicBaseModel.run_heuristic(data) -> baseline_solution.json (senin eski formatında)

RESCHEDULE:
  - solver_core.solve_reschedule(...) -> reschedule_solution.json

SCENARIO COMPAT:
  - supports:
      plan_start_iso, plan_calendar, scenario_name
      baseline_overrides, overrides
      disruptions { unavailable_machines, unavailable_stations }
      urgent_job / urgent_payload / urgent_payload_path
      reschedule { ... same keys ... }
  - if scenario has top-level "data" dict, uses it
    else uses solver_core.make_base_data(overrides=baseline_overrides)
"""

import json
import os
import sys
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

from HeuristicBaseModel import run_heuristic  # ✅ baseline burada

# ✅ Open-source rescheduling + base data builder burada olmalı
from solver_core import make_base_data, solve_reschedule


# ==========================================================
#  JSON IO
# ==========================================================

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: dict):
    # Atomic write to avoid half-written / invalid JSON if the script crashes.
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# ============================
#  Scenario parsing helpers
# ============================

def _get_scenario_dict(path: str) -> dict:
    sc = _load_json(path)

    # allow nesting under "reschedule"
    if "reschedule" in sc and isinstance(sc["reschedule"], dict):
        merged = dict(sc)
        for k, v in sc["reschedule"].items():
            if k not in merged:
                merged[k] = v
        # keep original too
        merged["_reschedule_block"] = sc["reschedule"]
        sc = merged

    return sc

def _get_plan_start_iso(sc: dict) -> Optional[str]:
    return sc.get("plan_start_iso") or sc.get("planStartISO") or sc.get("plan_start")

def _get_scenario_name(sc: dict) -> str:
    return sc.get("scenario_name") or sc.get("name") or sc.get("scenario") or "Unnamed Scenario"

def _get_overrides(sc: dict) -> dict:
    # baseline_overrides used for baseline creation; overrides for general
    return sc.get("baseline_overrides") or sc.get("overrides") or {}

def _get_disruptions(sc: dict) -> Tuple[list, list]:
    dis = sc.get("disruptions") or {}
    um = dis.get("unavailable_machines") or sc.get("unavailable_machines") or []
    us = dis.get("unavailable_stations") or sc.get("unavailable_stations") or []
    return um, us

def _get_mode(sc: dict) -> str:
    # only used if solver_core supports modes
    return sc.get("mode") or "default"

def _get_k1(sc: dict) -> float:
    return float(sc.get("k1", 2.0))

def _get_urgent_payload(sc: dict, base_dir: str) -> Optional[dict]:
    # supports:
    #  urgent_job : dict
    #  urgent_payload : dict
    #  urgent_payload_path : string
    if isinstance(sc.get("urgent_job"), dict):
        return sc["urgent_job"]
    if isinstance(sc.get("urgent_payload"), dict):
        return sc["urgent_payload"]

    p = sc.get("urgent_payload_path")
    if isinstance(p, str) and p.strip():
        p2 = p
        if not os.path.isabs(p2):
            p2 = os.path.join(base_dir, p2)
        if os.path.exists(p2):
            return _load_json(p2)
    return None

def _get_t_now_iso(sc: dict) -> Optional[str]:
    # user may define reschedule time explicitly
    return sc.get("t_now_iso") or sc.get("reschedule_time_iso") or sc.get("tNowISO")

def _parse_iso_dt(iso_str: str) -> datetime:
    # Accepts with/without timezone. If none, assume UTC.
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _compute_t0_from_iso(plan_start_iso: str, t_now_iso: str) -> float:
    ps = _parse_iso_dt(plan_start_iso)
    tn = _parse_iso_dt(t_now_iso)
    return max(0.0, (tn - ps).total_seconds() / 3600.0)


# ==========================================================
#  Paths
# ==========================================================

def _get_this_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _baseline_path() -> str:
    return os.path.join(_get_this_dir(), "baseline_solution.json")

def _reschedule_path() -> str:
    return os.path.join(_get_this_dir(), "reschedule_solution.json")

def _gantt_dir_for_scenario(sc: dict) -> str:
    name = _get_scenario_name(sc)
    # simple folder-friendly label
    safe = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name).strip("_")
    if not safe:
        safe = "scenario"
    return os.path.join(_get_this_dir(), f"{safe.lower()}_gantts")


# ==========================================================
#  Baseline (optional) helper
# ==========================================================

def ensure_baseline(sc: dict, scenario_path: str) -> dict:
    """
    Baseline must exist for rescheduling. If missing, generate it using the same data logic as baseline runner.
    """
    bp = _baseline_path()
    if os.path.exists(bp):
        return _load_json(bp)

    print("⚠️ baseline_solution.json not found. Creating baseline now...")

    overrides = _get_overrides(sc)
    if isinstance(sc.get("data"), dict):
        data = sc["data"]
    else:
        data = make_base_data(overrides=overrides)

    res = run_heuristic(data, k1=_get_k1(sc))

    out = {
        "scenario": _get_scenario_name(sc),
        "plan_start_iso": _get_plan_start_iso(sc),
        "objective": {"T_max": float(res.T_max), "C_max": float(res.C_max)},
        "schedule": [
            {
                "op_id": int(i),
                "op_label": str(i),
                "job_id": int(data["job_of"].get(int(i), -1)) if "job_of" in data else -1,
                "start": float(res.S[int(i)]),
                "finish": float(res.C[int(i)]),
                "machine": int(res.assign_machine.get(int(i))) if int(i) in res.assign_machine else None,
                "station": int(res.assign_station.get(int(i))) if int(i) in res.assign_station else None,
            }
            for i in sorted([int(x) for x in data["I"]], key=lambda ii: (res.S[ii], res.C[ii], ii))
        ],
        "note": "Baseline generated from run_reschedule.py (missing baseline_solution.json).",
    }

    _save_json(bp, out)
    print(f"✅ Baseline saved: {bp}")
    return out


# ==========================================================
#  Main
# ==========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: py run_reschedule.py scenarios/<scenario.json>")
        sys.exit(1)

    scenario_path = sys.argv[1]
    if not os.path.exists(scenario_path):
        print(f"❌ Scenario file not found: {scenario_path}")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(scenario_path))
    sc = _get_scenario_dict(scenario_path)

    print(f"scenario: {_get_scenario_name(sc)}")
    plan_start_iso = _get_plan_start_iso(sc)
    if plan_start_iso:
        print(f"plan_start_iso: {plan_start_iso}")

    # baseline required
    baseline = ensure_baseline(sc, scenario_path)

    # base data
    overrides = _get_overrides(sc)
    if isinstance(sc.get("data"), dict):
        data_base = sc["data"]
    else:
        data_base = make_base_data(overrides=overrides)

    # disruptions
    unavailable_machines, unavailable_stations = _get_disruptions(sc)

    # urgent payload
    urgent_payload = _get_urgent_payload(sc, base_dir)

    # reschedule time
    t0 = None
    t_now_iso = _get_t_now_iso(sc)
    if plan_start_iso and t_now_iso:
        try:
            t0 = _compute_t0_from_iso(plan_start_iso, t_now_iso)
            print(f"t_now_iso: {t_now_iso}  => t0(hours): {t0:.3f}")
        except Exception as e:
            print(f"⚠️ Could not parse ISO times for t0. ({e})")

    mode = _get_mode(sc)
    k1 = _get_k1(sc)

    # solve reschedule (ONE ENGINE)
    res = solve_reschedule(
        data_base=data_base,
        old_solution=baseline,
        urgent_payload=urgent_payload,
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        t0_override=t0,
        k1=k1,
        mode=mode,
    )

    out_path = _reschedule_path()
    _save_json(out_path, res)
    print(f"✅ Reschedule saved: {out_path}")
    print(f"objective: {res.get('objective')}")

    # plots (baseline + reschedule)
    gantt_dir = _gantt_dir_for_scenario(sc)
    os.makedirs(gantt_dir, exist_ok=True)

    print("📊 Generating Gantt charts...")
    try:
        # baseline + reschedule plotter
        plotter = os.path.join(_get_this_dir(), "plot_gantt_all.py")
        if os.path.exists(plotter):
            subprocess.run([sys.executable, plotter, gantt_dir], check=False)
        else:
            # fallback baseline plotter
            plotter2 = os.path.join(_get_this_dir(), "plot_gantt_baseline.py")
            if os.path.exists(plotter2):
                subprocess.run([sys.executable, plotter2, gantt_dir], check=False)
    except Exception as e:
        print(f"⚠️ Plot generation failed: {e}")

if __name__ == "__main__":
    main()
