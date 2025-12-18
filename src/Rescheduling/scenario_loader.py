#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def deep_update(dst: dict, src: dict) -> dict:
    """Recursive dict merge: src overwrites dst."""
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_scenario(scenario_path: str) -> dict:
    """
    Scenario JSON schema (minimal):
      {
        "name": "scenario_A",
        "plan_calendar": {...},
        "data_overrides": {...},
        "baseline_output": "baseline_solution.json",
        "reschedule_output": "reschedule_solution.json",
        "reschedule": {
          "mode": "continue" | "optimize",
          "unavailable_machines": [4,5],
          "unavailable_stations": [2,9],
          "urgent_payload_path": "scenarios/urgent_A.json"
        }
      }
    """
    sc = load_json(scenario_path)

    # defaults
    defaults = {
        "name": Path(scenario_path).stem,
        "plan_calendar": {
            "mode": "realtime_anchor",
            "utc_offset": "+03:00",
            "workdays": [0, 1, 2, 3, 4],
            "shift_start_local": "08:00",
            "workday_hours": 8.0,
            "lunch_break_included": False
        },
        "data_overrides": {},
        "baseline_output": f"baseline_solution_{Path(scenario_path).stem}.json",
        "reschedule_output": f"reschedule_solution_{Path(scenario_path).stem}.json",
        "reschedule": {
            "mode": "continue",
            "unavailable_machines": [],
            "unavailable_stations": [],
            "urgent_payload_path": ""
        }
    }

    sc = deep_update(defaults, sc)

    # normalize lists
    sc["reschedule"]["unavailable_machines"] = [int(x) for x in sc["reschedule"].get("unavailable_machines", [])]
    sc["reschedule"]["unavailable_stations"] = [int(x) for x in sc["reschedule"].get("unavailable_stations", [])]

    return sc

def load_urgent_payload(path: str) -> dict:
    if not path:
        raise ValueError("urgent_payload_path is empty in scenario json.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Urgent payload file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Urgent payload must be a JSON object (dict).")
    return payload
