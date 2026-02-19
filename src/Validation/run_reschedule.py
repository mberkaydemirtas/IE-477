#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import subprocess
from typing import Optional, Tuple
from datetime import datetime, time, timedelta, timezone

from solver_core import solve_reschedule
from adapter import build_data_from_operations

DEFAULT_BASE_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "data", "base_data.json")
)

OUT_BASELINE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "outputs", "baseline"))
OUT_RESCHEDULE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "outputs", "reschedule"))

REFERENCE_BASELINE_SOLUTION = os.path.join(OUT_BASELINE_DIR, "base_data_baseline_solution.json")

WINDOW_HOURS = 50.0
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

def _parse_iso_dt(s: str):
    if not s or not isinstance(s, str):
        return None
    s2 = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def _parse_utc_offset(offset_text: str) -> timezone:
    s = str(offset_text or "+00:00").strip()
    if len(s) == 6 and s[0] in "+-" and s[3] == ":":
        try:
            sign = 1 if s[0] == "+" else -1
            hh = int(s[1:3])
            mm = int(s[4:6])
            return timezone(sign * timedelta(hours=hh, minutes=mm))
        except Exception:
            pass
    return timezone.utc

def _first_work_instant(plan_local, workdays, shift_start, workday_hours):
    cur_day = plan_local.date()
    while True:
        if cur_day.weekday() in workdays:
            day_start = datetime.combine(cur_day, shift_start, tzinfo=plan_local.tzinfo)
            day_end = day_start + timedelta(hours=workday_hours)
            if plan_local <= day_start:
                return day_start
            if day_start < plan_local < day_end:
                return plan_local
        cur_day = cur_day + timedelta(days=1)

def _next_workday(d, workdays):
    cur = d
    for _ in range(8):
        cur = cur + timedelta(days=1)
        if cur.weekday() in workdays:
            return cur
    return cur

def _business_hours_to_local_dt(hours: float, cal: dict):
    rem = max(0.0, float(hours))
    workdays = cal["workdays"]
    shift_start = cal["shift_start"]
    workday_hours = cal["workday_hours"]
    cur = _first_work_instant(cal["plan_local"], workdays, shift_start, workday_hours)

    while rem > 1e-9:
        day_start = datetime.combine(cur.date(), shift_start, tzinfo=cur.tzinfo)
        day_end = day_start + timedelta(hours=workday_hours)
        if cur < day_start:
            cur = day_start
        if cur >= day_end:
            nd = _next_workday(cur.date(), workdays)
            cur = datetime.combine(nd, shift_start, tzinfo=cur.tzinfo)
            continue
        avail = (day_end - cur).total_seconds() / 3600.0
        step = min(rem, avail)
        cur = cur + timedelta(hours=step)
        rem -= step
        if rem > 1e-9 and cur >= day_end:
            nd = _next_workday(cur.date(), workdays)
            cur = datetime.combine(nd, shift_start, tzinfo=cur.tzinfo)
    return cur

def _build_calendar(plan_start_iso: str, plan_calendar: dict):
    ps = _parse_iso_dt(plan_start_iso)
    if ps is None:
        return None
    cal = plan_calendar if isinstance(plan_calendar, dict) else {}
    tz = _parse_utc_offset(cal.get("utc_offset", "+00:00"))
    workdays_raw = cal.get("workdays", [0, 1, 2, 3, 4])
    workdays = set()
    for w in workdays_raw if isinstance(workdays_raw, list) else [0, 1, 2, 3, 4]:
        try:
            wi = int(w)
            if 0 <= wi <= 6:
                workdays.add(wi)
        except Exception:
            continue
    if not workdays:
        workdays = {0, 1, 2, 3, 4}
    shift_text = str(cal.get("shift_start_local", "09:00"))
    try:
        hh, mm = shift_text.split(":")[:2]
        shift_start = time(hour=int(hh), minute=int(mm))
    except Exception:
        shift_start = time(hour=9, minute=0)
    try:
        workday_hours = float(cal.get("workday_hours", 8.0))
    except Exception:
        workday_hours = 8.0
    if workday_hours <= 0:
        workday_hours = 8.0
    return {
        "plan_local": ps.astimezone(tz),
        "workdays": workdays,
        "shift_start": shift_start,
        "workday_hours": workday_hours,
    }

def _print_job_delay_report(result: dict, plan_start_iso: str = None, plan_calendar: dict = None):
    rows = result.get("job_delays", []) if isinstance(result, dict) else []
    if not isinstance(rows, list):
        rows = []
    cal = _build_calendar(plan_start_iso, plan_calendar)

    all_rows = []
    pos = 0
    zero = 0
    neg = 0
    for r in rows:
        try:
            job_id = int(r.get("job_id"))
            due = float(r.get("due", 0.0))
            comp = float(r.get("completion", 0.0))
            delay = float(r.get("delay_hours", 0.0))
            signed = comp - due
            completion_total = None
            if cal is not None:
                completion_dt = _business_hours_to_local_dt(comp, cal)
                completion_total = (completion_dt - cal["plan_local"]).total_seconds() / 3600.0
            completion_print = completion_total if completion_total is not None else comp
            all_rows.append((job_id, delay, signed, due, completion_print))
            if signed > 1e-9:
                pos += 1
            elif signed < -1e-9:
                neg += 1
            else:
                zero += 1
        except Exception:
            continue

    all_rows.sort(key=lambda x: x[0])
    print(f"jobs_total: {len(all_rows)}")
    print(f"signed_delay_counts: pos={pos}, zero={zero}, neg={neg}")
    print("all_job_delays:")
    for job_id, delay, signed, due, completion in all_rows:
        print(
            f"job_id={job_id}, tardiness={delay:.3f}, "
            f"signed_delay_hours={signed:.3f}, due={due:.3f}, completion={completion:.3f}"
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
    if not t_now_iso:
        t_now_iso = DEFAULT_REFERENCE_NOW_ISO
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
    _print_job_delay_report(
        res,
        plan_start_iso=baseline_plan_start_iso,
        plan_calendar=plan_calendar,
    )

    compare_dir = os.path.join(OUT_RESCHEDULE_DIR, f"scenario{sid_txt}_compare_gantts")
    os.makedirs(compare_dir, exist_ok=True)

    print("Generating comparison Gantt charts...")
    _run_plotter("plot_gantt_baseline.py", REFERENCE_BASELINE_SOLUTION, compare_dir, sid_txt, WINDOW_HOURS, OVERLAP_HOURS)
    _run_plotter("plot_gantt_reschedule.py", out_path, compare_dir, sid_txt, WINDOW_HOURS, OVERLAP_HOURS)


if __name__ == "__main__":
    main()
