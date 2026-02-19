#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
from datetime import datetime, time, timedelta, timezone

from adapter import build_data_from_operations
from solver_core import solve_baseline

OUT_BASELINE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "outputs", "baseline")
)

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


def _print_utilization_report(result: dict):
    def _print_block(title: str, block: dict):
        rows = block.get("rows", []) if isinstance(block, dict) else []
        if not isinstance(rows, list):
            rows = []
        print(title)
        for r in rows:
            try:
                rid = int(r.get("resource_id"))
                label = str(r.get("label", rid))
                busy = float(r.get("busy_hours", 0.0))
                util = float(r.get("utilization_pct", 0.0))
                print(f"resource_id={rid}, label={label}, busy_hours={busy:.3f}, utilization_pct={util:.2f}")
            except Exception:
                continue

    _print_block("machine_utilization:", result.get("machine_utilization", {}))
    _print_block("station_utilization:", result.get("station_utilization", {}))


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
    _print_job_delay_report(
        baseline,
        plan_start_iso=baseline.get("plan_start_iso"),
        plan_calendar=baseline.get("plan_calendar"),
    )
    _print_utilization_report(baseline)

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
