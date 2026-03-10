#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
from datetime import datetime, time, timedelta, timezone

if __package__ in (None, ""):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Validation.adapter import build_data_from_operations
from src.Validation.solver_core import _build_job_delay_rows, _build_resource_utilization

OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "outputs", "current_plan")
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


def _extract_operations(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("operations", "assignments", "items", "data", "workOrderOperationDtoList"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []


def _parse_iso_dt(value):
    if isinstance(value, (list, tuple)):
        parts = list(value)
        if len(parts) >= 3:
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3]) if len(parts) >= 4 else 0
                minute = int(parts[4]) if len(parts) >= 5 else 0
                second = int(parts[5]) if len(parts) >= 6 else 0
                microsecond = 0
                if len(parts) >= 7:
                    microsecond = max(0, min(999999, int(parts[6]) // 1000))
                return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)
            except Exception:
                return None
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
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


def _local_dt_to_business_hours(dt, cal: dict) -> float:
    if dt is None or cal is None:
        return 0.0

    target = dt.astimezone(cal["plan_local"].tzinfo)
    cur = _first_work_instant(cal["plan_local"], cal["workdays"], cal["shift_start"], cal["workday_hours"])
    if target <= cur:
        return 0.0

    total = 0.0
    while cur < target:
        day_start = datetime.combine(cur.date(), cal["shift_start"], tzinfo=cur.tzinfo)
        day_end = day_start + timedelta(hours=cal["workday_hours"])
        if cur < day_start:
            cur = day_start
        if cur >= day_end:
            nd = _next_workday(cur.date(), cal["workdays"])
            cur = datetime.combine(nd, cal["shift_start"], tzinfo=cur.tzinfo)
            continue
        step_end = min(day_end, target)
        if step_end > cur:
            total += (step_end - cur).total_seconds() / 3600.0
            cur = step_end
        if cur >= day_end and cur < target:
            nd = _next_workday(cur.date(), cal["workdays"])
            cur = datetime.combine(nd, cal["shift_start"], tzinfo=cur.tzinfo)
    return total


def _op_to_job_map(o_j: dict):
    out = {}
    for j, ops in (o_j or {}).items():
        for op_id in ops or []:
            out[int(op_id)] = int(j)
    return out


def _normalize_input(raw):
    if isinstance(raw, list):
        return {"operations": raw}
    if isinstance(raw, dict):
        if "operations" in raw and isinstance(raw["operations"], list):
            return raw
        for alt_key in ("assignments", "items", "data", "workOrderOperationDtoList"):
            alt_val = raw.get(alt_key)
            if isinstance(alt_val, list):
                return {**raw, "operations": alt_val}
    return raw


def build_current_plan_solution(raw: dict, base_data_path: str) -> dict:
    base_data = _normalize_input(raw)
    operations = _extract_operations(raw)

    plan_start_iso = base_data.get("reference_now_iso") or DEFAULT_REFERENCE_NOW_ISO
    plan_calendar = (
        base_data.get("plan_calendar")
        or base_data.get("calendar")
        or {"utc_offset": "+03:00"}
    )

    adapter_data = build_data_from_operations(
        operations,
        base_data,
        plan_start_iso=plan_start_iso,
        plan_calendar=plan_calendar,
        location_map=base_data.get("location_map", {}) or {},
        system_config=None,
    )

    cal = _build_calendar(plan_start_iso, plan_calendar)
    job_of = _op_to_job_map(adapter_data.get("O_j", {}))

    schedule = []
    s_old = {}
    c_old = {}
    x_old = {}
    y_old = {}

    for op in operations:
        try:
            op_id = int(op.get("id"))
        except Exception:
            continue
        st = _parse_iso_dt(op.get("plannedStartDateTime"))
        en = _parse_iso_dt(op.get("plannedEndDateTime"))
        if st is None or en is None:
            continue

        st_h = float(_local_dt_to_business_hours(st, cal))
        en_h = float(_local_dt_to_business_hours(en, cal))
        if en_h < st_h:
            en_h = st_h

        machine = op.get("workCenterMachineId")
        station = op.get("workCenterId")

        row = {
            "op_id": op_id,
            "op_label": str(op_id),
            "job_id": int(job_of.get(op_id, -1)),
            "start": st_h,
            "finish": en_h,
            "machine": int(machine) if machine is not None else None,
            "station": int(station) if station is not None else None,
        }
        schedule.append(row)
        s_old[op_id] = st_h
        c_old[op_id] = en_h
        if row["machine"] is not None:
            x_old[f"{op_id},{int(row['machine'])}"] = 1
        if row["station"] is not None:
            y_old[f"{op_id},{int(row['station'])}"] = 1

    schedule.sort(key=lambda r: (r["start"], r["finish"], r["op_id"]))

    c_final = {}
    tardiness = {}
    for j in adapter_data.get("J", []):
        ops = [int(x) for x in adapter_data.get("O_j", {}).get(int(j), [])]
        finish_vals = [c_old[i] for i in ops if i in c_old]
        comp = max(finish_vals) if finish_vals else 0.0
        due = float(adapter_data.get("d_j", {}).get(int(j), 0.0))
        c_final[int(j)] = comp
        tardiness[int(j)] = max(comp - due, 0.0)

    t_max = max(tardiness.values()) if tardiness else 0.0
    c_max = max((r["finish"] for r in schedule), default=0.0)

    job_delays = _build_job_delay_rows(
        J=adapter_data.get("J", []),
        d_j=adapter_data.get("d_j", {}),
        C_final=c_final,
        T=tardiness,
    )
    machine_utilization = _build_resource_utilization(
        schedule=schedule,
        resource_key="machine",
        resource_ids=list(adapter_data.get("M", [])),
        label_map=dict(adapter_data.get("machine_label_map", {}) or {}),
    )
    station_utilization = _build_resource_utilization(
        schedule=schedule,
        resource_key="station",
        resource_ids=list(adapter_data.get("L", [])),
        label_map=dict(adapter_data.get("station_label_map", {}) or {}),
    )

    return {
        "plan_start_iso": plan_start_iso,
        "plan_calendar": plan_calendar,
        "objective": {"T_max": float(t_max), "C_max": float(c_max)},
        "S_old": s_old,
        "C_old": c_old,
        "x_old": x_old,
        "y_old": y_old,
        "schedule": schedule,
        "job_delays": job_delays,
        "machine_utilization": machine_utilization,
        "station_utilization": station_utilization,
        "note": "Current plan reconstructed from plannedStartDateTime/plannedEndDateTime",
        "M": list(adapter_data.get("M", [])),
        "L": list(adapter_data.get("L", [])),
        "machine_label_map": dict(adapter_data.get("machine_label_map", {}) or {}),
        "station_label_map": dict(adapter_data.get("station_label_map", {}) or {}),
        "base_data_file": os.path.basename(base_data_path),
        "result_type": "current_plan",
        "current_plan_run_iso_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: py run_current_plan.py data/input.json")
        sys.exit(1)

    base_data_path = sys.argv[1]
    if not os.path.exists(base_data_path):
        print(f"ERROR Input file not found: {base_data_path}")
        sys.exit(1)

    raw = _load_json(base_data_path)
    result = build_current_plan_solution(raw, base_data_path)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "current_plan_solution.json")
    _save_json_atomic(out_json, result)
    print(f"Current plan saved: {out_json}")
    print("plan_start_iso:", result.get("plan_start_iso"))
    print("objective:", result.get("objective"))

    gantt_dir = os.path.join(OUT_DIR, "current_plan_gantts")
    os.makedirs(gantt_dir, exist_ok=True)

    plotter = os.path.join(os.path.dirname(__file__), "plot_gantt_baseline.py")
    if os.path.exists(plotter):
        print("Generating CURRENT PLAN Gantt charts...")
        subprocess.run(
            [
                sys.executable,
                plotter,
                out_json,
                gantt_dir,
                "current",
                str(WINDOW_HOURS),
                str(OVERLAP_HOURS),
            ],
            cwd=os.path.dirname(__file__),
            check=False,
        )


if __name__ == "__main__":
    main()
