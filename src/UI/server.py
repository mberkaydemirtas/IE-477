#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py – Flask backend for the Teknopar Production Scheduling Website.

Endpoints:
  POST /optimize        – Run baseline optimizer on uploaded JSON, generate Gantt charts
  GET  /charts          – List available Gantt PNG filenames
  GET  /chart/<name>    – Serve a specific PNG
  POST /reschedule      – Run rescheduling (for future use)

Usage:
  pip install flask flask-cors matplotlib
  python server.py
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Add project module directories to sys.path so imports work when launched from UI.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "Validation"))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, VALIDATION_DIR)

from src.Validation.adapter import build_data_from_operations
from src.Validation.solver_core import solve_baseline, solve_reschedule

app = Flask(__name__)
CORS(app)  # Allow requests from the website (file:// or localhost)

# Output directory for generated charts
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
BASELINE_DIR = os.path.join(OUTPUTS_DIR, "baseline")
BASELINE_SOLUTION_PATH = os.path.join(BASELINE_DIR, "base_data_baseline_solution.json")

DEFAULT_REFERENCE_NOW_ISO = "2026-01-30T00:00:00+00:00"
WINDOW_HOURS = 50.0
OVERLAP_HOURS = 0.0

BASELINE_SCHEDULE_OUTPUT_PATH = os.path.join(BASELINE_DIR, "base_data_schedule_output.json")


def _save_json_atomic(path: str, obj: dict):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)



# ---------------------------------------------------------------------------
# Calendar / datetime helpers (mirrors run_baseline.py)
# ---------------------------------------------------------------------------

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


def _parse_any_dt(value):
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
    return _parse_iso_dt(value)


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
    """Convert business-hours offset from plan_start into a real local datetime."""
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
    for w in (workdays_raw if isinstance(workdays_raw, list) else [0, 1, 2, 3, 4]):
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


def _build_schedule_output(original_ops: list, baseline: dict, plan_start_iso: str, plan_calendar: dict) -> list:
    """
    Build a list of operation objects in the same format as the uploaded JSON,
    but with realPlannedStartDateTime and realPlannedEndDateTime updated from
    the optimizer's schedule results.
    """
    cal = _build_calendar(plan_start_iso, plan_calendar)

    # Build op_id -> {start, finish} lookup from schedule
    sched_map = {}
    for entry in baseline.get("schedule", []):
        try:
            op_id = int(entry["op_id"])
            sched_map[op_id] = {
                "start": float(entry["start"]),
                "finish": float(entry["finish"]),
            }
        except Exception:
            continue

    result = []
    for op in original_ops:
        new_op = copy.deepcopy(op)
        try:
            op_id = int(op.get("id", -1))
        except Exception:
            op_id = -1

        if op_id in sched_map and cal is not None:
            start_bh = sched_map[op_id]["start"]
            finish_bh = sched_map[op_id]["finish"]
            start_dt = _business_hours_to_local_dt(start_bh, cal)
            finish_dt = _business_hours_to_local_dt(finish_bh, cal)
            # Format as ISO without microseconds, without timezone suffix (matches source format)
            new_op["realPlannedStartDateTime"] = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
            new_op["realPlannedEndDateTime"] = finish_dt.strftime("%Y-%m-%dT%H:%M:%S")

        result.append(new_op)
    return result


def _normalize_input(raw):
    """Accept list, {operations:[...]}, {assignments:[...]}, or bare dict."""
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


def _extract_operations(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("operations", "assignments", "items", "data", "workOrderOperationDtoList"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []


def _replace_operations(raw, operations):
    if isinstance(raw, list):
        return operations
    if not isinstance(raw, dict):
        return {"operations": operations}
    for key in ("operations", "assignments", "items", "data", "workOrderOperationDtoList"):
        if isinstance(raw.get(key), list):
            out = copy.deepcopy(raw)
            out[key] = operations
            return out
    out = copy.deepcopy(raw)
    out["operations"] = operations
    return out


def _apply_baseline_schedule_as_planned(schedule_ops: list) -> list:
    updated = []
    for op in schedule_ops or []:
        row = copy.deepcopy(op)
        real_start = row.get("realPlannedStartDateTime")
        real_end = row.get("realPlannedEndDateTime")
        if real_start:
            row["plannedStartDateTime"] = real_start
        if real_end:
            row["plannedEndDateTime"] = real_end
        updated.append(row)
    return updated


def _load_reschedule_base_data():
    raw_input_path = os.path.join(BASELINE_DIR, "uploaded_base_data.json")
    if not os.path.exists(raw_input_path):
        raise FileNotFoundError("No uploaded data found. Please re-upload and optimize.")

    with open(raw_input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if os.path.exists(BASELINE_SCHEDULE_OUTPUT_PATH):
        with open(BASELINE_SCHEDULE_OUTPUT_PATH, "r", encoding="utf-8") as f:
            schedule_output = json.load(f)
        if isinstance(schedule_output, list) and schedule_output:
            base_ops = _apply_baseline_schedule_as_planned(schedule_output)
            raw_data = _replace_operations(raw_data, base_ops)

    return raw_data


def _update_due_date_for_operation(raw_data, operation_id, new_due_date):
    ops = _extract_operations(raw_data)
    if not ops:
        return False

    target_work_order = None
    target_parent = None
    for op in ops:
        try:
            if int(op.get("id")) == int(operation_id):
                target_work_order = str(op.get("workOrderNumber") or "").strip()
                target_parent = op.get("parentId")
                break
        except Exception:
            continue

    if target_work_order is None and target_parent is None:
        return False

    updated = False
    for op in ops:
        same_work_order = target_work_order and str(op.get("workOrderNumber") or "").strip() == target_work_order
        same_parent = target_parent is not None and op.get("parentId") == target_parent
        if same_work_order or same_parent:
            op["endDate"] = new_due_date
            updated = True
    return updated


def _build_urgent_payload_from_ops(ops: list, base_data: dict, plan_start_iso: str, plan_calendar: dict):
    if not ops:
        return None

    temp_base = {"operations": copy.deepcopy(ops)}
    temp_solver = build_data_from_operations(
        temp_base["operations"],
        temp_base,
        plan_start_iso=plan_start_iso,
        plan_calendar=plan_calendar,
        location_map=base_data.get("location_map", {}) or {},
        system_config=None,
    )

    job_ids = sorted(int(j) for j in temp_solver.get("J", []))
    op_ids = sorted(int(i) for i in temp_solver.get("I", []))
    if not job_ids or not op_ids:
        return None

    due_abs = temp_solver.get("d_j", {}).get(job_ids[0], 8.0)
    due_rel = max(1.0, float(due_abs))
    urgent_ops = []
    for op_id in op_ids:
        feas_m = [int(x) for x in temp_solver.get("M_i", {}).get(op_id, [])]
        feas_l = [int(x) for x in temp_solver.get("L_i", {}).get(op_id, [])]
        pt_map = {}
        for m in feas_m:
            val = temp_solver.get("p_im", {}).get((op_id, m))
            if val is not None:
                pt_map[str(int(m))] = float(val)
        urgent_ops.append({
            "op_id": int(op_id),
            "feasible_machines": feas_m,
            "feasible_stations": feas_l,
            "processing_time_by_machine": pt_map,
        })

    preds = []
    for succ, pred_list in (temp_solver.get("Pred_i", {}) or {}).items():
        for pred in pred_list or []:
            preds.append([int(pred), int(succ)])

    return {
        "urgent_job": {
            "job_id": int(max(base_data.get("J", [0])) + 1) if base_data.get("J") else 999001,
            "release_time_mode": "t0",
            "due_time_mode": "t0_plus",
            "due_time_hours": float(due_rel),
            "ops": urgent_ops,
            "precedence_edges": preds,
            "preempt_for_urgent": True,
        }
    }


def _resolve_urgent_payload(urgent_data, raw_data, base_data, plan_start_iso, plan_calendar):
    if not urgent_data:
        return None
    if isinstance(urgent_data, dict) and (
        isinstance(urgent_data.get("urgent_job"), dict) or
        isinstance(urgent_data.get("ops"), list)
    ):
        return urgent_data

    base_ops = _extract_operations(raw_data)
    base_ids = set()
    for op in base_ops:
        try:
            base_ids.add(int(op.get("id")))
        except Exception:
            continue

    urgent_ops = []
    extracted = _extract_operations(urgent_data)
    if extracted:
        for op in extracted:
            try:
                op_id = int(op.get("id"))
            except Exception:
                op_id = None
            if op_id is None or op_id not in base_ids:
                urgent_ops.append(op)
    elif isinstance(urgent_data, dict) and urgent_data.get("id") is not None:
        urgent_ops = [urgent_data]

    return _build_urgent_payload_from_ops(urgent_ops, base_data, plan_start_iso, plan_calendar)


def _run_gantt_plotter(script_name: str, json_path: str, outdir: str, sid: str = "xx"):
    script = os.path.join(VALIDATION_DIR, script_name)
    if not os.path.exists(script):
        print(f"[server] Gantt plotter not found: {script}")
        return
    try:
        subprocess.run(
            [sys.executable, script, json_path, outdir, sid, str(WINDOW_HOURS), str(OVERLAP_HOURS)],
            cwd=VALIDATION_DIR,
            check=False,
            timeout=120,
        )
    except Exception as e:
        print(f"[server] Gantt plotter error: {e}")


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    Accept a JSON body (the uploaded operations file).
    Run solve_baseline, generate Gantt charts, return the baseline solution.
    """
    try:
        raw = request.get_json(force=True, silent=True)
        if raw is None:
            return jsonify({"error": "No JSON body received"}), 400

        base_data = _normalize_input(raw)

        plan_start_iso = (
            base_data.get("reference_now_iso")
            or DEFAULT_REFERENCE_NOW_ISO
        )

        plan_calendar = (
            base_data.get("plan_calendar")
            or base_data.get("calendar")
            or {"utc_offset": "+03:00"}
        )

        location_map = base_data.get("location_map", {}) or {}

        # Run the adapter to build solver data
        if "operations" in base_data and isinstance(base_data["operations"], list):
            solver_data = build_data_from_operations(
                base_data["operations"],
                base_data,
                plan_start_iso=plan_start_iso,
                plan_calendar=plan_calendar,
                location_map=location_map,
                system_config=None,
            )
        else:
            return jsonify({"error": "No operations list found in JSON"}), 400

        # Run baseline solver
        baseline = solve_baseline(
            solver_data,
            plan_start_iso=plan_start_iso,
            plan_calendar=plan_calendar,
            k1=float(base_data.get("k1", 2.0)),
        )

        baseline["base_data_file"] = "uploaded_data.json"
        baseline["baseline_run_iso_utc"] = datetime.now(timezone.utc).isoformat()
        baseline["result_type"] = "baseline"

        # Save the baseline solution (existing format)
        os.makedirs(BASELINE_DIR, exist_ok=True)
        _save_json_atomic(BASELINE_SOLUTION_PATH, baseline)

        # Save second JSON: uploaded format with optimized realPlannedStart/EndDateTime
        original_ops = base_data.get("operations", [])
        schedule_output = _build_schedule_output(
            original_ops, baseline, plan_start_iso, plan_calendar
        )
        _save_json_atomic(BASELINE_SCHEDULE_OUTPUT_PATH, schedule_output)
        print(f"[server] Schedule output saved: {BASELINE_SCHEDULE_OUTPUT_PATH}")

        # Also save the raw input for rescheduling later
        raw_input_path = os.path.join(BASELINE_DIR, "uploaded_base_data.json")
        _save_json_atomic(raw_input_path, raw)

        # Generate Gantt charts
        gantt_dir = os.path.join(BASELINE_DIR, "base_data_gantts")
        os.makedirs(gantt_dir, exist_ok=True)
        _run_gantt_plotter("plot_gantt_baseline.py", BASELINE_SOLUTION_PATH, gantt_dir, "xx")

        # Collect generated chart filenames
        chart_files = sorted([
            f for f in os.listdir(gantt_dir)
            if f.lower().endswith(".png")
        ])

        return jsonify({
            "status": "ok",
            "baseline": baseline,
            "schedule_output_path": BASELINE_SCHEDULE_OUTPUT_PATH,
            "chart_count": len(chart_files),
            "charts": chart_files,
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[server] /optimize error:\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/charts", methods=["GET"])
def list_charts():
    """Return list of all available PNG filenames (baseline + reschedule)."""
    all_charts = []

    # Baseline gantts
    gantt_dir = os.path.join(BASELINE_DIR, "base_data_gantts")
    if os.path.isdir(gantt_dir):
        for f in sorted(os.listdir(gantt_dir)):
            if f.lower().endswith(".png"):
                all_charts.append({"type": "baseline", "filename": f})

    # Reschedule gantts
    reschedule_dir = os.path.join(OUTPUTS_DIR, "reschedule")
    if os.path.isdir(reschedule_dir):
        for sub in sorted(os.listdir(reschedule_dir)):
            sub_path = os.path.join(reschedule_dir, sub)
            if os.path.isdir(sub_path) and "gantt" in sub.lower():
                for f in sorted(os.listdir(sub_path)):
                    if f.lower().endswith(".png"):
                        all_charts.append({"type": "reschedule", "filename": f, "folder": sub})

    return jsonify({"charts": all_charts})


@app.route("/chart/<path:filename>", methods=["GET"])
def serve_chart(filename):
    """Serve a PNG chart file by name. Searches all output subdirectories."""
    # Search baseline gantts
    gantt_dir = os.path.join(BASELINE_DIR, "base_data_gantts")
    candidate = os.path.join(gantt_dir, filename)
    if os.path.isfile(candidate):
        return send_file(candidate, mimetype="image/png")

    # Search reschedule gantts
    reschedule_dir = os.path.join(OUTPUTS_DIR, "reschedule")
    if os.path.isdir(reschedule_dir):
        for sub in os.listdir(reschedule_dir):
            sub_path = os.path.join(reschedule_dir, sub)
            if os.path.isdir(sub_path):
                candidate = os.path.join(sub_path, filename)
                if os.path.isfile(candidate):
                    return send_file(candidate, mimetype="image/png")

    return jsonify({"error": f"Chart not found: {filename}"}), 404


@app.route("/reschedule", methods=["POST"])
def reschedule():
    """
    Run rescheduling on top of the existing baseline.
    Body: { type, machineId?, newCount?, stationId?, newCount?, operationId?, newDueDate? }
    """
    try:
        if not os.path.exists(BASELINE_SOLUTION_PATH):
            return jsonify({"error": "No baseline solution found. Please optimize first."}), 400

        params = request.get_json(force=True, silent=True) or {}
        reschedule_type = params.get("type", "")

        # Load baseline and raw data
        with open(BASELINE_SOLUTION_PATH, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        raw_data = _load_reschedule_base_data()
        original_raw_data = copy.deepcopy(raw_data)

        base_data = _normalize_input(raw_data)
        plan_start_iso = baseline.get("plan_start_iso") or DEFAULT_REFERENCE_NOW_ISO
        plan_calendar = baseline.get("plan_calendar") or {"utc_offset": "+03:00"}

        # Build unavailability lists based on type
        unavailable_machines = []
        unavailable_stations = []
        urgent_payload = None
        mode = "continue"

        # BUG FIX 4: parameter names were previously swapped between the two branches.
        if reschedule_type == "station-changes":
            machine_id = params.get("machineId")
            station_id = params.get("stationId")
            if machine_id:
                try:
                    unavailable_machines = [int(machine_id)]
                except Exception:
                    pass
            if station_id:
                try:
                    unavailable_stations = [int(station_id)]
                except Exception:
                    pass

        elif reschedule_type == "machine-changes":
            station_id = params.get("stationId")
            machine_id = params.get("machineId")
            if station_id:
                try:
                    unavailable_stations = [int(station_id)]
                except Exception:
                    pass
            if machine_id:
                try:
                    unavailable_machines = [int(machine_id)]
                except Exception:
                    pass

        elif reschedule_type == "urgent-job":
            urgent_data = params.get("urgentJobData")
            if not urgent_data:
                return jsonify({"error": "urgentJobData is required"}), 400

            # Always keep the baseline-derived base_data as the reschedule source.
            # If a full updated dataset is uploaded, only the delta operations that do
            # not exist in the baseline are converted into an urgent payload. Otherwise
            # the same urgent operations would be scheduled twice.
            urgent_payload = _resolve_urgent_payload(
                urgent_data,
                original_raw_data,
                base_data,
                plan_start_iso,
                plan_calendar,
            )
            if not urgent_payload:
                return jsonify({
                    "error": "No new urgent operations were found in uploaded JSON"
                }), 400
            mode = "continue"
        elif reschedule_type == "due-date":
            operation_id = params.get("operationId")
            new_due_date = params.get("newDueDate")
            if not operation_id or not new_due_date:
                return jsonify({"error": "operationId and newDueDate are required"}), 400
            if not _update_due_date_for_operation(raw_data, operation_id, new_due_date):
                return jsonify({"error": f"Operation not found for due date change: {operation_id}"}), 400
            base_data = _normalize_input(raw_data)

        elif reschedule_type not in {"station-changes", "machine-changes", "urgent-job", "due-date"}:
            return jsonify({"error": f"Unsupported reschedule type: {reschedule_type}"}), 400

        # Build solver data using adapter
        if "operations" in base_data and isinstance(base_data["operations"], list):
            solver_data = build_data_from_operations(
                base_data["operations"],
                base_data,
                plan_start_iso=plan_start_iso,
                plan_calendar=plan_calendar,
                location_map=base_data.get("location_map", {}) or {},
                system_config=None,
            )
        else:
            return jsonify({"error": "No operations list found in base data"}), 400

        result = solve_reschedule(
            data_base=solver_data,
            old_solution=baseline,
            urgent_payload=urgent_payload,
            unavailable_machines=unavailable_machines,
            unavailable_stations=unavailable_stations,
            mode=mode,
            k1=float(base_data.get("k1", 2.0)),
            t_now_iso=baseline.get("plan_start_iso") or DEFAULT_REFERENCE_NOW_ISO,
        )

        result["result_type"] = "reschedule"
        result["reschedule_run_iso_utc"] = datetime.now(timezone.utc).isoformat()

        # Save and generate charts
        sid = "01"
        reschedule_dir = os.path.join(OUTPUTS_DIR, "reschedule")
        os.makedirs(reschedule_dir, exist_ok=True)
        out_path = os.path.join(reschedule_dir, f"scenario{sid}_reschedule_solution.json")
        _save_json_atomic(out_path, result)

        compare_dir = os.path.join(reschedule_dir, f"scenario{sid}_compare_gantts")
        os.makedirs(compare_dir, exist_ok=True)
        _run_gantt_plotter("plot_gantt_baseline.py", BASELINE_SOLUTION_PATH, compare_dir, sid)
        _run_gantt_plotter("plot_gantt_reschedule.py", out_path, compare_dir, sid)

        chart_files = sorted([
            f for f in os.listdir(compare_dir)
            if f.lower().endswith(".png")
        ])

        return jsonify({
            "status": "ok",
            "result": result,
            "chart_count": len(chart_files),
            "charts": chart_files,
            "folder": f"scenario{sid}_compare_gantts",
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[server] /reschedule error:\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "server": "Teknopar Scheduling API"})


if __name__ == "__main__":
    os.makedirs(BASELINE_DIR, exist_ok=True)
    print("=" * 55)
    print("  Teknopar Scheduling Server")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
