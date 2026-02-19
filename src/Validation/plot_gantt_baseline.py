#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
from datetime import datetime, time, timedelta, timezone

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unique_sorted(vals):
    vals = [v for v in vals if v is not None]
    return sorted(set(vals), key=lambda x: (int(x) if str(x).isdigit() else str(x)))


def _norm_res(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        return s
    try:
        if isinstance(v, float) and v.is_integer():
            return int(v)
    except Exception:
        pass
    return v


def _make_job_color_map(schedule):
    jobs = _unique_sorted(
        [
            r.get("job_id")
            for r in schedule
            if r.get("job_id") is not None and int(r.get("job_id")) >= 0
        ]
    )
    cmap = plt.get_cmap("tab20")
    job_color = {}
    for idx, j in enumerate(jobs):
        job_color[int(j)] = cmap(idx % 20)
    return job_color


def _extract_sid(meta: dict) -> str:
    for key in ["scenario_id", "scenario_file", "scenario_name", "name", "scenario"]:
        v = meta.get(key, None)
        if not v:
            continue
        s = str(v)
        m2 = re.search(r"scenario[_\- ]*(\d+)", s, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).zfill(2)
        m = re.search(r"\b(\d{1,2})\b", s)
        if m:
            return m.group(1).zfill(2)
    return "xx"


def _parse_iso(s):
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
    m = re.fullmatch(r"([+-])(\d{2}):(\d{2})", s)
    if not m:
        return timezone.utc
    sign = 1 if m.group(1) == "+" else -1
    hh = int(m.group(2))
    mm = int(m.group(3))
    return timezone(sign * timedelta(hours=hh, minutes=mm))


def _calendar_from_meta(meta: dict):
    cal = meta.get("plan_calendar", {}) if isinstance(meta, dict) else {}
    if not isinstance(cal, dict):
        cal = {}

    workdays_raw = cal.get("workdays", [0, 1, 2, 3, 4])
    workdays = set()
    for wd in workdays_raw if isinstance(workdays_raw, list) else [0, 1, 2, 3, 4]:
        try:
            i = int(wd)
            if 0 <= i <= 6:
                workdays.add(i)
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

    tz = _parse_utc_offset(cal.get("utc_offset", "+00:00"))
    plan_start = _parse_iso(meta.get("plan_start_iso")) or datetime.now(timezone.utc)
    plan_local = plan_start.astimezone(tz)

    return {
        "tz": tz,
        "plan_local": plan_local,
        "workdays": workdays,
        "shift_start": shift_start,
        "workday_hours": workday_hours,
    }


def _next_workday(d, workdays):
    cur = d
    for _ in range(8):
        cur = cur + timedelta(days=1)
        if cur.weekday() in workdays:
            return cur
    return cur


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


def _business_hours_to_local_dt(hours, cal):
    try:
        rem = float(hours)
    except Exception:
        rem = 0.0
    rem = max(0.0, rem)

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


def _convert_schedule_to_local(schedule, cal):
    out = []
    for r in schedule:
        if r.get("start") is None or r.get("finish") is None:
            continue
        try:
            st_h = float(r["start"])
            en_h = float(r["finish"])
        except Exception:
            continue
        st_dt = _business_hours_to_local_dt(st_h, cal)
        en_dt = _business_hours_to_local_dt(en_h, cal)
        rr = dict(r)
        rr["_start_local"] = st_dt
        rr["_finish_local"] = en_dt
        out.append(rr)
    return out


def _resource_labels(resources, label_map):
    if not isinstance(label_map, dict):
        return [str(r) for r in resources]
    labels = []
    for r in resources:
        k = str(r)
        labels.append(str(label_map.get(k, r)))
    return labels


def _plot_day_gantt(
    schedule,
    key: str,
    title: str,
    out_png: str,
    day_start,
    day_end,
    shift_start,
    workday_hours,
    full_resources=None,
    resource_label_map=None,
    is_weekend=False,
):
    if full_resources is not None:
        resources = [_norm_res(x) for x in list(full_resources)]
    else:
        resources = _unique_sorted([_norm_res(r.get(key)) for r in schedule if key in r])

    if not resources:
        print(f"WARNING: No resources found for {key}.")
        return

    idx = {res: i for i, res in enumerate(resources)}
    job_color = _make_job_color_map(schedule)

    rows = [r for r in schedule if (key in r and r.get("_start_local") is not None and r.get("_finish_local") is not None)]
    rows.sort(key=lambda r: (idx.get(_norm_res(r.get(key)), 10**9), r["_start_local"]))

    fig_h = max(6.0, min(18.0, 2.5 + 0.45 * len(resources)))
    fig, ax = plt.subplots(figsize=(16, fig_h))

    for r in rows:
        if is_weekend:
            continue
        res = _norm_res(r.get(key))
        y = idx.get(res, None)
        if y is None:
            continue

        seg_start = max(r["_start_local"], day_start)
        seg_end = min(r["_finish_local"], day_end)
        if seg_end <= seg_start:
            continue

        left = (seg_start - day_start).total_seconds() / 3600.0
        dur = (seg_end - seg_start).total_seconds() / 3600.0

        jid = int(r.get("job_id", -1))
        color = job_color.get(jid, (0.75, 0.75, 0.75, 1.0))

        ax.barh(y, dur, left=left, height=0.6, color=color, edgecolor="black", linewidth=0.6)

        if dur >= 0.2:
            label = str(r.get("op_label", r.get("op_id", "")))
            ax.text(left + max(0.03, dur * 0.02), y, label, va="center", fontsize=8, color="black")

    tick_count = int(round(workday_hours))
    tick_count = max(1, tick_count)
    ax.set_xlim(0.0, workday_hours)
    ax.set_xticks([float(h) for h in range(0, tick_count + 1)])
    ax.set_xticklabels([f"{(shift_start.hour + h) % 24:02d}:00" for h in range(0, tick_count + 1)])
    ax.set_xlabel("Saat")
    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels(_resource_labels(resources, resource_label_map))
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.3, alpha=0.4)

    if is_weekend:
        ax.text(workday_hours / 2.0, max(0, len(resources) - 1), "Hafta Sonu", ha="center", va="center", fontsize=10, alpha=0.7)

    legend_jobs = sorted([j for j in job_color.keys() if j >= 0])
    handles = [Patch(facecolor=job_color[j], edgecolor="black", label=f"Job {j}") for j in legend_jobs]
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Color -> Job")

    fig.subplots_adjust(left=0.07, right=0.82, top=0.90, bottom=0.12)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else "baseline_solution.json"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(baseline_path))
    sid_override = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(baseline_path):
        print(f"ERROR: baseline file not found: {baseline_path}")
        sys.exit(1)

    base = _load(baseline_path)
    base_sched = base.get("schedule", [])

    sid = sid_override.zfill(2) if (sid_override and str(sid_override).isdigit()) else _extract_sid(base)

    os.makedirs(outdir, exist_ok=True)

    for fn in os.listdir(outdir):
        if fn.startswith(f"scenario{sid}_baseline_machine_") and fn.endswith(".png"):
            os.remove(os.path.join(outdir, fn))
        if fn.startswith(f"scenario{sid}_baseline_station_") and fn.endswith(".png"):
            os.remove(os.path.join(outdir, fn))

    cal = _calendar_from_meta(base)
    schedule_local = _convert_schedule_to_local(base_sched, cal)
    if not schedule_local:
        print("WARNING: No schedule rows to plot.")
        return

    min_dt = min(r["_start_local"] for r in schedule_local)
    max_dt = max(r["_finish_local"] for r in schedule_local)

    day_cursor = min_dt.date()
    day_last = max_dt.date()

    full_machines = base.get("M", None) or _unique_sorted([r.get("machine") for r in base_sched])
    full_machines = [m for m in full_machines if _norm_res(m) not in (29, 46)]
    full_stations = base.get("L", None) or _unique_sorted([r.get("station") for r in base_sched])
    machine_label_map = base.get("machine_label_map", {}) or {}
    station_label_map = base.get("station_label_map", {}) or {}

    while day_cursor <= day_last:
        day_start = datetime.combine(day_cursor, cal["shift_start"], tzinfo=cal["tz"])
        day_end = day_start + timedelta(hours=cal["workday_hours"])
        weekday_name = day_start.strftime("%A")
        date_text = day_start.strftime("%Y-%m-%d")
        is_weekend = day_cursor.weekday() not in cal["workdays"]

        _plot_day_gantt(
            schedule_local,
            "machine",
            f"Scenario {sid} - Baseline (Machine) - {date_text} {weekday_name}",
            os.path.join(outdir, f"scenario{sid}_baseline_machine_d{date_text}.png"),
            day_start,
            day_end,
            shift_start=cal["shift_start"],
            workday_hours=cal["workday_hours"],
            full_resources=full_machines,
            resource_label_map=machine_label_map,
            is_weekend=is_weekend,
        )

        _plot_day_gantt(
            schedule_local,
            "station",
            f"Scenario {sid} - Baseline (Station) - {date_text} {weekday_name}",
            os.path.join(outdir, f"scenario{sid}_baseline_station_d{date_text}.png"),
            day_start,
            day_end,
            shift_start=cal["shift_start"],
            workday_hours=cal["workday_hours"],
            full_resources=full_stations,
            resource_label_map=station_label_map,
            is_weekend=is_weekend,
        )

        day_cursor = day_cursor + timedelta(days=1)


if __name__ == "__main__":
    main()
