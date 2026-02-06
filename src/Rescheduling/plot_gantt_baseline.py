#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _unique_sorted(vals):
    vals = [v for v in vals if v is not None]
    return sorted(set(vals), key=lambda x: (int(x) if str(x).isdigit() else str(x)))

def _norm_res(v):
    """Normalize resource IDs so '1' and 1 map to same key."""
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
    jobs = _unique_sorted([
        r.get("job_id")
        for r in schedule
        if r.get("job_id") is not None and int(r.get("job_id")) >= 0
    ])
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

def _time_span(rows):
    if not rows:
        return 0.0, 0.0
    tmin = min(float(r["start"]) for r in rows if r.get("start") is not None)
    tmax = max(float(r["finish"]) for r in rows if r.get("finish") is not None)
    return tmin, tmax

def _make_windows(tmin, tmax, window, overlap=0.0):
    # window <= 0 => single plot
    if window is None or window <= 0 or (tmax - tmin) <= window:
        return [(tmin, tmax)]
    step = max(1e-9, window - max(0.0, overlap))
    windows = []
    s = tmin
    while s < tmax:
        e = s + window
        windows.append((s, min(e, tmax)))
        s += step
        if len(windows) > 5000:  # safety
            break
    return windows

def _plot_gantt(schedule, key: str, title: str, out_png: str, xlim=None, full_resources=None):
    rows = [r for r in schedule if (key in r and r.get("start") is not None and r.get("finish") is not None)]
    if not rows:
        print(f"⚠️ No rows to plot for {key}.")
        return

    # resources: show all even if unused + normalize
    if full_resources is not None:
        resources = [_norm_res(x) for x in list(full_resources)]
    else:
        resources = _unique_sorted([_norm_res(r.get(key)) for r in rows])

    if not resources:
        print(f"⚠️ No resources found for {key}.")
        return

    idx = {res: i for i, res in enumerate(resources)}
    job_color = _make_job_color_map(rows)

    rows.sort(key=lambda r: (idx.get(_norm_res(r.get(key)), 10**9), float(r["start"])))

    fig, ax = plt.subplots(figsize=(16, 7))

    ws = we = None
    if xlim is not None:
        try:
            ws, we = float(xlim[0]), float(xlim[1])
        except Exception:
            ws = we = None

    for r in rows:
        res = _norm_res(r.get(key))
        y = idx.get(res, None)
        if y is None:
            continue

        start = float(r["start"])
        finish = float(r["finish"])

        if ws is not None and we is not None:
            if finish <= ws or start >= we:
                continue

        dur = max(0.0, finish - start)
        if dur <= 0:
            continue

        jid = int(r.get("job_id", -1))
        color = job_color.get(jid, (0.75, 0.75, 0.75, 1.0))

        ax.barh(y, dur, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.6)

        label = str(r.get("op_label", r.get("op_id", "")))
        ax.text(start + dur * 0.02, y, label, va="center", fontsize=8, color="black")

    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels([str(r) for r in resources])
    ax.set_xlabel("Time (hours)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.3, alpha=0.4)

    if ws is not None and we is not None:
        ax.set_xlim(ws, we)

    legend_jobs = sorted([j for j in job_color.keys() if j >= 0])
    handles = [Patch(facecolor=job_color[j], edgecolor="black", label=f"Job {j}") for j in legend_jobs]
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Color → Job")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_png}")

def main():
    # usage:
    #   py plot_gantt_baseline.py <baseline_json_path> <outdir> [sid_override] [window_hours] [overlap_hours]
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else "baseline_solution.json"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(baseline_path))
    sid_override = sys.argv[3] if len(sys.argv) > 3 else None

    window = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    overlap = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    base = _load(baseline_path)
    base_sched = base.get("schedule", [])

    sid = sid_override.zfill(2) if (sid_override and str(sid_override).isdigit()) else _extract_sid(base)

    os.makedirs(outdir, exist_ok=True)

    rows_machine = [r for r in base_sched if ("machine" in r and r.get("start") is not None and r.get("finish") is not None)]
    tmin, tmax = _time_span(rows_machine)
    windows = _make_windows(tmin, tmax, window, overlap)

    full_machines = list(range(1, 13))
    full_stations = list(range(1, 10))

    for w_i, (ws, we) in enumerate(windows, start=1):
        suffix = f"_w{w_i:02d}_{ws:.1f}-{we:.1f}"
        _plot_gantt(
            base_sched,
            "machine",
            f"Scenario {sid} — Baseline (Machine Gantt) [{ws:.1f}, {we:.1f}]",
            os.path.join(outdir, f"scenario{sid}_baseline_machine{suffix}.png"),
            xlim=(ws, we),
            full_resources=full_machines,
        )

    for w_i, (ws, we) in enumerate(windows, start=1):
        suffix = f"_w{w_i:02d}_{ws:.1f}-{we:.1f}"
        _plot_gantt(
            base_sched,
            "station",
            f"Scenario {sid} — Baseline (Station Gantt) [{ws:.1f}, {we:.1f}]",
            os.path.join(outdir, f"scenario{sid}_baseline_station{suffix}.png"),
            xlim=(ws, we),
            full_resources=full_stations,
        )

if __name__ == "__main__":
    main()
