#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unique_sorted(vals):
    vals = [v for v in vals if v is not None]
    return sorted(set(vals), key=lambda x: (int(x) if str(x).isdigit() else str(x)))


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


def _plot_gantt(schedule, key: str, title: str, out_png: str, t0: float | None = None):
    rows = []
    for r in schedule:
        if key not in r:
            continue
        if r.get("start") is None or r.get("finish") is None:
            continue
        rows.append(r)

    if not rows:
        print(f"⚠️ No rows to plot for {key}.")
        return

    resources = _unique_sorted([r.get(key) for r in rows])
    if not resources:
        print(f"⚠️ No resources found for {key}.")
        return

    idx = {res: i for i, res in enumerate(resources)}

    # consistent colors per job_id
    job_color = _make_job_color_map(rows)

    # Sort bars by resource then start
    rows.sort(key=lambda r: (idx.get(r.get(key), 10**9), float(r["start"])))

    fig, ax = plt.subplots(figsize=(16, 7))

    for r in rows:
        res = r.get(key)
        y = idx.get(res, None)
        if y is None:
            continue

        start = float(r["start"])
        finish = float(r["finish"])
        dur = max(0.0, finish - start)
        if dur <= 0:
            continue

        jid = int(r.get("job_id", -1))
        color = job_color.get(jid, None)

        ax.barh(y, dur, left=start, height=0.6, color=color)

        label = str(r.get("op_label", r.get("op_id", "")))
        ax.text(start + dur * 0.02, y, label, va="center", fontsize=8)

    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels([str(r) for r in resources])
    ax.set_xlabel("Time (hours)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    # t0 marker line (reschedule moment)
    if t0 is not None:
        ax.axvline(x=float(t0), linestyle="--", linewidth=2.0)
        ax.text(float(t0), len(resources) - 0.2, "t0 (reschedule)", rotation=90,
                va="top", ha="right", fontsize=9)

    # legend (Color -> Job)
    legend_jobs = sorted([j for j in job_color.keys() if j >= 0])
    handles = [Patch(facecolor=job_color[j], edgecolor="none", label=f"Job {j}") for j in legend_jobs]
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Color → Job")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_png}")


def main():
    base_path = "baseline_solution.json"
    res_path = "reschedule_solution.json"

    if not os.path.exists(base_path):
        raise FileNotFoundError("Missing baseline_solution.json")
    if not os.path.exists(res_path):
        raise FileNotFoundError("Missing reschedule_solution.json")

    base = _load(base_path)
    res = _load(res_path)

    base_sched = base.get("schedule", [])
    res_sched = res.get("schedule", [])

    t0 = res.get("t0", None)
    try:
        t0 = float(t0) if t0 is not None else None
    except Exception:
        t0 = None

    _plot_gantt(base_sched, "machine", "Baseline — Machine-wise Gantt", "gantt_machine_baseline.png", t0=None)
    _plot_gantt(base_sched, "station", "Baseline — Station-wise Gantt", "gantt_station_baseline.png", t0=None)

    _plot_gantt(res_sched, "machine", "Reschedule — Machine-wise Gantt", "gantt_machine_reschedule.png", t0=t0)
    _plot_gantt(res_sched, "station", "Reschedule — Station-wise Gantt", "gantt_station_reschedule.png", t0=t0)


if __name__ == "__main__":
    main()
