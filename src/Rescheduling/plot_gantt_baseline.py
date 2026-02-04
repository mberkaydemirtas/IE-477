#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
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


def _extract_scenario_id(meta: dict) -> str:
    """
    Attempts to infer scenario id (e.g., '04') from meta fields:
      - meta['scenario'] like 'scenario_04_urgent_job_shock.json'
      - meta['scenario_id'] like 4 / "04"
      - meta['scenario_name'] like 'Scenario 04 — ...'
    Fallback: 'xx'
    """
    for key in ["scenario_id", "scenario", "scenario_name"]:
        v = meta.get(key, None)
        if not v:
            continue
        s = str(v)
        m = re.search(r"\b(\d{1,2})\b", s)
        if m:
            return m.group(1).zfill(2)

        # scenario_04 pattern
        m2 = re.search(r"scenario[_\- ]*(\d+)", s, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).zfill(2)

    return "xx"


def _plot_gantt(schedule, key: str, title: str, out_png: str, t0=None):
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
    job_color = _make_job_color_map(rows)

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
        color = job_color.get(jid, (0.75, 0.75, 0.75, 1.0))

        ax.barh(
            y, dur, left=start, height=0.6,
            color=color, edgecolor="black", linewidth=0.6
        )

        label = str(r.get("op_label", r.get("op_id", "")))
        ax.text(start + dur * 0.02, y, label, va="center", fontsize=8, color="black")

    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels([str(r) for r in resources])
    ax.set_xlabel("Time (hours)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    if t0 is not None:
        try:
            t0f = float(t0)
            ax.axvline(t0f, linestyle="--", linewidth=2.0)
            ax.text(t0f, len(resources) - 0.2, "t0", rotation=90, va="top", ha="right")
        except Exception:
            pass

    legend_jobs = sorted([j for j in job_color.keys() if j >= 0])
    handles = [Patch(facecolor=job_color[j], edgecolor="black", label=f"Job {j}") for j in legend_jobs]
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Color → Job")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_png}")


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))

    base_path = os.path.join(workdir, "baseline_solution.json")
    res_path = os.path.join(workdir, "reschedule_solution.json")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing baseline_solution.json at: {base_path}")

    base = _load(base_path)
    base_sched = base.get("schedule", [])

    # scenario id (nice output folder)
    sid = _extract_scenario_id(base)

    # output folder
    outdir = os.path.join(workdir, f"scenario{sid}_gantts")
    os.makedirs(outdir, exist_ok=True)

    # baseline plots
    _plot_gantt(
        base_sched,
        "machine",
        f"Scenario {sid} — Baseline (Machine Gantt)",
        os.path.join(outdir, "gantt_machine_baseline.png"),
        t0=None
    )
    _plot_gantt(
        base_sched,
        "station",
        f"Scenario {sid} — Baseline (Station Gantt)",
        os.path.join(outdir, "gantt_station_baseline.png"),
        t0=None
    )

    # reschedule plots (optional)
    if os.path.exists(res_path):
        res = _load(res_path)
        res_sched = res.get("schedule", [])
        t0 = res.get("t0", None)

        _plot_gantt(
            res_sched,
            "machine",
            f"Scenario {sid} — Reschedule (Machine Gantt)",
            os.path.join(outdir, f"gantt_machine_reschedule_s{sid}.png"),
            t0=t0
        )
        _plot_gantt(
            res_sched,
            "station",
            f"Scenario {sid} — Reschedule (Station Gantt)",
            os.path.join(outdir, f"gantt_station_reschedule_s{sid}.png"),
            t0=t0
        )
    else:
        print(f"ℹ️ reschedule_solution.json not found (ok). Only baseline plots were generated.")


if __name__ == "__main__":
    main()
