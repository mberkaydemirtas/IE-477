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


def _extract_scenario_id_from_meta(meta: dict) -> str:
    """
    Tries to infer scenario id (e.g., '04') from meta fields.
    Looks for:
      - meta['scenario'] like 'scenario_04_urgent_job_shock.json'
      - meta['scenario_id'] like '04'
    Falls back to 'xx'.
    """
    # 1) direct id
    sid = meta.get("scenario_id", None)
    if sid is not None:
        m = re.search(r"\d+", str(sid))
        if m:
            return m.group(0).zfill(2)

    # 2) scenario filename/name
    s = meta.get("scenario", None)
    if isinstance(s, str) and s.strip():
        # try scenario_04... pattern
        m = re.search(r"scenario[_\- ]*(\d+)", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).zfill(2)
        # fallback: any number sequence
        m2 = re.search(r"(\d+)", s)
        if m2:
            return m2.group(1).zfill(2)

    return "xx"


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

    # t0 line (reschedule)
    if t0 is not None:
        ax.axvline(float(t0), linestyle="--", linewidth=2.0)
        ax.text(float(t0), len(resources) - 0.2, "t0", rotation=90, va="top", ha="right")

    # legend: Color -> Job
    legend_jobs = sorted([j for j in job_color.keys() if j >= 0])
    handles = [Patch(facecolor=job_color[j], edgecolor="black", label=f"Job {j}") for j in legend_jobs]
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

    # infer scenario id from reschedule first, then baseline
    sid = _extract_scenario_id_from_meta(res)
    if sid == "xx":
        sid = _extract_scenario_id_from_meta(base)

    outdir = f"scenario{sid}_gantts"
    os.makedirs(outdir, exist_ok=True)

    suffix = f"_s{sid}"

    # t0 for reschedule line
    t0 = res.get("t0", None)

    # ONLY 4 FILES
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

    _plot_gantt(
        res_sched,
        "machine",
        f"Scenario {sid} — Reschedule (Machine Gantt)",
        os.path.join(outdir, f"gantt_machine_reschedule{suffix}.png"),
        t0=t0
    )
    _plot_gantt(
        res_sched,
        "station",
        f"Scenario {sid} — Reschedule (Station Gantt)",
        os.path.join(outdir, f"gantt_station_reschedule{suffix}.png"),
        t0=t0
    )


if __name__ == "__main__":
    main()
