#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import shutil
import subprocess
from datetime import datetime, timezone

# baseline/reschedule entry points (existing)
from solver_core import make_base_data, solve_baseline, solve_reschedule

# import helper pieces to run "disruption-only" reschedule WITHOUT adding urgent job
from solver_core import (
    normalize_old_solution,
    compute_t0_from_plan_start,
    classify_ops_by_t0,
    add_split_remainders_for_running_ops,
    build_model,
    op_to_job_map,
    extract_assignment_for_ops,
)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _scenario_tag(scn: dict, scenario_path: str) -> str:
    sid = scn.get("scenario_id", None)
    if sid is not None:
        s = str(sid).strip()
        return f"s{s.zfill(2)}" if s.isdigit() else f"s_{s}"

    base = os.path.basename(scenario_path).replace(".json", "")
    parts = base.split("_")
    for p in parts:
        if p.isdigit():
            return f"s{p.zfill(2)}"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"s_{ts}"


def _pick_scenario_name(scn: dict, scenario_path: str) -> str:
    return (
        scn.get("scenario_name")
        or scn.get("name")
        or scn.get("title")
        or os.path.basename(scenario_path)
    )


def _load_urgent_payload_from_path(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        workdir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(workdir, path)
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"urgent_payload_path not found: {path}")

    uj = _load_json(path)
    # allow file to be either {"urgent_job": {...}} OR directly {...}
    if isinstance(uj, dict) and "urgent_job" in uj:
        return uj
    if isinstance(uj, dict) and "job_id" in uj:
        return {"urgent_job": uj}
    return uj if isinstance(uj, dict) else {}


def _pick_urgent_payload(scn: dict) -> dict:
    """
    Supports multiple formats:
    - "urgent_job": {...}
    - "urgent_payload": {...}
    - "urgent_payload_path": "scenarios/urgent_x.json"
    - legacy nested: "reschedule": {"urgent_payload_path": "..."}
    """
    if "urgent_job" in scn and isinstance(scn["urgent_job"], dict):
        return {"urgent_job": scn["urgent_job"]}

    if "urgent_payload" in scn and isinstance(scn["urgent_payload"], dict):
        return scn["urgent_payload"]

    if "urgent_payload_path" in scn:
        return _load_urgent_payload_from_path(scn.get("urgent_payload_path"))

    if "reschedule" in scn and isinstance(scn["reschedule"], dict):
        rp = scn["reschedule"]
        if "urgent_payload" in rp and isinstance(rp["urgent_payload"], dict):
            return rp["urgent_payload"]
        if "urgent_payload_path" in rp:
            return _load_urgent_payload_from_path(rp.get("urgent_payload_path"))

    return {}


def _validate_urgent_payload_if_present(urgent_payload: dict, scenario_path: str):
    if not urgent_payload:
        return
    uj = urgent_payload.get("urgent_job", urgent_payload)
    if not isinstance(uj, dict) or "job_id" not in uj:
        raise ValueError(
            f"Urgent job payload is missing required key: 'job_id'\n"
            f"File: {scenario_path}"
        )


def _pick_disruptions(scn: dict):
    """
    Supports:
    - "disruptions": {...}
    - legacy nested: "reschedule": {"unavailable_machines": [...], "unavailable_stations": [...]}
    - flat keys
    """
    if "disruptions" in scn and isinstance(scn["disruptions"], dict):
        d = scn["disruptions"]
        return d.get("unavailable_machines", []), d.get("unavailable_stations", [])

    if "reschedule" in scn and isinstance(scn["reschedule"], dict):
        r = scn["reschedule"]
        return r.get("unavailable_machines", []), r.get("unavailable_stations", [])

    return scn.get("unavailable_machines", []), scn.get("unavailable_stations", [])


def _pick_mode(scn: dict) -> str:
    if "mode" in scn:
        return scn.get("mode") or "optimize"
    if "reschedule" in scn and isinstance(scn["reschedule"], dict):
        return scn["reschedule"].get("mode", "optimize")
    return "optimize"


def _pick_outputs(scn: dict, workdir: str):
    base_out = scn.get("baseline_output") or "baseline_solution.json"
    res_out = scn.get("reschedule_output") or "reschedule_solution.json"

    if not os.path.isabs(base_out):
        base_out = os.path.join(workdir, base_out)
    if not os.path.isabs(res_out):
        res_out = os.path.join(workdir, res_out)

    canonical_base = os.path.join(workdir, "baseline_solution.json")
    canonical_res = os.path.join(workdir, "reschedule_solution.json")
    return base_out, res_out, canonical_base, canonical_res


def _run_plotter_and_archive(tag: str, workdir: str):
    plot_script = os.path.join(workdir, "plot_gantt_all.py")
    if not os.path.exists(plot_script):
        print(f"⚠️ plot_gantt_all.py not found at: {plot_script}")
        return

    print("📊 Generating Gantt charts (with legend)...")
    subprocess.run([sys.executable, plot_script], cwd=workdir, check=True)

    outputs = [
        "gantt_machine_baseline.png",
        "gantt_station_baseline.png",
        "gantt_machine_reschedule.png",
        "gantt_station_reschedule.png",
    ]

    for fn in outputs:
        src = os.path.join(workdir, fn)
        if os.path.exists(src):
            name, ext = os.path.splitext(fn)
            dst = os.path.join(workdir, f"{name}_{tag}{ext}")
            shutil.copyfile(src, dst)
            print(f"✅ Saved: {dst}")


def solve_reschedule_disruption_only(
    data_base: dict,
    old_solution: dict,
    unavailable_machines=None,
    unavailable_stations=None,
    mode="optimize",
):
    """
    Reschedule WITHOUT adding urgent job.
    Purpose: scenarios like 2,3,6 where only disruptions exist.

    - Computes t0 from plan_start_iso + plan_calendar
    - Classifies ops at t0: done / running / free
    - Adds remainder ops for running ones (cont)
    - Builds and solves the same model via build_model(...)
    """
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []

    old_solution = normalize_old_solution(old_solution)

    if "plan_start_iso" not in old_solution:
        raise ValueError("baseline_solution.json is missing plan_start_iso. Re-run baseline.")

    plan_calendar = old_solution.get("plan_calendar", {})
    t0 = compute_t0_from_plan_start(old_solution["plan_start_iso"], plan_calendar=plan_calendar)

    I = data_base["I"]
    I_done, I_run, I_free = classify_ops_by_t0(I, old_solution["S_old"], old_solution["C_old"], t0)

    # Only split running ops (so remainder can be reassigned if resource is broken)
    data2 = add_split_remainders_for_running_ops(data_base, I_run, t0, old_solution)

    model, x, y, S, C, Cw, Cf, T, Tmax, Cmax, keep = build_model(
        name="Reschedule_DisruptionOnly",
        data=data2,
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        freeze_done_ops=I_done,
        running_ops=I_run,
        mode=mode,
        t0=t0,
        free_ops=I_free,
        old_solution=old_solution
    )
    model.optimize()

    if model.SolCount == 0:
        model.computeIIS()
        model.write("iis.ilp")
        raise RuntimeError(f"No feasible reschedule (disruption-only). Status={model.Status}")

    keep_out = {}
    if mode == "optimize" and keep is not None:
        for i in I_run:
            keep_out[int(i)] = int(round(keep[int(i)].X))

    job_of = op_to_job_map(data2["O_j"])
    chosen_m, chosen_l = extract_assignment_for_ops(data2["I"], data2["M_i"], data2["L_i"], x, y)

    rem_map = data2.get("rem_map", {})
    rem_to_orig = {int(v): int(k) for k, v in rem_map.items()}

    schedule_new = []
    for i in data2["I"]:
        i = int(i)
        label = str(i)
        if i in rem_to_orig:
            label = f"{rem_to_orig[i]}(cont)"
        schedule_new.append({
            "op_id": i,
            "op_label": label,
            "job_id": int(job_of.get(i, -1)),
            "start": float(S[i].X),
            "finish": float(C[i].X),
            "machine": chosen_m[i],
            "station": chosen_l[i],
        })

    changed_ops = []
    for i in data_base["I"]:
        old_s = old_solution["S_old"].get(int(i), None)
        old_c = old_solution["C_old"].get(int(i), None)
        if old_s is None:
            continue
        new_s = float(S[i].X) if i in S else None
        new_c = float(C[i].X) if i in C else None
        if new_s is None or new_c is None:
            continue
        if abs(new_s - old_s) > 1e-6 or abs(new_c - old_c) > 1e-6:
            changed_ops.append({
                "op_id": int(i),
                "old_start": float(old_s), "old_finish": float(old_c),
                "new_start": float(new_s), "new_finish": float(new_c),
            })

    return {
        "t0": float(t0),
        "sets": {"I_done": I_done, "I_run": I_run, "I_free": I_free},
        "objective": {"T_max": float(Tmax.X), "C_max": float(Cmax.X)},
        "keep_decisions": keep_out,
        "schedule": schedule_new,
        "changed_ops": changed_ops,
        "note": "Disruption-only reschedule (no urgent job added)."
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_reschedule.py scenarios/<scenario>.json")
        sys.exit(1)

    scenario_path = sys.argv[1]
    scn = _load_json(scenario_path)

    workdir = os.path.dirname(os.path.abspath(__file__))

    scenario_name = _pick_scenario_name(scn, scenario_path)
    plan_start_iso = scn.get("plan_start_iso", "2025-12-18T05:00:00+00:00")
    plan_calendar = scn.get("plan_calendar", {"utc_offset": "+03:00"})

    overrides = scn.get("overrides", {})
    data = make_base_data(overrides=overrides)

    tag = _scenario_tag(scn, scenario_path)
    baseline_out, reschedule_out, canonical_base, canonical_res = _pick_outputs(scn, workdir)

    # --- BASELINE ---
    baseline = solve_baseline(data, plan_start_iso=plan_start_iso)
    baseline["plan_calendar"] = plan_calendar
    baseline["scenario_name"] = scenario_name

    _save_json(baseline_out, baseline)
    _save_json(canonical_base, baseline)

    print(f"✅ Baseline saved: {baseline_out}")
    print("scenario:", scenario_name)
    print("plan_start_iso (UTC):", baseline["plan_start_iso"])
    print("objective:", baseline["objective"])

    # --- RESCHEDULE ---
    urgent_payload = _pick_urgent_payload(scn)
    _validate_urgent_payload_if_present(urgent_payload, scenario_path)

    unavailable_machines, unavailable_stations = _pick_disruptions(scn)
    mode = _pick_mode(scn)

    if urgent_payload:
        # normal urgent reschedule
        res = solve_reschedule(
            data_base=data,
            old_solution=baseline,
            urgent_payload=urgent_payload,
            unavailable_machines=unavailable_machines,
            unavailable_stations=unavailable_stations,
            mode=mode
        )
        res["note"] = "Urgent-job reschedule."
    else:
        # disruption-only reschedule (NO urgent added)
        res = solve_reschedule_disruption_only(
            data_base=data,
            old_solution=baseline,
            unavailable_machines=unavailable_machines,
            unavailable_stations=unavailable_stations,
            mode=mode
        )

    res["scenario_name"] = scenario_name
    res["mode"] = mode
    res["disruptions"] = {
        "unavailable_machines": unavailable_machines,
        "unavailable_stations": unavailable_stations
    }

    _save_json(reschedule_out, res)
    _save_json(canonical_res, res)

    print(f"✅ Reschedule saved: {reschedule_out}")
    print("t0:", res.get("t0"))
    print("objective:", res.get("objective"))

    _run_plotter_and_archive(tag=tag, workdir=workdir)


if __name__ == "__main__":
    main()
